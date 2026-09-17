"""
Core estimation endpoint implementing the DietAI24 architecture:
Input (photo / description) -> Multimodal Recognition -> Multi-Query RAG Retrieval
-> Ambiguity Resolution -> Nutrient Estimation.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Form, UploadFile
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db
from app.rag.retriever import RAGRetriever
from app.services import estimation_service, vision_service
from app.services.vision_service import VisionRecognitionError

router = APIRouter(prefix="/estimate", tags=["estimate"])

# Shared singleton RAG retriever
_retrieval = RAGRetriever()


@router.post("", response_model=schemas.EstimateResponse)
async def estimate(
    image: Optional[UploadFile] = None,
    debug_description: Optional[str] = Form(
        default=None, description="Bypass vision AI with a plain-text description (offline / testing use)."
    ),
    portion_grams: Optional[float] = Form(default=None, description="Override the default portion size in grams."),
    clarification_attribute: Optional[str] = Form(
        default=None, description="Attribute key from a previous needs_clarification response."
    ),
    clarification_answer: Optional[str] = Form(
        default=None, description="Selected option from a previous needs_clarification response."
    ),
    db: Session = Depends(get_db),
):
    # --- Step 1: Clarification resolution round-trip ---
    if clarification_attribute and clarification_answer:
        food = estimation_service.resolve_from_clarification(db, clarification_attribute, clarification_answer)
        if not food:
            return schemas.EstimateResponse(
                status="no_match",
                notes=f"No database match for {clarification_attribute} = {clarification_answer}.",
            )
        return _build_success_response(food, portion_grams)

    # --- Step 2: Visual recognition ---
    image_bytes = await image.read() if image is not None else None
    try:
        description = vision_service.recognize_food(image_bytes=image_bytes, debug_description=debug_description)
    except VisionRecognitionError as e:
        return schemas.EstimateResponse(status="no_match", notes=str(e))

    # --- Step 3: DietAI24 RAG Retrieval ---
    if not _retrieval.is_indexed():
        return schemas.EstimateResponse(
            status="no_match",
            notes="Vector index is empty — run python -m scripts.setup_data first to initialize the database.",
        )

    candidates = _retrieval.search(description, top_k=5)
    if not candidates:
        return schemas.EstimateResponse(
            status="no_match",
            notes=f"No nutrition database match found for description: '{description}'",
        )

    candidate_schemas = [
        schemas.CandidateFood(
            food_code=c["food_code"],
            food_name=c["food_name"],
            region_variant=c.get("region_variant", "Standard"),
            food_group=c.get("food_group"),
            similarity=c["similarity"],
            matched_queries=c.get("matched_queries", []),
        )
        for c in candidates
    ]

    top = candidates[0]

    # --- Step 4: Ambiguity check (regional style / invisible ingredient) ---
    attribute_key, variant_rows = estimation_service.get_ambiguous_set(db, top["food_name"])
    if len(variant_rows) > 1:
        clarification = estimation_service.build_clarification(attribute_key, variant_rows)
        return schemas.EstimateResponse(
            status="needs_clarification",
            candidates=candidate_schemas,
            clarification=clarification,
            notes=f"Recognized as '{description}', matched to '{top['food_name']}' — attribute selection required.",
        )

    if len(variant_rows) == 1:
        food = variant_rows[0]
    else:
        food = db.query(models.Food).filter(models.Food.food_code == top["food_code"]).first()

    if not food:
        return schemas.EstimateResponse(
            status="no_match",
            candidates=candidate_schemas,
            notes="Matched food entry code was not found in the SQL database.",
        )

    return _build_success_response(
        food=food,
        portion_grams=portion_grams,
        description=description,
        similarity=top.get("similarity"),
        candidates=candidate_schemas,
    )


def _build_success_response(
    food: models.Food,
    portion_grams: Optional[float],
    description: str | None = None,
    similarity: float | None = None,
    candidates: list[schemas.CandidateFood] | None = None,
) -> schemas.EstimateResponse:
    grams = portion_grams or food.typical_portion_grams or 100.0
    nutrients = estimation_service.compute_nutrients(food, grams)
    recognized = schemas.RecognizedFood(
        food_code=food.food_code,
        food_name=food.food_name,
        region_variant=food.region_variant or "Standard",
        matched_confidence=similarity if similarity is not None else 1.0,
        portion_unit=food.typical_portion_unit or "plate",
        portion_grams=grams,
        nutrients=nutrients,
    )
    return schemas.EstimateResponse(
        status="ok",
        foods=[recognized],
        candidates=candidates or [],
        totals=nutrients,
        notes=f"Recognized as: {description}" if description else None,
    )
