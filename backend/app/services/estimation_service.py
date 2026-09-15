"""
Estimation service. Everything downstream of retrieval: deciding whether a
match is ambiguous enough to need a clarification question (the invisible-
ingredient fallback described in "Future Features"), and computing final
nutrient values. Computation is plain arithmetic — never AI-generated.

Two kinds of ambiguity are handled:
1. Same-dish variants: rows that share one food_name but differ by
   region_variant (e.g. Chicken Biryani — Hyderabadi / Awadhi / Kolkata).
2. Cross-dish attribute groups: visually similar/identical items whose
   food_name itself differs by the very attribute a camera can't see
   (e.g. "milk" — Full Cream Milk / Toned Milk / Skimmed Milk).
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app import models, schemas

CLARIFICATION_PROMPTS: dict[str, str] = {
    "Chicken Biryani": "Which regional style is this biryani?",
    "Toor Dal": "Is this dal lightly tempered, or rich with ghee tadka?",
    "milk": "Is this milk toned, full-fat, or skimmed?",
    "rice": "Is this rice polished (white) or unpolished (brown)?",
}

# Cross-dish attribute groups: group_key -> list of food_name members
CLARIFICATION_GROUPS: dict[str, list[str]] = {
    "milk": ["Full Cream Milk", "Toned Milk", "Skimmed Milk"],
    "rice": ["Steamed White Rice", "Steamed Brown Rice"],
}
FOOD_NAME_TO_GROUP: dict[str, str] = {
    member: group_key for group_key, members in CLARIFICATION_GROUPS.items() for member in members
}


def get_ambiguous_set(db: Session, food_name: str) -> tuple[str, list[models.Food]]:
    """
    Returns (attribute_key, variant_rows) for the given top-match food_name.
    attribute_key is either a group key ("milk", "rice") or the food_name
    itself, when the ambiguity is same-dish regional/attribute variants.
    """
    group_key = FOOD_NAME_TO_GROUP.get(food_name)
    if group_key:
        rows = db.query(models.Food).filter(models.Food.food_name.in_(CLARIFICATION_GROUPS[group_key])).all()
        return group_key, rows

    rows = db.query(models.Food).filter(models.Food.food_name == food_name).all()
    return food_name, rows


def build_clarification(attribute_key: str, variants: list[models.Food]) -> schemas.ClarificationQuestion:
    question = CLARIFICATION_PROMPTS.get(
        attribute_key, f"Multiple versions of {attribute_key} were found — which one matches?"
    )
    options = sorted({v.region_variant for v in variants})
    return schemas.ClarificationQuestion(attribute=attribute_key, question=question, options=options)


def resolve_from_clarification(db: Session, attribute_key: str, answer: str) -> models.Food | None:
    if attribute_key in CLARIFICATION_GROUPS:
        return (
            db.query(models.Food)
            .filter(
                models.Food.food_name.in_(CLARIFICATION_GROUPS[attribute_key]),
                models.Food.region_variant == answer,
            )
            .first()
        )
    return (
        db.query(models.Food)
        .filter(models.Food.food_name == attribute_key, models.Food.region_variant == answer)
        .first()
    )


def compute_nutrients(food: models.Food, portion_grams: float) -> schemas.NutrientBreakdown:
    """All source values in the dataset are per 100g — scale to the actual portion."""
    factor = portion_grams / 100.0
    return schemas.NutrientBreakdown(
        calories_kcal=round(food.calories_kcal * factor, 1),
        protein_g=round(food.protein_g * factor, 1),
        carbs_g=round(food.carbs_g * factor, 1),
        fat_g=round(food.fat_g * factor, 1),
        fiber_g=round(food.fiber_g * factor, 1),
        calcium_mg=round(food.calcium_mg * factor, 1),
        iron_mg=round(food.iron_mg * factor, 1),
    )


def sum_nutrients(items: list[schemas.NutrientBreakdown]) -> schemas.NutrientBreakdown:
    totals = schemas.NutrientBreakdown(
        calories_kcal=0, protein_g=0, carbs_g=0, fat_g=0, fiber_g=0, calcium_mg=0, iron_mg=0
    )
    for n in items:
        totals.calories_kcal += n.calories_kcal
        totals.protein_g += n.protein_g
        totals.carbs_g += n.carbs_g
        totals.fat_g += n.fat_g
        totals.fiber_g += n.fiber_g
        totals.calcium_mg += n.calcium_mg
        totals.iron_mg += n.iron_mg
    for field in totals.model_fields:
        setattr(totals, field, round(getattr(totals, field), 1))
    return totals
