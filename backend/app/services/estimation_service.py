"""
Estimation service for DietAI24.
Handles ambiguity detection (regional variants / invisible attributes),
follow-up clarification generation, and deterministic nutrient scaling.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app import models, schemas
from app.rag.prompts import DEFAULT_CLARIFICATION_PROMPTS

# Cross-dish attribute groups where visually identical foods differ by hidden attributes
CLARIFICATION_GROUPS: dict[str, list[str]] = {
    "milk": ["Full Cream Milk", "Toned Milk", "Skimmed Milk"],
    "rice": ["Steamed White Rice", "Steamed Brown Rice"],
}

FOOD_NAME_TO_GROUP: dict[str, str] = {
    member: group_key for group_key, members in CLARIFICATION_GROUPS.items() for member in members
}


def get_ambiguous_set(db: Session, food_name: str) -> tuple[str, list[models.Food]]:
    """
    Checks if the retrieved food name corresponds to multiple variants or invisible-ingredient groups.
    Returns (attribute_key, list_of_matching_food_records).
    """
    group_key = FOOD_NAME_TO_GROUP.get(food_name)
    if group_key:
        rows = db.query(models.Food).filter(models.Food.food_name.in_(CLARIFICATION_GROUPS[group_key])).all()
        return group_key, rows

    rows = db.query(models.Food).filter(models.Food.food_name == food_name).all()
    return food_name, rows


def build_clarification(attribute_key: str, variants: list[models.Food]) -> schemas.ClarificationQuestion:
    """Generates an unambiguous multiple-choice question for the frontend."""
    question = DEFAULT_CLARIFICATION_PROMPTS.get(
        attribute_key, f"Multiple preparations of {attribute_key} were found — which one matches?"
    )
    options = sorted({v.region_variant for v in variants if v.region_variant})
    return schemas.ClarificationQuestion(attribute=attribute_key, question=question, options=options)


def resolve_from_clarification(db: Session, attribute_key: str, answer: str) -> models.Food | None:
    """Resolves an ambiguous choice directly when the user selects a clarification option."""
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
    """
    Calculates nutrient values scaled by actual portion weight.
    Formula from DietAI24: N = v * (grams / 100), where v is nutrient amount per 100g.
    """
    factor = portion_grams / 100.0
    return schemas.NutrientBreakdown(
        calories_kcal=round((food.calories_kcal or 0.0) * factor, 1),
        protein_g=round((food.protein_g or 0.0) * factor, 1),
        carbs_g=round((food.carbs_g or 0.0) * factor, 1),
        fat_g=round((food.fat_g or 0.0) * factor, 1),
        fiber_g=round((food.fiber_g or 0.0) * factor, 1),
        calcium_mg=round((food.calcium_mg or 0.0) * factor, 1),
        iron_mg=round((food.iron_mg or 0.0) * factor, 1),
    )


def sum_nutrients(items: list[schemas.NutrientBreakdown]) -> schemas.NutrientBreakdown:
    """Sums nutrient breakdown across all recognized foods in a meal."""
    totals = schemas.NutrientBreakdown()
    for n in items:
        totals.calories_kcal += n.calories_kcal
        totals.protein_g += n.protein_g
        totals.carbs_g += n.carbs_g
        totals.fat_g += n.fat_g
        totals.fiber_g += n.fiber_g
        totals.calcium_mg += n.calcium_mg
        totals.iron_mg += n.iron_mg

    totals.calories_kcal = round(totals.calories_kcal, 1)
    totals.protein_g = round(totals.protein_g, 1)
    totals.carbs_g = round(totals.carbs_g, 1)
    totals.fat_g = round(totals.fat_g, 1)
    totals.fiber_g = round(totals.fiber_g, 1)
    totals.calcium_mg = round(totals.calcium_mg, 1)
    totals.iron_mg = round(totals.iron_mg, 1)
    return totals
