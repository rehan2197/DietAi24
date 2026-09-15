from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/foods", tags=["foods"])

_retrieval = RetrievalService()


def _to_food_out(food: models.Food) -> schemas.FoodOut:
    return schemas.FoodOut(
        food_code=food.food_code,
        food_name=food.food_name,
        region_variant=food.region_variant,
        food_group=food.food_group,
        description=food.description,
        nutrients=schemas.NutrientBreakdown(
            calories_kcal=food.calories_kcal,
            protein_g=food.protein_g,
            carbs_g=food.carbs_g,
            fat_g=food.fat_g,
            fiber_g=food.fiber_g,
            calcium_mg=food.calcium_mg,
            iron_mg=food.iron_mg,
        ),
        typical_portion_unit=food.typical_portion_unit,
        typical_portion_grams=food.typical_portion_grams,
    )


@router.get("/search", response_model=list[schemas.FoodOut])
def search_foods(q: str, top_k: int = 5, db: Session = Depends(get_db)):
    """Semantic search over the food database — same retrieval path the estimate pipeline uses."""
    candidates = _retrieval.search(q, top_k=top_k)
    results = []
    for c in candidates:
        food = db.query(models.Food).filter(models.Food.food_code == c["food_code"]).first()
        if food:
            results.append(_to_food_out(food))
    return results


@router.get("/{food_code}", response_model=schemas.FoodOut)
def get_food(food_code: str, db: Session = Depends(get_db)):
    food = db.query(models.Food).filter(models.Food.food_code == food_code).first()
    if not food:
        raise HTTPException(status_code=404, detail="Food not found")
    return _to_food_out(food)
