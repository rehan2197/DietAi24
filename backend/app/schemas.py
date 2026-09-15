from typing import Optional

from pydantic import BaseModel, Field


class NutrientBreakdown(BaseModel):
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    calcium_mg: float
    iron_mg: float


class RecognizedFood(BaseModel):
    food_code: str
    food_name: str
    region_variant: str
    matched_confidence: float = Field(..., ge=0.0, le=1.0)
    portion_unit: str
    portion_grams: float
    nutrients: NutrientBreakdown


class ClarificationQuestion(BaseModel):
    """
    A follow-up question for an attribute that's invisible to the camera
    (milk-fat %, oil vs ghee, polished vs unpolished rice, etc.) — the
    "Future Features" clarification-query concept from the architecture.
    """
    attribute: str
    question: str
    options: list[str]


class EstimateResponse(BaseModel):
    status: str  # "ok" | "needs_clarification" | "no_match"
    foods: list[RecognizedFood] = []
    totals: Optional[NutrientBreakdown] = None
    clarification: Optional[ClarificationQuestion] = None
    notes: Optional[str] = None


class ClarificationAnswer(BaseModel):
    attribute: str
    answer: str


class FoodOut(BaseModel):
    food_code: str
    food_name: str
    region_variant: str
    food_group: str
    description: Optional[str] = None
    nutrients: NutrientBreakdown
    typical_portion_unit: str
    typical_portion_grams: float

    class Config:
        from_attributes = True
