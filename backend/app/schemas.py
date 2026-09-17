"""
Pydantic schemas for the DietAI24 API.
Defines structured request/response shapes for nutrient breakdown,
RAG retrieval candidates, clarification queries, and estimation results.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


class NutrientBreakdown(BaseModel):
    """
    Standard nutrient profile.
    Contains the core macronutrients and key minerals, with dynamic extension
    support to accommodate the full 65-nutrient FNDDS / IFCT profile.
    """
    calories_kcal: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    calcium_mg: float = 0.0
    iron_mg: float = 0.0

    # Extended nutrients (FNDDS 65-nutrient components: vitamins, minerals, lipids, etc.)
    additional_nutrients: dict[str, float] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class CandidateFood(BaseModel):
    """A candidate food item retrieved by the RAG search."""
    food_code: str
    food_name: str
    region_variant: str = "Standard"
    food_group: Optional[str] = None
    similarity: float = Field(..., ge=0.0, le=1.0)
    matched_queries: list[str] = []


class RecognizedFood(BaseModel):
    food_code: str
    food_name: str
    region_variant: str = "Standard"
    matched_confidence: float = Field(..., ge=0.0, le=1.0)
    portion_unit: str = "plate"
    portion_grams: float
    nutrients: NutrientBreakdown


class ClarificationQuestion(BaseModel):
    """
    Follow-up query for attributes invisible to the camera
    (e.g., regional style, milk fat percentage, oil vs ghee).
    """
    attribute: str
    question: str
    options: list[str]


class EstimateResponse(BaseModel):
    status: str  # "ok" | "needs_clarification" | "no_match"
    foods: list[RecognizedFood] = []
    candidates: list[CandidateFood] = []
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
