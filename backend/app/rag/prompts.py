"""
Centralized prompts for the DietAI24 framework.
Matches the methodology described in DietAI24 (Nature Communications Medicine, 2025):
- Multimodal visual food description
- Multi-query expansion for RAG retrieval
- Food code inference and portion size selection
- Ambiguity clarification prompts
"""
from __future__ import annotations

VISION_FOOD_IDENTIFICATION_PROMPT = (
    "You are an expert dietary assessment AI. Identify all food and beverage item(s) "
    "visible in this photo. Respond with a concise, plain-text description naming each dish "
    "and distinct component clearly (e.g., 'a bowl of chicken biryani with sliced boiled egg, "
    "and a small side of cucumber raita').\n"
    "Do NOT estimate portion size, calories, or nutritional numbers here. "
    "Only describe what foods and ingredients are visually present."
)

MULTIQUERY_EXPANSION_PROMPT = (
    "Given the following description of a meal from a food photo:\n"
    "'{description}'\n\n"
    "Generate {count} distinct search queries to look up each food item in a standardized "
    "food composition database. Output one query per line, focusing on dish names, key ingredients, "
    "and common culinary preparation terms without punctuation or bullets."
)

PORTION_SELECTION_PROMPT = (
    "Given the recognized food item '{food_name}' and the following standardized portion options:\n"
    "{portion_options}\n\n"
    "Based on the visual appearance in the food image, select the most accurate portion option. "
    "If none matches exactly, specify the closest standard descriptor."
)

DEFAULT_CLARIFICATION_PROMPTS: dict[str, str] = {
    "Chicken Biryani": "Which regional style is this biryani?",
    "Toor Dal": "Is this dal lightly tempered, or rich with ghee tadka?",
    "milk": "Is this milk toned, full-fat, or skimmed?",
    "rice": "Is this rice polished (white) or unpolished (brown)?",
    "Curd/Dahi": "Is this whole milk curd, low-fat curd, or sweetened yogurt?",
    "Paneer": "Is this standard full-fat cottage cheese or low-fat paneer?",
}
