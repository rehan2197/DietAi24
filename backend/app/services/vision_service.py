"""
Vision AI Service for the DietAI24 architecture.
Generates an objective plain-text description of food items present in an image.
Nutrient numbers and portion standards are strictly decoupled and resolved later via RAG.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings
from app.rag.prompts import VISION_FOOD_IDENTIFICATION_PROMPT

logger = logging.getLogger(__name__)


class VisionRecognitionError(Exception):
    pass


def recognize_food(image_bytes: bytes | None = None, debug_description: str | None = None) -> str:
    """
    Identifies food items visually in the photo.
    Returns a textual description (e.g. 'A plate of chicken biryani with boiled egg and raita').
    """
    # Offline / Mock / Debug path
    if image_bytes is None:
        if debug_description:
            return debug_description.strip()
        raise VisionRecognitionError("Either image_bytes or debug_description is required.")

    settings = get_settings()

    # Image provided, but no Gemini API key configured
    if not settings.gemini_api_key:
        if debug_description:
            return debug_description.strip()
        raise VisionRecognitionError(
            "No GEMINI_API_KEY configured in .env. "
            "Please provide a Gemini API key for live photo recognition, "
            "or use the debug_description field for offline testing."
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                VISION_FOOD_IDENTIFICATION_PROMPT,
            ],
        )
        return (response.text or "").strip()
    except ImportError as e:
        raise VisionRecognitionError("google-genai package is not installed.") from e
    except Exception as e:
        logger.error(f"Error during vision recognition call: {e}")
        raise VisionRecognitionError(f"Vision recognition failed: {str(e)}") from e
