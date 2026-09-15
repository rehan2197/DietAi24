"""
Vision AI service. Its ONLY job is recognizing what food is in the photo and
describing it in plain text — it never returns a nutrient number itself.
Nutrient values always come later, from the database via the retrieval
service (see architecture: recognition and lookup are deliberately separate).

MOCK MODE: if no GEMINI_API_KEY is configured, `recognize_food` returns a
caller-supplied `debug_description` instead of calling the API. This keeps
the rest of the pipeline (retrieval, clarification, computation) fully
testable without live credentials.
"""
from __future__ import annotations

from app.config import get_settings

settings = get_settings()


class VisionRecognitionError(Exception):
    pass


def recognize_food(image_bytes: bytes | None = None, debug_description: str | None = None) -> str:
    """
    Returns a free-text description of the food(s) visible in the photo,
    e.g. "A plate of chicken biryani with raita on the side."
    """
    # No image supplied: only valid if a debug description was given (testing/dev path).
    if image_bytes is None:
        if debug_description:
            return debug_description
        raise VisionRecognitionError("Either image_bytes or debug_description is required.")

    # Image supplied but no API key configured: fall back to debug_description if given,
    # otherwise this is a hard error (can't recognize a real photo without the API).
    if not settings.gemini_api_key:
        if debug_description:
            return debug_description
        raise VisionRecognitionError(
            "No GEMINI_API_KEY configured. Set it in .env for live recognition, "
            "or pass debug_description for offline testing."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise VisionRecognitionError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from e

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = (
        "Identify the Indian food item(s) visible in this photo. Respond with a short, "
        "plain-text description naming the dish(es) — do not estimate nutrition, calories, "
        "or portion size. Just describe what food is present."
    )
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
    return response.text.strip()
