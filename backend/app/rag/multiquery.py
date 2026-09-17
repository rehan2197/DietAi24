"""
Multi-Query Generation for DietAI24 RAG retrieval.
Expands raw multimodal visual descriptions into multiple specialized search queries
to overcome vocabulary mismatch and maximize recall against the nutrition database.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.rag.prompts import MULTIQUERY_EXPANSION_PROMPT

logger = logging.getLogger(__name__)


class MultiQueryGenerator:
    def __init__(self, api_key: str = "", model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def generate_queries(self, description: str, count: int = 3) -> list[str]:
        """
        Takes a description and returns a list of diverse queries [q1, q2, ..., qm].
        Always includes the original cleaned description as the primary query.
        """
        if not description or not description.strip():
            return []

        cleaned_desc = description.strip()
        queries: list[str] = [cleaned_desc]

        # If live API key is available, use LLM for semantic query expansion
        if self.api_key:
            try:
                llm_queries = self._generate_with_llm(cleaned_desc, count)
                for q in llm_queries:
                    if q and q not in queries:
                        queries.append(q)
            except Exception as e:
                logger.warning(f"LLM query expansion failed, falling back to rule-based: {e}")

        # If we need more queries, complement with rule-based linguistic decomposition
        if len(queries) < count:
            rule_queries = self._rule_based_expansion(cleaned_desc)
            for q in rule_queries:
                if q and q not in queries:
                    queries.append(q)

        return queries[:count]

    def _generate_with_llm(self, description: str, count: int) -> list[str]:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        prompt = MULTIQUERY_EXPANSION_PROMPT.format(description=description, count=count)
        response = client.models.generate_content(
            model=self.model,
            contents=[prompt],
        )
        raw_text = response.text or ""
        lines = [line.strip().lstrip("1234567890.-* ") for line in raw_text.splitlines()]
        return [line for line in lines if line and len(line) > 2]

    @staticmethod
    def _rule_based_expansion(description: str) -> list[str]:
        """
        Linguistic decomposition for offline/zero-API execution.
        Splits composite dishes by conjunctions, prepositions, and meal separators.
        """
        expanded: list[str] = []

        # Split on conjunctions commonly separating dishes on a plate
        segments = re.split(
            r"\b(?:with|and|served with|accompanied by|along with|plus|side of|on the side)\b|,|;",
            description,
            flags=re.IGNORECASE,
        )

        for seg in segments:
            clean = seg.strip()
            # Remove leading articles and noise words
            clean = re.sub(r"^(?:a|an|the|some|fresh|plate of|bowl of|portion of|cup of|glass of)\s+", "", clean, flags=re.IGNORECASE).strip()
            if clean and len(clean) >= 3 and clean.lower() not in [e.lower() for e in expanded]:
                expanded.append(clean)

        return expanded
