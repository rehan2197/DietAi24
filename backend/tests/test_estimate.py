"""
End-to-end tests using debug_description to bypass the live vision API —
these exercise retrieval, clarification, and nutrient computation exactly
as a real request would, without needing a Gemini API key.

Run with: pytest -v
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ingest_service import run_full_ingest

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_data():
    run_full_ingest()
    yield


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_unambiguous_food_returns_ok():
    resp = client.post("/api/v1/estimate", data={"debug_description": "steamed idli"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["foods"][0]["food_name"] == "Idli"
    assert body["totals"]["calories_kcal"] > 0


def test_ambiguous_biryani_needs_clarification():
    resp = client.post("/api/v1/estimate", data={"debug_description": "a plate of chicken biryani"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "needs_clarification"
    assert body["clarification"]["attribute"] == "Chicken Biryani"
    assert set(body["clarification"]["options"]) == {"Hyderabadi", "Awadhi/Lucknowi", "Kolkata"}


def test_clarification_round_trip_resolves_to_exact_variant():
    resp = client.post(
        "/api/v1/estimate",
        data={
            "clarification_attribute": "Chicken Biryani",
            "clarification_answer": "Kolkata",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["foods"][0]["region_variant"] == "Kolkata"
    # 149 kcal/100g * 300g portion / 100 = 447.0
    assert body["totals"]["calories_kcal"] == pytest.approx(447.0, abs=0.5)


def test_milk_group_ambiguity_across_different_food_names():
    resp = client.post("/api/v1/estimate", data={"debug_description": "a glass of milk"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "needs_clarification"
    assert body["clarification"]["attribute"] == "milk"
    assert set(body["clarification"]["options"]) == {"Full-fat", "Toned", "Skimmed"}


def test_portion_override_scales_nutrients():
    resp = client.post(
        "/api/v1/estimate",
        data={"debug_description": "steamed idli", "portion_grams": "200"},
    )
    body = resp.json()
    assert body["foods"][0]["portion_grams"] == 200.0
    # 39 kcal/100g * 200g / 100 = 78.0
    assert body["totals"]["calories_kcal"] == pytest.approx(78.0, abs=0.5)


def test_foods_search_endpoint():
    resp = client.get("/api/v1/foods/search", params={"q": "dal tadka"})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) > 0


def test_no_gemini_key_and_no_debug_description_fails_gracefully():
    resp = client.post("/api/v1/estimate", data={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_match"
