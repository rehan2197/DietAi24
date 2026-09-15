"""
One-time (or periodic) data loading: reads the IFCT-style CSV and portion
mapping CSV, populates the SQL database, and builds the vector index used
for retrieval. Re-running is safe — it clears and rebuilds both.

To swap in the real IFCT 2017 dataset later: replace app/data/ifct_sample.csv
with a file using the same column headers (or update the loader below) and
re-run this script — nothing else in the pipeline needs to change.
"""
from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy.orm import Session

from app import models
from app.database import Base, SessionLocal, engine
from app.services.retrieval_service import RetrievalService

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FOODS_CSV = DATA_DIR / "ifct_sample.csv"
PORTIONS_CSV = DATA_DIR / "portion_mapping.csv"


def load_foods_csv() -> list[dict]:
    with open(FOODS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_portions_csv() -> list[dict]:
    with open(PORTIONS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def populate_sql(db: Session, food_rows: list[dict], portion_rows: list[dict]) -> None:
    db.query(models.Food).delete()
    db.query(models.PortionUnit).delete()

    for row in food_rows:
        db.add(
            models.Food(
                food_code=row["food_code"],
                food_name=row["food_name"],
                region_variant=row["region_variant"],
                food_group=row["food_group"],
                description=row["description"],
                calories_kcal=float(row["calories_kcal"]),
                protein_g=float(row["protein_g"]),
                carbs_g=float(row["carbs_g"]),
                fat_g=float(row["fat_g"]),
                fiber_g=float(row["fiber_g"]),
                calcium_mg=float(row["calcium_mg"]),
                iron_mg=float(row["iron_mg"]),
                typical_portion_unit=row["typical_portion_unit"],
                typical_portion_grams=float(row["typical_portion_grams"]),
            )
        )

    for row in portion_rows:
        db.add(models.PortionUnit(unit_name=row["unit_name"], grams=float(row["grams"]), notes=row["notes"]))

    db.commit()


def build_index(food_rows: list[dict]) -> None:
    retrieval = RetrievalService()
    retrieval.index_foods(food_rows)


def run_full_ingest() -> None:
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    print("Loading CSV data...")
    food_rows = load_foods_csv()
    portion_rows = load_portions_csv()

    print(f"Populating SQL database ({len(food_rows)} foods, {len(portion_rows)} portion units)...")
    db = SessionLocal()
    try:
        populate_sql(db, food_rows, portion_rows)
    finally:
        db.close()

    print("Building vector index...")
    build_index(food_rows)

    print("Ingest complete.")


if __name__ == "__main__":
    run_full_ingest()
