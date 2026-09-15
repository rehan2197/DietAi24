"""
Run once (and again whenever the dataset changes):
    python -m scripts.setup_data

Creates the database tables, loads the food + portion CSVs into SQL, and
builds the Chroma vector index used for retrieval.
"""
from app.services.ingest_service import run_full_ingest

if __name__ == "__main__":
    run_full_ingest()
