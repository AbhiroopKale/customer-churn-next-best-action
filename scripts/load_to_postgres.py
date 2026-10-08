"""Load the batch-scoring CSVs into PostgreSQL.  Run: python -m scripts.load_to_postgres"""
import json

import pandas as pd

from src import config, database


def main():
    p = config.PROCESSED
    meta = json.loads((config.MODELS / "meta.json").read_text())
    counts = database.load_batch(
        database.get_engine(),
        pd.read_csv(p / "clean_customers.csv"), pd.read_csv(p / "predictions.csv"),
        pd.read_csv(p / "explanations.csv"), pd.read_csv(p / "recommendations.csv"), meta)
    print("Loaded rows:", counts)


if __name__ == "__main__":
    main()
