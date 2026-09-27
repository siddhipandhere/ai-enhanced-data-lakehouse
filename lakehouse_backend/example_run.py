"""
End-to-end example: proves the backend pipeline runs start to finish.

    python example_run.py

Creates a small synthetic product dataset, pushes it through
Bronze -> Silver -> Gold, builds a FAISS index over the product
names, and runs one query through the full agent pipeline.
"""

import json
from pathlib import Path

import pandas as pd

from ingestion.loaders import ingest_file
from medallion.silver import clean_and_promote, load_silver_as_pandas
from medallion.gold import build_gold_table, load_gold_table
from embeddings.vector_store import build_index
from agents.orchestrator import handle_request

SAMPLE_CSV = Path("sample_products.csv")


def make_sample_data():
    df = pd.DataFrame({
        "id": range(1, 11),
        "product_name": [
            "Wireless bluetooth headphones", "Running shoes for men", "Stainless steel water bottle",
            "Organic cotton t-shirt", "Noise cancelling earbuds", "Trail running sneakers",
            "Insulated travel mug", "Yoga mat non-slip", "Leather laptop bag", "Smart fitness watch",
        ],
        "category": ["Electronics", "Footwear", "Home", "Apparel", "Electronics",
                     "Footwear", "Home", "Fitness", "Accessories", "Electronics"],
        "price": [79.99, 59.99, 19.99, 24.99, 129.99, 89.99, 14.99, 29.99, 49.99, 199.99],
        "region": ["North", "South", "North", "East", "West", "South", "North", "East", "West", "North"],
    })
    df.to_csv(SAMPLE_CSV, index=False)
    return SAMPLE_CSV


def main():
    print("1. Creating sample dataset...")
    csv_path = make_sample_data()

    print("2. Ingesting into Bronze layer...")
    bronze_path = ingest_file(csv_path, uploaded_by="demo_user")

    print("3. Promoting to Silver layer...")
    silver_path = clean_and_promote(bronze_path)

    print("4. Building Gold table...")
    # silver_path is a Delta table DIRECTORY, not a CSV file -- the old
    # pd.read_csv(silver_path) crashed here.
    silver_df = load_silver_as_pandas(silver_path)
    build_gold_table(silver_df, table_name="products")
    gold_df = load_gold_table("products")
    print(gold_df)

    print("5. Building FAISS vector index over product names...")
    build_index(gold_df, text_column="product_name", table_name="products")

    print("6. Running a query through the full agent pipeline...")
    result = handle_request("Find products similar to wireless headphones", gold_df,
                            table_name="products", text_column="product_name")

    print("\n--- Result ---")
    print("Plan:", result["plan"])
    print("Summary:", result["summary"])
    print("Results:\n", result["results"])
    print("Analysis:", json.dumps(result["analysis"], indent=2, default=str))


if __name__ == "__main__":
    main()
