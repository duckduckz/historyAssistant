from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, Generator

from elasticsearch import Elasticsearch, helpers

DEFAULT_ES_HOST = "localhost"
DEFAULT_ES_PORT = 9200
INDEX_NAME = "televic.history.location"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(
    SCRIPT_DIR, "../../../data/historyLocations/processed/fact_history_locations_v1.csv"
)

BOOLEAN_AS_INT_FIELDS = {
    "location_changed",
    "is_gap",
}

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "locationUpdateId":     {"type": "integer"},
            "personId":             {"type": "integer"},
            "personFirstName":      {"type": "keyword"},
            "personType":           {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "timestamp":            {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"},
            "altitude":             {"type": "float"},
            "accuracy":             {"type": "float"},
            "area":                 {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "areaId":               {"type": "integer"},
            "ward":                 {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "wardId":               {"type": "float"},
            "passage":              {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "passageId":            {"type": "float"},
            "zoneId":               {"type": "keyword"},
            "zoneName":             {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "time_gap_minutes":     {"type": "float"},
            "location_changed":     {"type": "integer"},
            "is_gap":               {"type": "integer"},
            "episode_id":           {"type": "integer"},
            "movement_status":      {"type": "keyword"},
            "location_in_episode":  {"type": "integer"},
            "movement_detail":      {"type": "keyword"},
            "prev_location":        {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        }
    }
}


def read_csv_and_generate_docs(csv_path: str) -> Generator[Dict[str, Any], None, None]:
    """Read location CSV and generate documents for Elasticsearch."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = {}
            for key, value in row.items():
                if value is None or (isinstance(value, str) and value.strip() in ("", "nan", "NaN")):
                    doc[key] = None
                elif key in BOOLEAN_AS_INT_FIELDS:
                    upper = value.strip().upper()
                    if upper == "TRUE":
                        doc[key] = 1
                    elif upper == "FALSE":
                        doc[key] = 0
                    else:
                        try:
                            doc[key] = int(value)
                        except (ValueError, TypeError):
                            doc[key] = 0
                else:
                    try:
                        if "." in value:
                            doc[key] = float(value)
                        else:
                            doc[key] = int(value)
                    except (ValueError, TypeError):
                        doc[key] = value

            yield {"_index": INDEX_NAME, "_source": doc}


def count_csv_rows(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # subtract header row


def main() -> None:
    parser = argparse.ArgumentParser(description="Load location CSV data into Elasticsearch")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="Path to CSV file")
    parser.add_argument("--es-host", default=DEFAULT_ES_HOST, help="Elasticsearch host")
    parser.add_argument("--es-port", type=int, default=DEFAULT_ES_PORT, help="Elasticsearch port")
    parser.add_argument("--recreate-index", action="store_true", help="Delete and recreate the index")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    print(f"Connecting to Elasticsearch at {args.es_host}:{args.es_port}...")

    es = Elasticsearch([f"http://{args.es_host}:{args.es_port}"])

    if not es.ping():
        print("Error: Could not connect to Elasticsearch")
        return

    print("Connected successfully!")

    if args.recreate_index and es.indices.exists(index=INDEX_NAME):
        print(f"Deleting existing index '{INDEX_NAME}'...")
        es.indices.delete(index=INDEX_NAME)

    if not es.indices.exists(index=INDEX_NAME):
        print(f"Creating index '{INDEX_NAME}' with mapping...")
        es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)

    total_docs = count_csv_rows(csv_path)
    print(f"Found {total_docs} records in CSV file")
    print(f"Indexing documents to '{INDEX_NAME}'...")

    success_count = 0
    error_count = 0

    for ok, response in helpers.streaming_bulk(
        es,
        read_csv_and_generate_docs(csv_path),
        chunk_size=500,
        raise_on_error=False,
    ):
        if ok:
            success_count += 1
        else:
            error_count += 1
            print(f"Error: {response}")

        if (success_count + error_count) % 5000 == 0:
            print(f"Progress: {success_count + error_count}/{total_docs} documents processed")

    print("\n" + "=" * 50)
    print("Indexing complete!")
    print(f"Successfully indexed: {success_count} documents")
    print(f"Errors: {error_count} documents")
    print("=" * 50)

    es.indices.refresh(index=INDEX_NAME)
    doc_count = es.count(index=INDEX_NAME)["count"]
    print(f"Total documents in index '{INDEX_NAME}': {doc_count}")


if __name__ == "__main__":
    main()
