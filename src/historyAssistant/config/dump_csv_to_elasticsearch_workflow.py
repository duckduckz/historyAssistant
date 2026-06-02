from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Any, Dict, Generator

from elasticsearch import Elasticsearch, helpers

DEFAULT_ES_HOST = "localhost"
DEFAULT_ES_PORT = 9200
INDEX_NAME = "televic.history.workflow"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(
    SCRIPT_DIR, "../../../data/historyWorkflows/processed/fact_history_workflows_v1.csv"
)

DATE_FIELDS = {"workflow_date"}
DATETIME_FIELDS = {"timestamp_utc", "cancel_timestamp_utc"}
INTEGER_FIELDS = {"workflow_hour", "workflow_dayofweek"}
FLOAT_FIELDS = {"activation_time_min", "duration_sec"}

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "workflow_id":           {"type": "keyword"},
            "timestamp_utc":         {"type": "date", "format": "strict_date_optional_time"},
            "cancel_timestamp_utc":  {"type": "date", "format": "strict_date_optional_time"},
            "workflow_date":         {"type": "date", "format": "yyyy-MM-dd"},
            "workflow_hour":         {"type": "integer"},
            "workflow_dayofweek":    {"type": "integer"},
            "shift":                 {"type": "keyword"},
            "workflow_name":         {"type": "keyword"},
            "workflow_type":         {"type": "keyword"},
            "ward_name":             {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "room_name":             {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "initiator_person":      {"type": "keyword"},
            "cancel_person":         {"type": "keyword"},
            "activation_time_min":   {"type": "float"},
            "duration_sec":          {"type": "float"},
        }
    }
}


def _convert_date(value: str) -> str:
    """Convert DD/MM/YYYY to YYYY-MM-DD."""
    return datetime.strptime(value, "%d/%m/%Y").strftime("%Y-%m-%d")


def _convert_datetime(value: str) -> str:
    """Convert CSV datetime strings to strict_date_optional_time-compatible ISO."""
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    if "T" in cleaned:
        return cleaned
    # CSV values look like "2024-01-31 05:30:12+00:00"; ES strict format needs T.
    return cleaned.replace(" ", "T", 1)


def _convert_int(value: str) -> int:
    return int(float(value.strip()))


def _convert_float(value: str) -> float:
    return float(value.strip())


def read_csv_and_generate_docs(csv_path: str) -> Generator[Dict[str, Any], None, None]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None or (isinstance(value, str) and value.strip() in ("", "nan", "NaN")):
                    doc[key] = None
                elif key in DATE_FIELDS:
                    doc[key] = _convert_date(value.strip())
                elif key in DATETIME_FIELDS:
                    doc[key] = _convert_datetime(value)
                elif key in INTEGER_FIELDS:
                    doc[key] = _convert_int(value)
                elif key in FLOAT_FIELDS:
                    doc[key] = _convert_float(value)
                else:
                    doc[key] = value
            yield {"_index": INDEX_NAME, "_source": doc}


def count_csv_rows(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Load workflow CSV data into Elasticsearch")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="Path to CSV file")
    parser.add_argument("--es-host", default=DEFAULT_ES_HOST, help="Elasticsearch host")
    parser.add_argument("--es-port", type=int, default=DEFAULT_ES_PORT, help="Elasticsearch port")
    parser.add_argument("--recreate-index", action="store_true", help="Delete and recreate the index")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
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

    total_docs = count_csv_rows(args.csv_path)
    print(f"Found {total_docs} records in CSV file")
    print(f"Indexing documents to '{INDEX_NAME}'...")

    success_count = 0
    error_count = 0

    for ok, response in helpers.streaming_bulk(
        es,
        read_csv_and_generate_docs(args.csv_path),
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
