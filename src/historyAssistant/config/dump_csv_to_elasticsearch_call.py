
import argparse
import csv
import os
from datetime import datetime
from typing import Any, Dict, Generator

from elasticsearch import Elasticsearch, helpers

# Default configuration
DEFAULT_ES_HOST = "localhost"
DEFAULT_ES_PORT = 9200
INDEX_NAME = "televic.history.call"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(
    SCRIPT_DIR, "../../../data/historyCalls/03_processed/fact_history_calls_v3.csv"
)

DATE_FIELDS = {"call_date"}

BOOLEAN_AS_INT_FIELDS = {
    "is_cancelled",
    "dq_unknown_location_flag",
    "dq_missing_staying_time_when_cancelled",
    "dq_missing_response_time",
    "dq_extreme_response_time_flag",
    "dq_invalid_priority_flag",
    "dq_invalid_hour_flag",
    "dq_invalid_dayofweek_flag",
    "dq_cancelled_missing_person_flag",
    "dq_unknown_person_type_flag",
    "dq_negative_response_time_flag",
    "dq_extreme_staying_time_flag",
    "dq_parse_error_flag",
    "qa_ready_flag",
}

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "call_id": {"type": "integer"},
            "timestamp_utc": {"type": "keyword"},
            "cancel_timestamp_utc": {"type": "keyword"},
            "call_date": {"type": "date", "format": "yyyy-MM-dd"},
            "call_month": {"type": "keyword"},
            "call_hour": {"type": "integer"},
            "call_dayofweek": {"type": "integer"},
            "shift": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "call_type": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "priority": {"type": "integer"},
            "ward_id": {"type": "integer"},
            "ward_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "room_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "zone_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "area_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "person_location": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "person_type": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "person_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "call_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "cancel_person": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "cancellation_trigger": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "triggered_by": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "cancel_location": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "is_cancelled": {"type": "integer"},
            "response_time_sec": {"type": "float"},
            "staying_time_sec": {"type": "float"},
            "staying_time_min": {"type": "float"},
            "dq_unknown_location_flag": {"type": "integer"},
            "dq_missing_staying_time_when_cancelled": {"type": "integer"},
            "dq_missing_response_time": {"type": "integer"},
            "dq_extreme_response_time_flag": {"type": "integer"},
            "dq_invalid_priority_flag": {"type": "integer"},
            "dq_invalid_hour_flag": {"type": "integer"},
            "dq_invalid_dayofweek_flag": {"type": "integer"},
            "dq_cancelled_missing_person_flag": {"type": "integer"},
            "dq_unknown_person_type_flag": {"type": "integer"},
            "dq_negative_response_time_flag": {"type": "integer"},
            "dq_extreme_staying_time_flag": {"type": "integer"},
            "dq_parse_error_flag": {"type": "integer"},
            "qa_ready_flag": {"type": "integer"},
        }
    }
}


def _convert_date(value: str) -> str:
    """Convert DD/MM/YYYY to YYYY-MM-DD."""
    return datetime.strptime(value, "%d/%m/%Y").strftime("%Y-%m-%d")


def read_csv_and_generate_docs(csv_path: str) -> Generator[Dict[str, Any], None, None]:
    """Read CSV file and generate documents for Elasticsearch."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = {}
            for key, value in row.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    doc[key] = None
                elif key in DATE_FIELDS:
                    converted = _convert_date(value)
                    doc[key] = converted
                    if key == "call_date":
                        doc["call_month"] = converted[:7]  # YYYY-MM
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
                    # Try to convert numeric strings to appropriate types
                    try:
                        if "." in value:
                            doc[key] = float(value)
                        else:
                            doc[key] = int(value)
                    except (ValueError, TypeError):
                        doc[key] = value

            yield {"_index": INDEX_NAME, "_source": doc}


def count_csv_rows(csv_path: str) -> int:
    """Count total number of data rows in the CSV file."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # subtract header row


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CSV data into Elasticsearch")
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

    # Connect to Elasticsearch
    es = Elasticsearch([f"http://{args.es_host}:{args.es_port}"])

    # Check connection
    if not es.ping():
        print("Error: Could not connect to Elasticsearch")
        return

    print("Connected successfully!")

    # Optionally recreate index
    if args.recreate_index and es.indices.exists(index=INDEX_NAME):
        print(f"Deleting existing index '{INDEX_NAME}'...")
        es.indices.delete(index=INDEX_NAME)

    # Create index with explicit mapping if it doesn't exist
    if not es.indices.exists(index=INDEX_NAME):
        print(f"Creating index '{INDEX_NAME}' with mapping...")
        es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)

    # Count documents in CSV
    total_docs = count_csv_rows(csv_path)
    print(f"Found {total_docs} records in CSV file")
    print(f"Indexing documents to '{INDEX_NAME}'...")

    # Bulk index documents
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

        # Progress update every 1000 documents
        if (success_count + error_count) % 1000 == 0:
            print(f"Progress: {success_count + error_count}/{total_docs} documents processed")

    print("\n" + "=" * 50)
    print("Indexing complete!")
    print(f"Successfully indexed: {success_count} documents")
    print(f"Errors: {error_count} documents")
    print("=" * 50)

    # Verify index
    es.indices.refresh(index=INDEX_NAME)
    doc_count = es.count(index=INDEX_NAME)["count"]
    print(f"Total documents in index '{INDEX_NAME}': {doc_count}")


if __name__ == "__main__":
    main()
