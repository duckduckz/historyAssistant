from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(
    SCRIPT_DIR, "../../../data/historyLocations/processed/fact_history_locations_v1.csv"
)

LOCATION_COLUMN_ORDER = [
    "locationUpdateId",
    "personId",
    "personFirstName",
    "personType",
    "timestamp",
    "altitude",
    "accuracy",
    "area",
    "areaId",
    "ward",
    "wardId",
    "zoneId",
    "zoneName",
    "time_gap_minutes",
    "location_changed",
    "is_gap",
    "episode_id",
    "movement_status",
    "location_in_episode",
    "movement_detail",
    "prev_location",
]


def rebuild_movement_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("s")
    df = df.sort_values(["personId", "timestamp"]).reset_index(drop=True)

    df["prev_timestamp"] = df.groupby("personId")["timestamp"].shift()
    df["prev_location"] = df.groupby("personId")["area"].shift()
    df["time_gap_minutes"] = (
        df["timestamp"] - df["prev_timestamp"]
    ).dt.total_seconds() / 60

    df["location_changed"] = (
        df["prev_location"].notna()
        & df["area"].notna()
        & (df["area"] != df["prev_location"])
    )
    df["is_gap"] = df["time_gap_minutes"].gt(15).fillna(False)
    df["episode_id"] = df.groupby("personId")["is_gap"].cumsum().astype(int)
    df["location_in_episode"] = (
        df.groupby(["personId", "episode_id"])["location_changed"]
        .cumsum()
        .astype(int)
    )

    df["movement_status"] = np.select(
        [df["location_changed"], df["is_gap"]],
        ["moved", "rested"],
        default="continuous",
    )
    df["movement_detail"] = np.select(
        [
            df["prev_location"].isna(),
            df["location_changed"] & df["is_gap"],
            df["location_changed"],
            df["is_gap"],
        ],
        [
            "session_start",
            "moved_after_gap",
            "moved",
            "rested",
        ],
        default="stationary",
    )

    df = df.drop(columns=["prev_timestamp"])
    df = df.sort_values(["timestamp", "personId"]).reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[[col for col in LOCATION_COLUMN_ORDER if col in df.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild movement fields in fact_history_locations_v1.csv"
    )
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="Location fact CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    rebuilt = rebuild_movement_fields(df)
    rebuilt.to_csv(args.csv_path, index=False)

    print("Updated movement fields.")
    print("\nmovement_status:")
    print(rebuilt["movement_status"].value_counts(dropna=False).to_string())
    print("\nmovement_detail:")
    print(rebuilt["movement_detail"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
