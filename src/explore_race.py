"""Minimal entry point to ingest or load the F1 race session data and inspect ingestion status."""

from pathlib import Path
import sys

# Ensure src package can be imported if executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion import load_or_fetch_race_laps


def main() -> None:
    print("=" * 60)
    print("F1 Strategy Intelligence Engine — Ingestion & Status")
    print("=" * 60)

    year = 2024
    country = "Bahrain"
    session = "Race"

    print(f"Target Session: {year} {country} Grand Prix - {session}")
    print("Checking data pipeline status...")

    try:
        laps_df, session_info, from_cache = load_or_fetch_race_laps(
            year=year,
            country_name=country,
            session_name=session,
            force_download=False,
        )
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}")
        sys.exit(1)

    source = "Local raw cache" if from_cache else "Live OpenF1 API"
    session_key = session_info.get("session_key", "Unknown")

    print("\n--- Ingestion Summary ---")
    print(f"Data source          : {source}")
    print(f"Session key          : {session_key}")
    print(f"Total lap records    : {len(laps_df):,}")
    print(f"Total columns        : {len(laps_df.columns)}")
    print(f"Unique drivers       : {laps_df['driver_number'].nunique()}")
    print(f"Min / Max lap number : {laps_df['lap_number'].min()} / {laps_df['lap_number'].max()}")
    print(f"Valid lap durations  : {laps_df['lap_duration'].notna().sum():,} non-null")
    print("-" * 60)
    print("Pre-data-science setup is complete. Data is ready for analysis!")


if __name__ == "__main__":
    main()
