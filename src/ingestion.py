"""Data ingestion module for F1 Strategy Intelligence Engine.

Interacts with the OpenF1 public REST API to retrieve session metadata
and lap timing data, validates core required fields, and provides
local caching to prevent redundant API queries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENF1_BASE_URL = "https://api.openf1.org/v1"

REQUIRED_LAP_COLUMNS = [
    "driver_number",
    "lap_number",
    "lap_duration",
]


class IngestionError(Exception):
    """Raised when data ingestion from the API fails."""


class ValidationError(Exception):
    """Raised when data fails schema or completeness validation."""


def get_session_metadata(
    year: int,
    country_name: str,
    session_name: str,
    base_url: str = OPENF1_BASE_URL,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Fetch session metadata matching the specified year, country, and session name.

    Args:
        year: Championship season year (e.g., 2024).
        country_name: Country where the Grand Prix was held (e.g., 'Bahrain').
        session_name: Name of the session (e.g., 'Race', 'Practice 1', 'Qualifying').
        base_url: Base endpoint URL for OpenF1 API.
        timeout: HTTP request timeout in seconds.

    Returns:
        A dictionary containing the session metadata.

    Raises:
        IngestionError: If the HTTP request fails, times out, or returns no matching session.
    """
    url = f"{base_url}/sessions"
    params = {
        "year": year,
        "country_name": country_name,
        "session_name": session_name,
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        sessions = response.json()
    except requests.exceptions.Timeout as exc:
        raise IngestionError(
            f"Timed out after {timeout}s while fetching session metadata for {year} {country_name} {session_name}."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise IngestionError(
            f"Network/HTTP error fetching session metadata: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise IngestionError(
            f"Received malformed JSON from session endpoint: {exc}"
        ) from exc

    if not isinstance(sessions, list) or not sessions:
        raise IngestionError(
            f"No session found for year={year}, country_name='{country_name}', session_name='{session_name}'."
        )

    return sessions[0]


def get_lap_data(
    session_key: int,
    base_url: str = OPENF1_BASE_URL,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch lap records for a specific session key.

    Args:
        session_key: OpenF1 integer key for the session (e.g., 9472).
        base_url: Base endpoint URL for OpenF1 API.
        timeout: HTTP request timeout in seconds.

    Returns:
        A Pandas DataFrame containing the raw lap records.

    Raises:
        IngestionError: If the HTTP request fails, times out, or returns no records.
    """
    url = f"{base_url}/laps"
    params = {"session_key": session_key}

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        records = response.json()
    except requests.exceptions.Timeout as exc:
        raise IngestionError(
            f"Timed out after {timeout}s while fetching lap records for session_key={session_key}."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise IngestionError(
            f"Network/HTTP error fetching lap records for session_key={session_key}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise IngestionError(
            f"Received malformed JSON from laps endpoint: {exc}"
        ) from exc

    if not isinstance(records, list) or not records:
        raise IngestionError(f"No lap records returned for session_key={session_key}.")

    df = pd.DataFrame(records)
    return df


def validate_lap_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> None:
    """Validate that the lap DataFrame is non-empty and contains required columns.

    Args:
        df: DataFrame of lap records.
        required_columns: List of columns that must be present. Defaults to
            REQUIRED_LAP_COLUMNS (['driver_number', 'lap_number', 'lap_duration']).

    Raises:
        ValidationError: If the DataFrame is empty or missing required columns.
    """
    if required_columns is None:
        required_columns = REQUIRED_LAP_COLUMNS

    if df is None or df.empty:
        raise ValidationError("Lap dataset is empty or None.")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(
            f"Lap dataset is missing required column(s): {missing}. Present columns: {list(df.columns)}"
        )


def save_raw_dataset(
    laps_df: pd.DataFrame,
    session_metadata: Dict[str, Any],
    raw_dir: Path,
    filename_prefix: str,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Save lap data as CSV and session metadata as JSON in raw_dir.

    Args:
        laps_df: Validated DataFrame of lap records.
        session_metadata: Session dictionary.
        raw_dir: Directory path where raw files should be stored.
        filename_prefix: Base prefix for output files (e.g., 'bahrain_2024').
        overwrite: If False, does not overwrite existing files.

    Returns:
        A tuple of (csv_path, json_path).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / f"{filename_prefix}_laps.csv"
    json_path = raw_dir / f"{filename_prefix}_session.json"

    if csv_path.exists() and not overwrite:
        logger.info("CSV file already exists at %s, skipping write.", csv_path)
    else:
        laps_df.to_csv(csv_path, index=False)
        logger.info("Saved lap data to %s", csv_path)

    if json_path.exists() and not overwrite:
        logger.info("JSON file already exists at %s, skipping write.", json_path)
    else:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(session_metadata, f, indent=2)
        logger.info("Saved session metadata to %s", json_path)

    return csv_path, json_path


def load_or_fetch_race_laps(
    year: int = 2024,
    country_name: str = "Bahrain",
    session_name: str = "Race",
    raw_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    """High-level function to retrieve session metadata and lap records.

    Uses existing local files from `raw_dir` when available to avoid unnecessary
    network requests, unless `force_download=True`.

    Args:
        year: Championship season year.
        country_name: Grand Prix host country.
        session_name: Session type (default 'Race').
        raw_dir: Directory where raw datasets are cached. Defaults to `data/raw`.
        force_download: If True, bypasses local files and downloads fresh data.

    Returns:
        A tuple of:
            - DataFrame of laps
            - Dict of session metadata
            - Boolean flag `from_cache` indicating whether local data was used.
    """
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    else:
        raw_dir = Path(raw_dir)

    prefix = f"{country_name.lower().replace(' ', '_')}_{year}"
    csv_path = raw_dir / f"{prefix}_laps.csv"
    json_path = raw_dir / f"{prefix}_session.json"

    # Check if local dataset already exists and is valid
    if not force_download and csv_path.is_file():
        logger.info("Found existing local dataset at %s. Loading...", csv_path)
        laps_df = pd.read_csv(csv_path)
        validate_lap_data(laps_df)

        session_metadata: Dict[str, Any] = {}
        if json_path.is_file():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    session_metadata = json.load(f)
            except Exception as e:
                logger.warning("Could not read local session metadata: %s", e)
        else:
            # Reconstruct basic metadata from lap columns if session_key exists
            if "session_key" in laps_df.columns and not laps_df.empty:
                session_key_val = int(laps_df["session_key"].iloc[0])
                session_metadata = {
                    "session_key": session_key_val,
                    "year": year,
                    "country_name": country_name,
                    "session_name": session_name,
                }
                # Also save the json so subsequent runs have it
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(session_metadata, f, indent=2)

        return laps_df, session_metadata, True

    # Otherwise fetch from OpenF1 API
    logger.info("Fetching session metadata for %s %s %s from OpenF1 API...", year, country_name, session_name)
    session_metadata = get_session_metadata(year=year, country_name=country_name, session_name=session_name)
    session_key = session_metadata.get("session_key")
    if not session_key:
        raise IngestionError(f"Session metadata does not contain a session_key: {session_metadata}")

    logger.info("Fetching laps for session_key=%s from OpenF1 API...", session_key)
    laps_df = get_lap_data(session_key=int(session_key))
    validate_lap_data(laps_df)

    save_raw_dataset(
        laps_df=laps_df,
        session_metadata=session_metadata,
        raw_dir=raw_dir,
        filename_prefix=prefix,
        overwrite=force_download,
    )

    return laps_df, session_metadata, False
