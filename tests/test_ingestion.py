"""Tests for data ingestion and validation module."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from src.ingestion import (
    IngestionError,
    ValidationError,
    get_lap_data,
    get_session_metadata,
    load_or_fetch_race_laps,
    save_raw_dataset,
    validate_lap_data,
)


class TestIngestionUnit(unittest.TestCase):
    """Unit tests using mocked HTTP responses to avoid network dependencies."""

    @patch("src.ingestion.requests.get")
    def test_get_session_metadata_success(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"session_key": 9472, "session_name": "Race", "year": 2024, "country_name": "Bahrain"}
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        meta = get_session_metadata(2024, "Bahrain", "Race")
        self.assertEqual(meta["session_key"], 9472)
        mock_get.assert_called_once()

    @patch("src.ingestion.requests.get")
    def test_get_session_metadata_empty(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with self.assertRaises(IngestionError) as ctx:
            get_session_metadata(2024, "Atlantis", "Race")
        self.assertIn("No session found", str(ctx.exception))

    @patch("src.ingestion.requests.get")
    def test_get_session_metadata_http_error(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with self.assertRaises(IngestionError) as ctx:
            get_session_metadata(2024, "Bahrain", "Race")
        self.assertIn("Network/HTTP error", str(ctx.exception))

    @patch("src.ingestion.requests.get")
    def test_get_session_metadata_timeout(self, mock_get: MagicMock):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        with self.assertRaises(IngestionError) as ctx:
            get_session_metadata(2024, "Bahrain", "Race")
        self.assertIn("Timed out", str(ctx.exception))

    @patch("src.ingestion.requests.get")
    def test_get_lap_data_success(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"driver_number": 1, "lap_number": 1, "lap_duration": 96.5},
            {"driver_number": 1, "lap_number": 2, "lap_duration": 94.2},
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        df = get_lap_data(9472)
        self.assertEqual(len(df), 2)
        self.assertIn("lap_duration", df.columns)

    @patch("src.ingestion.requests.get")
    def test_get_lap_data_empty(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with self.assertRaises(IngestionError) as ctx:
            get_lap_data(9472)
        self.assertIn("No lap records returned", str(ctx.exception))

    def test_validate_lap_data_valid(self):
        df = pd.DataFrame({
            "driver_number": [1, 11],
            "lap_number": [1, 1],
            "lap_duration": [95.0, 96.1],
        })
        # Should not raise
        validate_lap_data(df)

    def test_validate_lap_data_missing_column(self):
        df = pd.DataFrame({
            "driver_number": [1, 11],
            "lap_number": [1, 1],
        })
        with self.assertRaises(ValidationError) as ctx:
            validate_lap_data(df)
        self.assertIn("missing required column", str(ctx.exception))

    def test_validate_lap_data_empty(self):
        df = pd.DataFrame()
        with self.assertRaises(ValidationError) as ctx:
            validate_lap_data(df)
        self.assertIn("empty", str(ctx.exception))

    def test_save_and_load_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sample_df = pd.DataFrame({
                "driver_number": [1, 11],
                "lap_number": [1, 1],
                "lap_duration": [95.0, 96.1],
                "session_key": [9472, 9472],
            })
            meta = {"session_key": 9472, "year": 2024, "country_name": "Bahrain"}

            csv_path, json_path = save_raw_dataset(
                laps_df=sample_df,
                session_metadata=meta,
                raw_dir=tmp_path,
                filename_prefix="bahrain_2024",
                overwrite=True,
            )

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            # Verify load_or_fetch_race_laps reuses cache without network call
            df_loaded, meta_loaded, from_cache = load_or_fetch_race_laps(
                year=2024,
                country_name="Bahrain",
                session_name="Race",
                raw_dir=tmp_path,
                force_download=False,
            )
            self.assertTrue(from_cache)
            self.assertEqual(len(df_loaded), 2)
            self.assertEqual(meta_loaded["session_key"], 9472)


class TestSavedDatasetFixture(unittest.TestCase):
    """Verifies that the actual local saved Bahrain dataset meets all requirements."""

    def setUp(self):
        project_root = Path(__file__).resolve().parent.parent
        self.raw_csv = project_root / "data" / "raw" / "bahrain_2024_laps.csv"

    def test_saved_bahrain_csv_structure_and_values(self):
        self.assertTrue(self.raw_csv.exists(), f"Raw dataset not found at {self.raw_csv}")
        df = pd.read_csv(self.raw_csv)

        # 1. Row count should be 1,129 records
        self.assertEqual(len(df), 1129)

        # 2. Required columns must be present and valid
        validate_lap_data(df)

        # 3. Driver numbers and lap numbers should be convertible to numeric/integers
        self.assertTrue(pd.to_numeric(df["driver_number"], errors="coerce").notna().all())
        self.assertTrue(pd.to_numeric(df["lap_number"], errors="coerce").notna().all())

        # 4. Lap durations should have numeric values
        durations = pd.to_numeric(df["lap_duration"], errors="coerce")
        # Lap duration should have majority non-null valid values
        self.assertGreater(durations.notna().sum(), 1000)
        # Reasonableness check: lap duration should be positive
        valid_durations = durations.dropna()
        self.assertTrue((valid_durations > 0).all())


if __name__ == "__main__":
    unittest.main()
