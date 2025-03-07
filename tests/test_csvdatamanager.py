import os
import sys
import logging
import tempfile
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

# Adjust the import if your package structure is different.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agefreighter.csvdatamanager import CsvDataManager


class TestCsvDataManager(unittest.TestCase):
    def test_init_with_custom_data_dir(self):
        """Test CsvDataManager initialization with a custom data directory and base file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_file = "test_file"
            manager = CsvDataManager(
                data_dir=temp_dir, base_file=base_file, log_level=logging.DEBUG
            )
            # Expect that csv_file equals os.path.join(temp_dir, base_file) exactly.
            expected_csv_file = os.path.abspath(os.path.join(temp_dir, base_file))
            self.assertEqual(manager.csv_file, expected_csv_file)

    def test_init_with_none_data_dir(self):
        """Test that initializing CsvDataManager with data_dir=None does not set csv_file."""
        manager = CsvDataManager(data_dir=None, base_file="test_file")
        # Since data_dir is falsy, csv_file is not set.
        self.assertFalse(hasattr(manager, "csv_file"))

    def test_get_dataframe(self):
        """Test that get_dataframe() correctly reads a CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_file = "test_data"
            # Create the CSV file using the expected file name (without any extension appended).
            csv_path = os.path.join(temp_dir, base_file)
            # Create a sample DataFrame and write it to CSV.
            df_expected = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            df_expected.to_csv(csv_path, index=False)
            manager = CsvDataManager(data_dir=temp_dir, base_file=base_file)
            df_result = manager.get_dataframe()
            assert_frame_equal(df_result, df_expected)

    def test_get_chunks_regular(self):
        """Test that get_chunks() splits a DataFrame into expected chunks."""
        df = pd.DataFrame({"A": list(range(10))})
        chunk_size = 3
        chunks = list(CsvDataManager.get_chunks(df, chunk_size))
        # There should be 4 chunks (3 chunks of size 3 and 1 chunk of size 1)
        self.assertEqual(len(chunks), 4)
        # Check chunk sizes for all but the last chunk.
        for chunk in chunks[:-1]:
            self.assertEqual(len(chunk), chunk_size)
        # The last chunk should have the remaining rows.
        self.assertEqual(len(chunks[-1]), 10 % chunk_size)

    def test_get_chunks_chunk_size_larger_than_df(self):
        """Test that get_chunks() returns the entire DataFrame if chunk_size > len(df)."""
        df = pd.DataFrame({"A": [1, 2]})
        chunk_size = 5
        chunks = list(CsvDataManager.get_chunks(df, chunk_size))
        self.assertEqual(len(chunks), 1)
        assert_frame_equal(chunks[0], df)


if __name__ == "__main__":
    unittest.main()
