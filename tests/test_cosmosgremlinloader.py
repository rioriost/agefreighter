#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import io
import logging
import os
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agefreighter.cosmosgremlinloader import CosmosGremlinLoader
from agefreighter.csvdatamanager import CsvDataManager


# Dummy CSV manager to return a test DataFrame
class DummyCsvManager:
    def get_dataframe(self):
        data = {
            "CustomerID": ["C1"],
            "Name": ["Alice"],
            "Address": ["123 Main St"],
            "Email": ["alice@example.com"],
            "Phone": ["1234567890"],
            "ProductID": ["P1"],
            "Phrase": ["A great product"],
            "SKU": ["SKU1"],
            "Price": [100],
            "Color": ["Red"],
            "Size": ["M"],
            "Weight": [1.0],
        }
        return pd.DataFrame(data)


# Fake get_chunks function to replace CsvDataManager.get_chunks.
def fake_get_chunks(df, chunk_size):
    # For testing, simply return the full DataFrame as one chunk.
    return [df]


# Dummy future for simulating the Gremlin client's submitAsync method.
class DummyFuture:
    def __init__(self, succeed=True):
        self.succeed = succeed

    def result(self):
        if self.succeed:
            # Create a dummy result object with a chained .all().result() call.
            dummy_result = MagicMock()
            dummy_result.all.return_value.result.return_value = "dummy"
            return dummy_result
        else:
            raise Exception("failure")


class TestCosmosGremlinLoader(unittest.TestCase):
    def setUp(self):
        self.csv_manager = DummyCsvManager()
        self.loader = CosmosGremlinLoader(
            csv_manager=self.csv_manager,
            cosmos_gremlin_endpoint="dummy_endpoint",
            cosmos_key="dummy_key",
            cosmos_database="dummy_db",
            cosmos_container="dummy_container",
            log_level=logging.DEBUG,
        )

    @patch("time.sleep", return_value=None)
    def test_execute_gremlin_query_success(self, mock_sleep):
        # Create a dummy client whose submitAsync returns a successful dummy future.
        fake_future = DummyFuture(succeed=True)
        fake_client = MagicMock()
        fake_client.submitAsync.return_value = fake_future

        # Should complete without exception.
        self.loader.execute_gremlin_query(fake_client, "dummy_query")
        fake_client.submitAsync.assert_called_with("dummy_query")

    @patch("time.sleep", return_value=None)
    def test_execute_gremlin_query_failure(self, mock_sleep):
        # Create a dummy client whose submitAsync always returns a failing dummy future.
        fake_future = DummyFuture(succeed=False)
        fake_client = MagicMock()
        fake_client.submitAsync.return_value = fake_future

        # After 5 attempts, the method should raise an exception.
        with self.assertRaises(Exception) as context:
            self.loader.execute_gremlin_query(fake_client, "dummy_query")
        self.assertIn("Max retries exceeded", str(context.exception))
        # Check that 5 retry attempts were made.
        self.assertEqual(fake_client.submitAsync.call_count, 5)

    @patch.object(CsvDataManager, "get_chunks", side_effect=fake_get_chunks)
    @patch("gremlin_python.driver.client.Client")
    def test_load_data_success(self, mock_client_constructor, mock_get_chunks):
        # Set up a fake client that always returns a successful dummy future.
        fake_future = DummyFuture(succeed=True)
        fake_client = MagicMock()
        fake_client.submitAsync.return_value = fake_future
        mock_client_constructor.return_value = fake_client

        # Capture printed output (including _show_time output).
        f = io.StringIO()
        with redirect_stdout(f):
            asyncio.run(self.loader.load_data())
        output = f.getvalue()

        # Verify that the fake client was closed and the _show_time method printed timing info.
        self.assertTrue(fake_client.close.called)
        self.assertIn("Time for load_data:", output)

    @patch(
        "gremlin_python.driver.client.Client", side_effect=Exception("connect failure")
    )
    def test_load_data_connection_failure(self, mock_client_constructor):
        # In this case, the client constructor raises an exception.
        f = io.StringIO()
        with redirect_stdout(f):
            asyncio.run(self.loader.load_data())
        output = f.getvalue()
        self.assertIn("Failed to connect to Gremlin server: connect failure", output)

    def test_show_time(self):
        # Test the _show_time static method by capturing its output.
        start_time = time.time() - 2  # simulate 2 seconds elapsed
        f = io.StringIO()
        with redirect_stdout(f):
            CosmosGremlinLoader._show_time(start_time, "test_method")
        output = f.getvalue().strip()
        self.assertTrue(output.startswith("Time for test_method: "))
        # Extract the elapsed seconds from the printed message and verify it's approximately 2 seconds.
        elapsed_str = output.split(": ")[-1].split(" ")[0]
        elapsed = float(elapsed_str)
        self.assertAlmostEqual(elapsed, 2, delta=0.5)


if __name__ == "__main__":
    unittest.main()
