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
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the class and constants to be tested
from agefreighter.neo4jloader import Neo4jLoader


# Dummy CSV manager returning a DataFrame with all required columns.
class DummyCsvManager:
    def get_dataframe(self):
        data = {
            "CustomerID": ["C1"],
            "start_vertex_type": ["Customer"],
            "Name": ["Alice"],
            "Address": ["123 Main St"],
            "Email": ["alice@example.com"],
            "Phone": ["1234567890"],
            "ProductID": ["P1"],
            "end_vertex_type": ["Product"],
            "SKU": ["SKU1"],
            "Price": [100],
            "Color": ["Red"],
            "Size": ["M"],
            "Weight": [1.0],
        }
        return pd.DataFrame(data)


# Fake asynchronous session to simulate Neo4j session behavior.
class FakeSession:
    def __init__(self):
        self.run_calls = []  # Collects all calls to session.run

    async def run(self, query, **kwargs):
        self.run_calls.append((query, kwargs))
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


# Fake asynchronous driver to simulate Neo4j driver behavior.
class FakeDriver:
    def __init__(self):
        self.session_instance = FakeSession()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def session(self):
        return self.session_instance


class TestNeo4jLoader(unittest.TestCase):
    def setUp(self):
        self.csv_manager = DummyCsvManager()
        self.loader = Neo4jLoader(
            csv_manager=self.csv_manager,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            log_level=logging.DEBUG,
        )

    @patch("neo4j.AsyncGraphDatabase.driver")
    def test_load_data_success(self, mock_driver):
        # Create a fake driver instance.
        fake_driver = FakeDriver()
        mock_driver.return_value = fake_driver

        # Capture printed output (from _show_time)
        f = io.StringIO()
        with redirect_stdout(f):
            asyncio.run(self.loader.load_data())
        output = f.getvalue()
        self.assertIn("Time for load_data:", output)

        # Verify that session.run was called the expected number of times.
        # Expected calls:
        #   - 2 for clearing the database
        #   - 4 for dropping/creating indices
        #   - 1 for inserting start nodes
        #   - 1 for inserting end nodes
        #   - 1 for creating edges
        # Total = 2 + 4 + 1 + 1 + 1 = 9 calls.
        run_calls = fake_driver.session_instance.run_calls
        self.assertEqual(len(run_calls), 9)

        # Optionally check that key query substrings appear.
        self.assertTrue(
            any("MATCH (a)-[r]->() DELETE a, r" in query for query, _ in run_calls)
        )
        self.assertTrue(any("CREATE INDEX" in query for query, _ in run_calls))
        self.assertTrue(any("UNWIND $starts AS row" in query for query, _ in run_calls))
        self.assertTrue(any("UNWIND $ends AS row" in query for query, _ in run_calls))
        self.assertTrue(
            any("MATCH (from:Customer {CustomerID:" in query for query, _ in run_calls)
        )

    @patch("neo4j.AsyncGraphDatabase.driver", side_effect=Exception("connect failure"))
    def test_load_data_connection_failure(self, mock_driver):
        # When the driver cannot be created, load_data should propagate the exception.
        with self.assertRaises(Exception) as context:
            asyncio.run(self.loader.load_data())
        self.assertIn("connect failure", str(context.exception))

    def test_show_time(self):
        # Test the _show_time static method by capturing its output.
        start_time = time.time() - 2  # simulate 2 seconds elapsed
        f = io.StringIO()
        with redirect_stdout(f):
            Neo4jLoader._show_time(start_time, "test_method")
        output = f.getvalue().strip()
        self.assertTrue(output.startswith("Time for test_method: "))
        # Extract the elapsed seconds from the printed message and verify it's approximately 2 seconds.
        elapsed_str = output.split(": ")[-1].split(" ")[0]
        elapsed = float(elapsed_str)
        self.assertAlmostEqual(elapsed, 2, delta=0.5)


if __name__ == "__main__":
    unittest.main()
