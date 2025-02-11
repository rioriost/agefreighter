#!/usr/bin/env python3
"""
An example test module using unittest for testing CSVFreighter.
It re‑implements the pytest tests using IsolatedAsyncioTestCase,
and uses unittest.mock.patch for monkey‑patching.
"""

import asyncio
import io
import os
import sys
import warnings
import contextlib
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import pandas as pd

# Insert the src directory to the sys.path if needed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agefreighter.csvfreighter import CSVFreighter  # adjust as needed


# Dummy (patched) implementations for the AgeFreighter async context manager methods.
# These replace the base class’s __aenter__ and __aexit__.
async def dummy_aenter(self):
    # Could log that it was called if needed.
    return self


async def dummy_aexit(self, exc_type, exc, tb):
    # If an exception is provided, print out the exception message.
    if exc is not None:
        print(f"Exception: {exc_type} - {exc}")
    # Simply do nothing else; in real usage, this might close a connection, etc.
    # Note: __aexit__ should return a boolean (or None). Returning None is acceptable.
    return None


# A helper dummy for createGraphFromDataFrame used inside load.
# This dummy will record its call arguments.
class GraphCreatorRecorder:
    def __init__(self):
        self.calls = []

    async def record_call(self, **kwargs):
        self.calls.append(kwargs)
        # simulate async work
        await asyncio.sleep(0)


# A dummy for close method.
class CloserRecorder:
    def __init__(self):
        self.called = False

    async def record_close(self):
        self.called = True
        await asyncio.sleep(0)


class TestCSVFreighter(IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a new instance of CSVFreighter for each test.
        self.csv_freighter = CSVFreighter()

        # Patch the inherited AgeFreighter async context manager methods.
        # We assume that CSVFreighter inherits from a class (here AgeFreighter)
        # that defines __aenter__ and __aexit__. We monkey-patch those methods.
        # (Alternatively, you could use patch.object.)
        from agefreighter.csvfreighter import AgeFreighter  # adjust if needed

        AgeFreighter.__aenter__ = dummy_aenter
        AgeFreighter.__aexit__ = dummy_aexit

    async def test_async_context_manager(self):
        # Test that async context manager returns self.
        async with self.csv_freighter as cf:
            self.assertIs(cf, self.csv_freighter)
        # No exception should have been raised.

    async def test_aexit_with_exception(self):
        # Test that __aexit__ prints a message when an exception is passed.
        test_exc = ValueError("Test error")

        # Capture stdout output.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            await self.csv_freighter.__aexit__(ValueError, test_exc, None)
        output = out.getvalue()

        self.assertIn("Exception: <class 'ValueError'>", output)
        self.assertIn("Test error", output)

    async def test_load_calls_createGraphFromDataFrame_and_close(self):
        """
        Test that load calls createGraphFromDataFrame for each chunk and then calls close.
        We simulate reading a CSV by patching pandas.read_csv so that it returns two dummy dataframes.
        """
        # Create dummy dataframes as chunks.
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5], "b": [6]})

        # Define a dummy read_csv that simulates chunking by returning an iterator over our dataframes.
        def dummy_read_csv(csv_path, chunksize):
            # Ignore csv_path and chunksize; return an iterator that yields our dataframes.
            return (df for df in [df1, df2])

        # Patch pandas.read_csv using unittest.mock.patch.
        with patch.object(pd, "read_csv", side_effect=dummy_read_csv):
            # Prepare a recorder for createGraphFromDataFrame calls.
            recorder = GraphCreatorRecorder()
            self.csv_freighter.createGraphFromDataFrame = recorder.record_call

            # Prepare a recorder for the close method.
            closer = CloserRecorder()
            self.csv_freighter.close = closer.record_close

            # For testing, provide a progress parameter.
            progress_val = "dummy progress"

            # Call load with a dummy csv_path; many parameters are passed through.
            await self.csv_freighter.load(
                csv_path="dummy_path.csv",
                start_v_label="StartLabel",
                start_id="id1",
                start_props=["p1", "p2"],
                edge_type="edge",
                edge_props=["e1"],
                end_v_label="EndLabel",
                end_id="id2",
                end_props=["p3"],
                graph_name="graph_test",
                chunk_size=10,  # small chunk size for testing
                direct_loading=True,
                create_graph=True,
                use_copy=False,
                progress=progress_val,
            )

            # Check that the progress parameter was assigned to the instance.
            self.assertEqual(
                getattr(self.csv_freighter, "progress", None), progress_val
            )

            # There should be two calls to createGraphFromDataFrame (one per chunk).
            self.assertEqual(len(recorder.calls), 2)

            # Verify that the first call has first_chunk==True and that subsequent call has first_chunk==False.
            first_call = recorder.calls[0]
            second_call = recorder.calls[1]
            self.assertTrue(first_call.get("first_chunk"))
            self.assertFalse(second_call.get("first_chunk"))

            # Check that other parameters were passed correctly in each call.
            for call in recorder.calls:
                self.assertEqual(call.get("graph_name"), "graph_test")
                self.assertEqual(call.get("start_v_label"), "StartLabel")
                self.assertEqual(call.get("start_id"), "id1")
                self.assertEqual(call.get("start_props"), ["p1", "p2"])
                self.assertEqual(call.get("edge_type"), "edge")
                self.assertEqual(call.get("edge_props"), ["e1"])
                self.assertEqual(call.get("end_v_label"), "EndLabel")
                self.assertEqual(call.get("end_id"), "id2")
                self.assertEqual(call.get("end_props"), ["p3"])
                self.assertEqual(call.get("chunk_size"), 10)
                self.assertTrue(call.get("direct_loading"))
                self.assertTrue(call.get("create_graph"))
                self.assertFalse(call.get("use_copy"))
                # existing_node_ids should exist (its content is not checked here)
                self.assertIn("existing_node_ids", call)

            # Finally, verify that close was called after processing chunks.
            self.assertTrue(closer.called)

    async def test_load_deprecated_csv_kwarg(self):
        """
        Test that passing the old "csv" keyword produces a DeprecationWarning and uses its value.
        """

        # First, define a dummy read_csv that yields no chunks.
        def dummy_read_csv(csv_path, chunksize):
            # Yield nothing.
            return iter([])

        # Patch pandas.read_csv to our dummy.
        with patch.object(pd, "read_csv", side_effect=dummy_read_csv):
            # Prepare dummy async functions for createGraphFromDataFrame and close so that load proceeds.
            async def dummy_cg(*args, **kwargs):
                await asyncio.sleep(0)

            self.csv_freighter.createGraphFromDataFrame = dummy_cg

            async def dummy_close():
                await asyncio.sleep(0)

            self.csv_freighter.close = dummy_close

            # Use warnings.catch_warnings to ensure a DeprecationWarning is issued.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await self.csv_freighter.load(csv="deprecated_path.csv")
            # Check that at least one warning mentioned "csv" in its message.
            self.assertTrue(any("csv" in str(warning.message) for warning in w))

            # Next, check that the value passed via the deprecated csv keyword is used as csv_path.
            # We capture the csv_path passed to read_csv.
            called_csv_path = None

            def capture_read_csv(csv_path, chunksize):
                nonlocal called_csv_path
                called_csv_path = csv_path
                return iter([])

            with patch.object(pd, "read_csv", side_effect=capture_read_csv):
                await self.csv_freighter.load(csv="deprecated_again.csv")
            self.assertEqual(called_csv_path, "deprecated_again.csv")


if __name__ == "__main__":
    unittest.main()
