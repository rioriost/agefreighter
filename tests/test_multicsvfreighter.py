#!/usr/bin/env python3
"""
Unit tests for MultiCSVFreighter.
"""

import os
import sys
import warnings
import unittest
import tempfile
import csv
from unittest.mock import AsyncMock, MagicMock


# Insert the src directory into sys.path if needed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agefreighter.multicsvfreighter import MultiCSVFreighter  # adjust if needed


class TestMultiCSVFreighter(unittest.IsolatedAsyncioTestCase):
    # Helper method to create a temporary CSV file with given header and a single row.
    def create_temp_csv(self, header, row):
        temp = tempfile.NamedTemporaryFile(
            mode="w", delete=False, newline="", suffix=".csv"
        )
        writer = csv.DictWriter(temp, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)
        temp.close()
        return temp.name

    async def asyncSetUp(self):
        # Create dummy CSV files for vertices and edges.
        # For vertices, we need at least the column "id".
        self.vertex_csv = self.create_temp_csv(header=["id"], row={"id": "1"})
        # For edges, we need columns "start_id", "start_vertex_type", "end_id", "end_vertex_type".
        # We include one extra column "weight" for edge properties.
        self.edge_csv = self.create_temp_csv(
            header=[
                "start_id",
                "start_vertex_type",
                "end_id",
                "end_vertex_type",
                "weight",
            ],
            row={
                "start_id": "1",
                "start_vertex_type": "A",
                "end_id": "2",
                "end_vertex_type": "B",
                "weight": "3.14",
            },
        )

        # Create an instance of MultiCSVFreighter.
        self.freighter = MultiCSVFreighter()

        # Replace the base class methods with AsyncMocks so that we avoid
        # actual side effects and can track calls.
        # Replace the dependent methods with mocks.
        self.freighter.setUpGraph = AsyncMock(return_value=None)
        self.freighter.createLabelType = AsyncMock(return_value=None)
        self.freighter.createVertices = AsyncMock(return_value=None)
        self.freighter.createEdges = AsyncMock(return_value=None)
        # Use a synchronous MagicMock for checkKeys,
        # because load() calls it without await.
        self.freighter.checkKeys = MagicMock(return_value=None)
        self.freighter.close = AsyncMock(return_value=None)

    async def asyncTearDown(self):
        # Clean up the temporary files.
        os.remove(self.vertex_csv)
        os.remove(self.edge_csv)

    async def test_normal_load(self):
        await self.freighter.load(
            vertex_csv_paths=[self.vertex_csv],
            vertex_labels=["VertexLabel"],
            edge_csv_paths=[self.edge_csv],
            edge_types=["EdgeType"],
            graph_name="test_graph",
            chunk_size=1,  # small size to force one chunk for testing
            direct_loading=True,
            create_graph=True,
            use_copy=True,
        )

        self.freighter.setUpGraph.assert_awaited_once_with(
            graph_name="test_graph", create_graph=True
        )

        # Since checkKeys is a MagicMock, its call count is stored in call_count
        self.assertTrue(self.freighter.checkKeys.call_count >= 1)
        first_vertex_call = self.freighter.checkKeys.call_args_list[0]
        vertex_keys = list(first_vertex_call.args[0])
        self.assertIn("id", vertex_keys)

        self.freighter.createLabelType.assert_any_await(
            label_type="vertex", value="VertexLabel"
        )
        self.assertTrue(self.freighter.createVertices.await_count >= 1)

        self.assertTrue(self.freighter.checkKeys.call_count >= 2)
        first_edge_call = self.freighter.checkKeys.call_args_list[1]
        edge_keys = list(first_edge_call.args[0])
        expected_edge_keys = [
            "start_id",
            "start_vertex_type",
            "end_id",
            "end_vertex_type",
            "weight",
        ]
        for key in expected_edge_keys:
            self.assertIn(key, edge_keys)

        self.freighter.createLabelType.assert_any_await(
            label_type="edge", value="EdgeType"
        )
        self.assertTrue(self.freighter.createEdges.await_count >= 1)

        create_edges_call = self.freighter.createEdges.await_args_list[0]
        if "edges" in create_edges_call.kwargs:
            edges_df = create_edges_call.kwargs["edges"]
        else:
            edges_df = create_edges_call.args[0]
        self.assertIn("start_v_label", edges_df.columns)
        self.assertIn("end_v_label", edges_df.columns)
        self.assertNotIn("start_vertex_type", edges_df.columns)
        self.assertNotIn("end_vertex_type", edges_df.columns)

        self.freighter.close.assert_awaited_once()

    async def test_deprecated_parameters(self):
        # This test verifies that using the deprecated parameters "vertex_csvs" and "edge_csvs"
        # produces a DeprecationWarning.
        # We'll use warnings.catch_warnings to record warnings.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Call load with deprecated keyword arguments.
            await self.freighter.load(
                vertex_csv_paths=[],  # empty list; will be overridden by deprecated parameter below.
                vertex_labels=["DeprecatedVertex"],
                edge_csv_paths=[],  # empty list
                edge_types=["DeprecatedEdge"],
                graph_name="deprecated_graph",
                vertex_csvs=[self.vertex_csv],
                edge_csvs=[self.edge_csv],
                chunk_size=1,
                direct_loading=False,
                create_graph=False,
                use_copy=False,
            )

            # Check that we got at least two deprecation warnings, one for each deprecated kwarg.
            dep_warnings = [
                warn for warn in w if issubclass(warn.category, DeprecationWarning)
            ]
            self.assertGreaterEqual(len(dep_warnings), 2)

        # Also verify that methods were still called (setUpGraph, etc.).
        self.freighter.setUpGraph.assert_awaited_once_with(
            graph_name="deprecated_graph", create_graph=False
        )
        self.freighter.createLabelType.assert_any_await(
            label_type="vertex", value="DeprecatedVertex"
        )
        self.freighter.createLabelType.assert_any_await(
            label_type="edge", value="DeprecatedEdge"
        )
        self.freighter.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
