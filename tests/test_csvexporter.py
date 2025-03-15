#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import concurrent.futures
import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agefreighter.csvexporter import (
    ConfigManager,
    CSVExporter,
)


# ========= Tests for ConfigManager =========
class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to hold dummy CSV files and a config file.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir = self.temp_dir.name

        # Create dummy CSV files (they only need a header row).
        self.edge_csv = os.path.join(self.dir, "edge.csv")
        self.start_csv = os.path.join(self.dir, "start_vertex.csv")
        self.end_csv = os.path.join(self.dir, "end_vertex.csv")
        for path in [self.edge_csv, self.start_csv, self.end_csv]:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("id,label,props\n")

        # Create a valid config dictionary (edge config as dict)
        self.config_dict_dict = {
            "edge": {
                "csv_path": self.edge_csv,
                "start_vertex": {
                    "csv_path": self.start_csv,
                    "id": "id",
                    "label": "start",
                    "props": "dummy",
                },
                "end_vertex": {
                    "csv_path": self.end_csv,
                    "id": "id",
                    "label": "end",
                    "props": "dummy",
                },
                "type": "connects",
                "props": "dummy",
            }
        }

        # And one with edge config as a list.
        self.edge_csv2 = os.path.join(self.dir, "edge2.csv")
        self.vertex_csv = os.path.join(self.dir, "vertex.csv")
        with open(self.edge_csv2, "w", encoding="utf-8", newline="") as f:
            f.write("id,label,props\n")
        self.config_dict_list = {
            "edge": [
                {
                    "csv_path": self.edge_csv2,
                    "type": "works_with",
                    "props": "dummy",
                    "vertex": {
                        "csv_path": self.vertex_csv,
                        "id": "id",
                        "label": "employee",
                        "props": "dummy",
                    },
                },
                {
                    "csv_path": self.edge_csv,
                    "type": "manages",
                    "props": "dummy",
                    "start_vertex": {
                        "csv_path": self.start_csv,
                        "id": "id",
                        "label": "manager",
                        "props": "dummy",
                    },
                    "end_vertex": {
                        "csv_path": self.end_csv,
                        "id": "id",
                        "label": "employee",
                        "props": "dummy",
                    },
                },
            ]
        }

        # Write config files to temporary paths.
        self.config_path_dict = os.path.join(self.dir, "config_dict.json")
        with open(self.config_path_dict, "w", encoding="utf-8") as f:
            json.dump(self.config_dict_dict, f)
        self.config_path_list = os.path.join(self.dir, "config_list.json")
        with open(self.config_path_list, "w", encoding="utf-8") as f:
            json.dump(self.config_dict_list, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_require_key_success(self):
        cm = ConfigManager(self.config_path_dict)
        value = cm.require_key({"a": 1}, "a", "test")
        self.assertEqual(value, 1)

    def test_require_key_missing(self):
        cm = ConfigManager(self.config_path_dict)
        with self.assertRaises(ValueError) as cm_exc:
            cm.require_key({"a": 1}, "b", "test")
        self.assertIn("Missing 'b' key", str(cm_exc.exception))

    def test_load_config_invalid_json(self):
        # Write an invalid JSON file.
        bad_config_path = os.path.join(self.dir, "bad.json")
        with open(bad_config_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")
        cm = ConfigManager(bad_config_path)
        with self.assertRaises(ValueError) as exc:
            cm.load_config()
        self.assertIn("Invalid JSON format", str(exc.exception))

    def test_load_config_valid_dict(self):
        cm = ConfigManager(self.config_path_dict)
        config = cm.load_config()
        self.assertEqual(config, self.config_dict_dict)
        self.assertIsNotNone(cm.parse_result)
        # Check that CSV file existence was verified.
        self.assertTrue(os.path.exists(os.path.abspath(self.edge_csv)))

    def test_load_config_valid_list(self):
        cm = ConfigManager(self.config_path_list)
        config = cm.load_config()
        self.assertEqual(config, self.config_dict_list)
        self.assertIsNotNone(cm.parse_result)

    def test_load_config_missing_csv(self):
        # Create a config with a non-existent CSV file.
        bad_config = {
            "edge": {
                "csv_path": os.path.join(self.dir, "nonexistent.csv"),
                "start_vertex": {
                    "csv_path": self.start_csv,
                    "id": "id",
                    "label": "start",
                    "props": "dummy",
                },
                "end_vertex": {
                    "csv_path": self.end_csv,
                    "id": "id",
                    "label": "end",
                    "props": "dummy",
                },
                "type": "connects",
                "props": "dummy",
            }
        }
        bad_config_path = os.path.join(self.dir, "bad_config.json")
        with open(bad_config_path, "w", encoding="utf-8") as f:
            json.dump(bad_config, f)
        cm = ConfigManager(bad_config_path)
        with self.assertRaises(ValueError) as exc:
            cm.load_config()
        self.assertIn("does not exist", str(exc.exception))


# ========= Tests for CSVExporter =========


class TestCSVExporter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a temporary directory with dummy CSV files and a config file.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir = self.temp_dir.name

        # Dummy CSV files for node export.
        self.edge_csv = os.path.join(self.dir, "edge.csv")
        self.start_csv = os.path.join(self.dir, "start_vertex.csv")
        self.end_csv = os.path.join(self.dir, "end_vertex.csv")
        for path in [self.edge_csv, self.start_csv, self.end_csv]:
            with open(path, "w", encoding="utf-8", newline="") as f:
                # Write header and two rows.
                writer = csv.DictWriter(f, fieldnames=["id", "name"])
                writer.writeheader()
                writer.writerow({"id": "1", "name": '"Alice"'})
                writer.writerow({"id": "2", "name": '"Bob"'})

        # Create a valid config dictionary (edge config as dict).
        self.config_dict = {
            "edge": {
                "csv_path": self.edge_csv,
                "start_vertex": {
                    "csv_path": self.start_csv,
                    "id": "id",
                    "label": "StartLabel",
                    "props": "dummy",
                },
                "end_vertex": {
                    "csv_path": self.end_csv,
                    "id": "id",
                    "label": "EndLabel",
                    "props": "dummy",
                },
                "type": "connects",
                "props": "dummy",
            }
        }
        self.config_path = os.path.join(self.dir, "config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_dict, f)

        # Create an instance of CSVExporter.
        # Patch out AgeFreighter methods that we don't want to run.
        self.exporter = CSVExporter(
            dsn="dummy",
            min_connections=1,
            max_connections=1,
            config=self.config_path,
            trial=False,
            no_of_edges_trial=100,
            save_temps=False,
            progress=False,
            graph_name="dummy_graph",
            chunk_size=1,
            log_level=0,
        )
        # Patch asynchronous methods from AgeFreighter.
        self.exporter.create_label_type = AsyncMock(return_value=None)
        self.exporter.get_first_id = AsyncMock(return_value=1000)
        self.exporter.write_csv = AsyncMock(return_value="dummy_output.csv")
        self.exporter.set_up_graph = AsyncMock(return_value=None)
        # Also, initialize id_maps to an empty dict.
        self.exporter.id_maps = {}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_clean_row(self):
        # Test _clean_row static method.
        row = {'"key"': '"value"', "other": "noquotes"}
        cleaned = self.exporter._clean_row(row)
        self.assertEqual(cleaned, {"key": "value", "other": "noquotes"})

    def test_get_labels_dict(self):
        # Test get_labels for config with edge as dict.
        labels = self.exporter.get_labels()
        # For dict config, labels come from start_vertex and end_vertex.
        self.assertCountEqual(labels, ["StartLabel", "EndLabel"])

    def test_get_relationship_types_dict(self):
        types = self.exporter.get_relationship_types()
        self.assertEqual(types, ["connects"])

    def test_count_nodes_csv(self):
        # Count the number of rows (should be 2) in start_csv.
        count = self.exporter._count_nodes_csv(self.start_csv)
        self.assertEqual(count, 2)

    def test_fetch_nodes_chunk_csv(self):
        # Test fetching nodes chunk from start_csv.
        vertex_config = {"id": "id"}
        # Skip 0 rows, chunk_size 1 should return one row.
        chunk = self.exporter._fetch_nodes_chunk_csv(
            self.start_csv, skip=0, chunk_size=1, vertex_config=vertex_config
        )
        self.assertEqual(len(chunk), 1)
        self.assertEqual(chunk[0]["_elementid"], chunk[0]["id"])

    def test_fetch_nodes_by_ids_chunk_csv(self):
        # Test filtering nodes by IDs.
        vertex_config = {"id": "id"}
        rows = self.exporter._fetch_nodes_by_ids_chunk_csv(
            self.start_csv, node_ids=["2"], vertex_config=vertex_config
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "2")
        self.assertEqual(rows[0]["_elementid"], "2")

    def test_get_edge_csv_path_dict(self):
        # For dict config, if type matches, return edge csv path.
        path = self.exporter.get_edge_csv_path("connects")
        self.assertEqual(os.path.abspath(path), os.path.abspath(self.edge_csv))
        with self.assertRaises(ValueError):
            self.exporter.get_edge_csv_path("nonexistent")

    def test_count_edges(self):
        # Count rows in edge CSV.
        count = self.exporter._count_edges("connects")
        self.assertEqual(count, 2)

    def test_fetch_edge_chunk_csv(self):
        # Test fetching edge chunk.
        chunk = self.exporter._fetch_edge_chunk_csv("connects", skip=0, chunk_size=1)
        self.assertEqual(len(chunk), 1)
        self.assertNotIn('"Alice"', chunk[0].get("name", ""))

    def test_build_vertex_configs(self):
        # Test _build_vertex_configs for dict config.
        vc = self.exporter._build_vertex_configs()
        self.assertIn("StartLabel", vc)
        self.assertIn("EndLabel", vc)
        self.assertIsInstance(vc["StartLabel"], list)
        self.assertGreaterEqual(len(vc["StartLabel"]), 1)

    async def test_export_nodes_full(self):
        # Test export_nodes in full export (non-trial).
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            nodes_args = await self.exporter.export_nodes(pool)
        for label in ["StartLabel", "EndLabel"]:
            self.assertIn(label, nodes_args)
            self.assertEqual(
                nodes_args[label]["csv_path"], "dummy_output.csv".replace("\\", "\\\\")
            )
            self.assertEqual(nodes_args[label]["original_id"], "_elementid")
            # Expect next_val to equal "2" since each CSV has 2 rows.
            self.assertEqual(nodes_args[label]["next_val"], "2")
        self.assertIn("StartLabel", self.exporter.id_maps)
        self.assertIn("EndLabel", self.exporter.id_maps)

    async def test_export_edges(self):
        # Prepare id_maps for mapping start and end IDs.
        self.exporter.id_maps = {
            "StartLabel": {"1": 1001, "2": 1002},
            "EndLabel": {"1": 2001, "2": 2002},
        }
        # Create a temporary CSV file for edge export.
        temp_edge_csv = os.path.join(self.dir, "temp_edge.csv")
        with open(temp_edge_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "start_vertex_type",
                    "start_id",
                    "end_vertex_type",
                    "end_id",
                    "attr",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "start_vertex_type": "StartLabel",
                    "start_id": "1",
                    "end_vertex_type": "EndLabel",
                    "end_id": "2",
                    "attr": "foo",
                }
            )
            writer.writerow(
                {
                    "start_vertex_type": "StartLabel",
                    "start_id": "2",
                    "end_vertex_type": "EndLabel",
                    "end_id": "1",
                    "attr": "bar",
                }
            )
        # Patch get_edge_csv_path and _count_edges.
        self.exporter.get_edge_csv_path = (
            lambda rel_type: temp_edge_csv if rel_type == "connects" else ""
        )
        self.exporter._count_edges = lambda rel_type: 2
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            edges_args = await self.exporter.export_edges(pool)
        self.assertIn("connects", edges_args)
        self.assertEqual(
            edges_args["connects"]["csv_path"], "dummy_output.csv".replace("\\", "\\\\")
        )
        self.assertEqual(edges_args["connects"]["original_id"], "_elementid")
        # Expect next_val to equal "2" (two edge rows processed).
        self.assertEqual(edges_args["connects"]["next_val"], "2")
        os.remove(temp_edge_csv)

    async def test_list_nodes(self):
        # Create a temporary edge CSV for listing nodes.
        temp_edge_csv = os.path.join(self.dir, "temp_edge_list.csv")
        with open(temp_edge_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "start_vertex_type",
                    "start_id",
                    "end_vertex_type",
                    "end_id",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "start_vertex_type": "A",
                    "start_id": "1",
                    "end_vertex_type": "B",
                    "end_id": "2",
                }
            )
        self.exporter.get_edge_csv_path = (
            lambda rel_type: temp_edge_csv if rel_type == "connects" else ""
        )
        self.exporter._count_edges = lambda rel_type: 1
        self.exporter.get_relationship_types = lambda: ["connects"]
        await self.exporter.list_nodes()
        self.assertIn("connects", self.exporter.trial_nodes_by_label)
        nodes_by_label = self.exporter.trial_nodes_by_label["connects"]
        self.assertIn("A", nodes_by_label)
        self.assertIn("B", nodes_by_label)
        os.remove(temp_edge_csv)

    async def test_export_method(self):
        # Test the top-level export method.
        self.exporter.set_up_graph = AsyncMock(return_value=None)
        self.exporter.list_nodes = AsyncMock(return_value=None)
        self.exporter.export_nodes = AsyncMock(
            return_value={
                "v": {"csv_path": "v.csv", "original_id": "id", "next_val": "10"}
            }
        )
        self.exporter.export_edges = AsyncMock(
            return_value={
                "e": {"csv_path": "e.csv", "original_id": "id", "next_val": "5"}
            }
        )
        await self.exporter.export()
        self.assertEqual(
            self.exporter.vertices,
            {
                "v": {
                    "csv_path": "v.csv".replace("\\", "\\\\"),
                    "original_id": "id",
                    "next_val": "10",
                }
            },
        )
        self.assertEqual(
            self.exporter.edges,
            {
                "e": {
                    "csv_path": "e.csv".replace("\\", "\\\\"),
                    "original_id": "id",
                    "next_val": "5",
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
