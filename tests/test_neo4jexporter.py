#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import concurrent.futures
import unittest
import sys
import os
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the module under test.
from agefreighter.neo4jexporter import Neo4jExporter


# --- Dummy Classes to simulate Neo4j driver behavior ---
class DummyResult:
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def single(self):
        return self.records[0] if self.records else None


class DummyNode:
    def __init__(
        self, element_id: str, properties: Dict[str, Any], labels: List[str] = None
    ):
        self.element_id = element_id
        self._properties = properties
        self.labels = labels if labels is not None else []


class DummyRelationship:
    def __init__(self, element_id: str, properties: Dict[str, Any]):
        self.element_id = element_id
        self._properties = properties


class DummySession:
    def __init__(self, query_handler):
        """
        query_handler: a function that receives the query (and kwargs)
                       and returns a DummyResult.
        """
        self.query_handler = query_handler

    def run(self, query: str, **kwargs) -> DummyResult:
        return self.query_handler(query, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self, session_callable):
        self.session_callable = session_callable
        self.closed = False

    def session(self):
        return self.session_callable()

    def close(self):
        self.closed = True


# --- Test Suite for Neo4jExporter ---
class TestNeo4jExporter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Patch GraphDatabase.driver so that our exporter uses a dummy driver.
        patcher = patch(
            "agefreighter.neo4jexporter.GraphDatabase.driver",
            return_value=DummyDriver(
                lambda: DummySession(lambda q, **kw: DummyResult([]))
            ),
        )
        self.addCleanup(patcher.stop)
        self.mock_driver = patcher.start()

        # Create an instance of Neo4jExporter with dummy parameters.
        self.exporter = Neo4jExporter(
            dsn="dummy_dsn",
            min_connections=1,
            max_connections=1,
            uri="bolt://dummy",
            user="dummy_user",
            password="dummy_pass",
            database="dummy_db",
            trial=False,
            save_temps=False,
            progress=False,
            graph_name="dummy_graph",
            chunk_size=2,  # use small chunk for testing
            log_level=0,
        )
        # Patch inherited asynchronous methods.
        self.exporter.create_label_type = AsyncMock(return_value=None)
        self.exporter.get_first_id = AsyncMock(return_value=1000)
        self.exporter.write_csv = AsyncMock(return_value="dummy_output.csv")
        self.exporter.set_up_graph = AsyncMock(return_value=None)
        # Initialize id_maps
        self.exporter.id_maps = {}

    async def test_init_success(self):
        # __init__ should have set driver (using patched GraphDatabase.driver).
        self.assertIsNotNone(self.exporter.driver)

    async def test_aenter_aexit(self):
        # Test async context manager calls driver.close()
        dummy_drv = DummyDriver(lambda: DummySession(lambda q, **kw: DummyResult([])))
        self.exporter.driver = dummy_drv
        self.exporter.connect = AsyncMock(return_value=None)
        async with self.exporter as exp:
            self.assertIs(exp, self.exporter)
        self.assertTrue(dummy_drv.closed)

    def test_get_labels_no_unlabeled(self):
        # Test get_labels when no unlabeled nodes are found.
        def query_handler(query: str, **kwargs):
            if query.startswith("CALL db.labels()"):
                return DummyResult([{"label": "Person"}])
            elif query.startswith("MATCH (n) WHERE size(labels(n))"):
                return DummyResult([{"cnt": 0}])
            return DummyResult([])

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        labels = self.exporter.get_labels()
        self.assertEqual(labels, ["Person"])

    def test_get_labels_with_unlabeled(self):
        # Test get_labels when there are unlabeled nodes.
        def query_handler(query: str, **kwargs):
            if query.startswith("CALL db.labels()"):
                return DummyResult([{"label": "Person"}])
            elif query.startswith("MATCH (n) WHERE size(labels(n))"):
                return DummyResult([{"cnt": 3}])
            return DummyResult([])

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        labels = self.exporter.get_labels()
        self.assertEqual(labels, ["Person", "NO_LABEL"])

    def test_get_labels_exception(self):
        # Simulate exception in get_labels.
        def query_handler(query: str, **kwargs):
            raise Exception("fail")

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        labels = self.exporter.get_labels()
        self.assertEqual(labels, [])

    def test_get_relationship_types(self):
        # Test get_relationship_types returning list.
        def query_handler(query: str, **kwargs):
            return DummyResult(
                [{"relationshipType": "FRIEND"}, {"relationshipType": "COLLEAGUE"}]
            )

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        types = self.exporter.get_relationship_types()
        self.assertEqual(types, ["FRIEND", "COLLEAGUE"])

    def test_get_relationship_types_exception(self):
        def query_handler(query: str, **kwargs):
            raise Exception("fail")

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        types = self.exporter.get_relationship_types()
        self.assertEqual(types, [])

    def test_count_nodes_string(self):
        # For a normal label.
        def query_handler(query: str, **kwargs):
            return DummyResult([{"cnt": 10}])

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        count = self.exporter._count_nodes("Person")
        self.assertEqual(count, 10)

    def test_count_nodes_no_label(self):
        def query_handler(query: str, **kwargs):
            self.assertTrue("size(labels(n)) = 0" in query)
            return DummyResult([{"cnt": 5}])

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        count = self.exporter._count_nodes("NO_LABEL")
        self.assertEqual(count, 5)

    def test_count_nodes_list(self):
        def query_handler(query: str, **kwargs):
            self.assertTrue("ANY(lbl IN labels(n)" in query)
            return DummyResult([{"cnt": 7}])

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        count = self.exporter._count_nodes(["Person", "Animal"])
        self.assertEqual(count, 7)

    def test_count_nodes_exception(self):
        def query_handler(query: str, **kwargs):
            raise Exception("fail")

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        count = self.exporter._count_nodes("Person")
        self.assertEqual(count, 0)

    def test_count_edges(self):
        def query_handler(query: str, **kwargs):
            self.assertIn("MATCH ()-[r:LIKES]->()", query)
            return DummyResult([{"cnt": 4}])

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        count = self.exporter._count_edges("LIKES")
        self.assertEqual(count, 4)

    def test_count_edges_exception(self):
        def query_handler(query: str, **kwargs):
            raise Exception("fail")

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        count = self.exporter._count_edges("LIKES")
        self.assertEqual(count, 0)

    def test_fetch_nodes_chunk(self):
        # Simulate a query returning one node.
        def query_handler(query: str, **kwargs):
            node = DummyNode("node1", {"name": "Alice"})
            return DummyResult([{"n": node}])

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        result = self.exporter._fetch_nodes_chunk("MATCH (n:Person) RETURN n")
        self.assertEqual(result, [{"_elementid": "node1", "name": "Alice"}])

    def test_fetch_nodes_chunk_exception(self):
        def query_handler(query: str, **kwargs):
            raise Exception("fail")

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        result = self.exporter._fetch_nodes_chunk("MATCH (n:Person) RETURN n")
        self.assertEqual(result, [])

    def test_fetch_nodes_by_ids_chunk(self):
        # Simulate fetching nodes by IDs.
        def query_handler(query: str, **kwargs):
            self.assertIn("elementId(n) IN", query)
            node = DummyNode("node2", {"name": "Bob"})
            return DummyResult([{"n": node}])

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        result = self.exporter._fetch_nodes_by_ids_chunk("Person", ["node2"])
        self.assertEqual(result, [{"_elementid": "node2", "name": "Bob"}])

    def test_fetch_nodes_by_ids_chunk_exception(self):
        def query_handler(query: str, **kwargs):
            raise Exception("fail")

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        result = self.exporter._fetch_nodes_by_ids_chunk("Person", ["node2"])
        self.assertEqual(result, [])

    def test_fetch_edge_chunk(self):
        # Simulate an edge query returning one record.
        def query_handler(query: str, **kwargs):
            m = DummyNode("m1", {}, labels=["StartLabel"])
            n = DummyNode("n1", {}, labels=["EndLabel"])
            r = DummyRelationship("r1", {"rprop": "val"})
            return DummyResult([{"m": m, "r": r, "n": n}])

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        result = self.exporter._fetch_edge_chunk("LIKES", skip=0, limit=10)
        expected = [
            {
                "_elementid": "r1",
                "start_id": "m1",
                "start_vertex_type": "StartLabel",
                "end_id": "n1",
                "end_vertex_type": "EndLabel",
                "rprop": "val",
            }
        ]
        self.assertEqual(result, expected)

    def test_fetch_edge_chunk_exception(self):
        def query_handler(query: str, **kwargs):
            raise Exception("fail")

        self.exporter.driver = DummyDriver(lambda: DummySession(query_handler))
        result = self.exporter._fetch_edge_chunk("LIKES", skip=0, limit=10)
        self.assertEqual(result, [])

    async def test_export_nodes_non_trial(self):
        # Test export_nodes when not in trial mode.
        self.exporter.get_labels = MagicMock(return_value=["TestLabel"])
        self.exporter._count_nodes = MagicMock(return_value=2)
        self.exporter.chunk_size = 1  # force two iterations
        self.exporter._fetch_nodes_chunk = MagicMock(
            side_effect=[
                [{"_elementid": "n1", "prop": "val1"}],
                [{"_elementid": "n2", "prop": "val2"}],
            ]
        )
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            nodes_args = await self.exporter.export_nodes(pool)
        # Expect vertex_args to have the new CSV spec using "next_val".
        self.assertIn("TestLabel", nodes_args)
        self.assertEqual(
            nodes_args["TestLabel"]["csv_path"],
            "dummy_output.csv".replace("\\", "\\\\"),
        )
        # Expect next_val to equal "2" (number of nodes exported)
        self.assertEqual(nodes_args["TestLabel"]["next_val"], "2")
        self.assertIn("TestLabel", self.exporter.id_maps)
        self.assertEqual(len(self.exporter.id_maps["TestLabel"]), 2)

    async def test_export_edges(self):
        # Prepare id_maps so that edge export can map start and end IDs.
        self.exporter.id_maps = {"A": {"s1": 101}, "B": {"e1": 202}}
        self.exporter.get_relationship_types = MagicMock(return_value=["REL"])
        self.exporter._count_edges = MagicMock(return_value=2)
        self.exporter._fetch_edge_chunk = MagicMock(
            return_value=[
                {
                    "start_vertex_type": "A",
                    "start_id": "s1",
                    "end_vertex_type": "B",
                    "end_id": "e1",
                    "attr": "x",
                }
            ]
        )
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            edges_args = await self.exporter.export_edges(pool)
        self.assertIn("REL", edges_args)
        self.assertEqual(
            edges_args["REL"]["csv_path"], "dummy_output.csv".replace("\\", "\\\\")
        )
        # Expect next_val to be the number of edges processed (1 in this case).
        self.assertEqual(edges_args["REL"]["next_val"], "1")

    async def test_list_nodes(self):
        # Test list_nodes in trial mode.
        self.exporter.get_relationship_types = MagicMock(return_value=["REL"])
        self.exporter._count_edges = MagicMock(return_value=1)
        self.exporter._fetch_edge_chunk = MagicMock(
            return_value=[
                {
                    "start_vertex_type": "A",
                    "start_id": "s1",
                    "end_vertex_type": "B",
                    "end_id": "e1",
                }
            ]
        )
        await self.exporter.list_nodes()
        self.assertIn("REL", self.exporter.trial_nodes_by_label)
        mapping = self.exporter.trial_nodes_by_label["REL"]
        self.assertEqual(mapping.get("A"), ["s1"])
        self.assertEqual(mapping.get("B"), ["e1"])

    async def test_export_method(self):
        # Test the top-level export method.
        self.exporter.set_up_graph = AsyncMock(return_value=None)
        self.exporter.list_nodes = AsyncMock(return_value=None)
        self.exporter.export_nodes = AsyncMock(
            return_value={"v": {"csv_path": "v.csv", "next_val": "10"}}
        )
        self.exporter.export_edges = AsyncMock(
            return_value={"e": {"csv_path": "e.csv", "next_val": "5"}}
        )
        dummy_drv = DummyDriver(lambda: DummySession(lambda q, **kw: DummyResult([])))
        self.exporter.driver = dummy_drv
        await self.exporter.export()
        self.assertEqual(
            self.exporter.vertices,
            {"v": {"csv_path": "v.csv".replace("\\", "\\\\"), "next_val": "10"}},
        )
        self.assertEqual(
            self.exporter.edges,
            {"e": {"csv_path": "e.csv".replace("\\", "\\\\"), "next_val": "5"}},
        )
        self.assertTrue(dummy_drv.closed)


if __name__ == "__main__":
    unittest.main()
