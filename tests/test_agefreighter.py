#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for agefreighter.py
Aiming for > 50% coverage, all methods and DB connection parts are mocked as much as possible using MagicMock/AsyncMock.
"""

import sys
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
import os
from psycopg_pool import PoolTimeout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agefreighter.agefreighter import (
    Factory,
    AgeFreighter,
)

# Inject dummy modules into sys.modules so that Factory.create_instance works
dummy_module = type("dummy", (), {})()
dummy_class = type("DummyFreighter", (), {"__init__": lambda self: None})
dummy_module.AzureStorageFreighter = dummy_class
dummy_module.MultiAzureStorageFreighter = dummy_class
dummy_module.AvroFreighter = dummy_class
dummy_module.CosmosGremlinFreighter = dummy_class
dummy_module.CSVFreighter = dummy_class
dummy_module.MultiCSVFreighter = dummy_class
dummy_module.Neo4jFreighter = dummy_class
dummy_module.NetworkXFreighter = dummy_class
dummy_module.ParquetFreighter = dummy_class
dummy_module.PGFreighter = dummy_class

for mod in [
    ("agefreighter.azurestoragefreighter", dummy_module),
    ("agefreighter.multiazurestoragefreighter", dummy_module),
    ("agefreighter.avrofreighter", dummy_module),
    ("agefreighter.cosmosgremlinfreighter", dummy_module),
    ("agefreighter.csvfreighter", dummy_module),
    ("agefreighter.multicsvfreighter", dummy_module),
    ("agefreighter.neo4jfreighter", dummy_module),
    ("agefreighter.networkxfreighter", dummy_module),
    ("agefreighter.parquetfreighter", dummy_module),
    ("agefreighter.pgfreighter", dummy_module),
]:
    sys.modules[mod[0]] = mod[1]


# ----- Tests for Factory -----
class TestFactory(unittest.TestCase):
    def test_create_instance_unknown(self):
        # A ValueError should be raised when specifying an unknown type
        with self.assertRaises(ValueError) as context:
            Factory.create_instance("UnknownFreighter")
        self.assertIn("Unknown type", str(context.exception))

    def test_create_instance_valid(self):
        # Test one valid type; since we have injected dummy modules in sys.modules,
        # we expect a DummyFreighter instance.
        instance = Factory.create_instance("AzureStorageFreighter")
        self.assertEqual(type(instance).__name__, "AzureStorageFreighter")


# ----- Tests for synchronous methods -----
class TestAgeFreighterSync(unittest.TestCase):
    def setUp(self):
        self.af = AgeFreighter()
        self.af.graph_name = "mygraph"

    def test_checkKeys_valid(self):
        # No exception should be raised if keys contain the specified elements
        keys = ["a", "b", "c"]
        elements = ["a", "c"]
        try:
            AgeFreighter.checkKeys(
                keys, elements, "Error: expected {elements} columns, got {keys}"
            )
        except Exception as e:
            self.fail(f"checkKeys() raised Exception unexpectedly: {e}")

    def test_checkKeys_invalid(self):
        # A ValueError should be raised if keys do not include all the specified elements
        keys = ["a", "b"]
        elements = ["a", "b", "c"]
        with self.assertRaises(ValueError):
            AgeFreighter.checkKeys(
                keys, elements, "Error: expected {elements} columns, got {keys}"
            )

    def test_quotedGraphName_already_lower(self):
        self.assertEqual(AgeFreighter.quotedGraphName("mygraph"), "mygraph")

    def test_quotedGraphName_not_lower(self):
        self.assertEqual(AgeFreighter.quotedGraphName("MyGraph"), '"MyGraph"')

    def test_createEdgeCypher_without_props(self):
        # Create a test row as a Series
        row = pd.Series(
            {"start_v_label": "A", "start_id": "1", "end_v_label": "B", "end_id": "2"}
        )
        query = self.af.createEdgeCypher(row, "RELATE", [])
        expected = """MATCH (n:{0} {{id: '{1}'}}),
                        (m:{2} {{id: '{3}'}})
                        CREATE (n)-[:{4}]->(m)""".format("A", "1", "B", "2", "RELATE")
        # Allow differences in whitespace and newlines, so compare stripped strings
        self.assertEqual(query.strip(), expected.strip())

    def test_createEdgeCypher_with_props(self):
        row = pd.Series(
            {
                "start_v_label": "A",
                "start_id": "1",
                "end_v_label": "B",
                "end_id": "2",
                "prop1": "val1",
                "prop2": "val2",
            }
        )
        query = self.af.createEdgeCypher(row, "RELATE", ["prop1", "prop2"])
        props = "prop1:'val1',prop2:'val2'"
        expected = """MATCH (n:{0} {{id: '{1}'}}),
                        (m:{2} {{id: '{3}'}})
                        CREATE (n)-[:{4} {{{5}}}]->(m)""".format(
            "A", "1", "B", "2", "RELATE", props
        )
        self.assertEqual(query.strip(), expected.strip())

    def test_processChunkDirectly_and_createValuesDirectly(self):
        # id_maps provided as a dictionary
        id_maps = {"A": {"1": "100"}, "B": {"2": "200"}}

        # simple DataFrame
        df = pd.DataFrame(
            {
                "start_v_label": ["A"],
                "start_id": ["1"],
                "end_v_label": ["B"],
                "end_id": ["2"],
                "edge_prop1": ["val"],
            }
        )
        # Configuration
        self.af.graph_name = "mygraph"

        graph_name = self.af.quotedGraphName(self.af.graph_name)
        edge_type = "RELATE"
        # The query to use
        query = f'INSERT INTO {graph_name}."{edge_type}" (start_id, end_id, properties) VALUES {{values}};'

        # processChunkDirectly calls createValuesDirectly for each row
        parts = self.af.processChunkDirectly(df, query, ["edge_prop1"], id_maps)
        # Expecting a single element in the returned list
        self.assertEqual(len(parts), 1)
        # Check the values generated by createValuesDirectly
        value_str = self.af.createValuesDirectly(df.iloc[0], ["edge_prop1"], id_maps)
        expected_value = (
            "('100'::graphid, '200'::graphid, '{\"edge_prop1\":\"val\"}'::agtype)"
        )
        self.assertEqual(value_str, expected_value)

    def test_showProgress(self):
        # Capture the standard output
        from io import StringIO

        captured_output = StringIO()
        sys_stdout = sys.stdout
        try:
            sys.stdout = captured_output
            self.af.showProgress("test", 5, 10)
            output = captured_output.getvalue().strip()
            self.assertIn("Loading test: 5/10", output)
        finally:
            sys.stdout = sys_stdout


# ----- Additional tests to cover more parts of AgeFreighter -----
class TestAgeFreighterAdditionalSync(unittest.TestCase):
    def setUp(self):
        self.af = AgeFreighter()
        # Dummy pool so that we can call synchronous parts
        self.af.pool = MagicMock()

    def test___aenter__sets_defaults(self):
        # Call __aenter__ and verify that variables are set (even though they are overwritten with the same values)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.af.__aenter__())
        self.assertEqual(self.af.dsn, "")
        self.assertEqual(self.af.graph_name, "")

    def test_createVertices_direct_loading(self):
        # Patch createVerticesDirectly and simulate direct_loading branch
        self.af.createVerticesDirectly = AsyncMock()
        df = pd.DataFrame({"col1": ["a"], "col2": ["b"]})
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self.af.createVertices(
                vertices=df, vertex_label="V", chunk_size=1, direct_loading=True
            )
        )
        self.af.createVerticesDirectly.assert_called_once()

    def test_createVertices_cypher_mode(self):
        # Test the branch where direct_loading is False and use_copy is False.
        self.af.createVerticesCypher = AsyncMock()
        df = pd.DataFrame({"col1": ["a"], "col2": ["b"]})
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self.af.createVertices(
                vertices=df,
                vertex_label="V",
                chunk_size=1,
                direct_loading=False,
                use_copy=False,
            )
        )
        self.af.createVerticesCypher.assert_called_once()

    def test_createEdges_direct_loading(self):
        self.af.createEdgesDirectly = AsyncMock()
        df = pd.DataFrame(
            {
                "start_v_label": ["A"],
                "start_id": ["1"],
                "end_v_label": ["B"],
                "end_id": ["2"],
                "prop": ["x"],
            }
        )
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self.af.createEdges(
                edges=df,
                edge_type="REL",
                edge_props=["prop"],
                chunk_size=1,
                direct_loading=True,
            )
        )
        self.af.createEdgesDirectly.assert_called_once()

    def test_createEdges_cypher_mode(self):
        self.af.createEdgesCypher = AsyncMock()
        df = pd.DataFrame(
            {
                "start_v_label": ["A"],
                "start_id": ["1"],
                "end_v_label": ["B"],
                "end_id": ["2"],
                "prop": ["y"],
            }
        )
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            self.af.createEdges(
                edges=df,
                edge_type="REL",
                edge_props=["prop"],
                chunk_size=1,
                direct_loading=False,
                use_copy=False,
            )
        )
        self.af.createEdgesCypher.assert_called_once()


# ----- Tests for asynchronous methods -----
class TestAgeFreighterAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.af = AgeFreighter()
        self.af.graph_name = "mygraph"
        # Create a dummy connection and cursor:
        self.mock_cursor = MagicMock()
        self.mock_cursor.__aenter__ = AsyncMock(return_value=self.mock_cursor)
        self.mock_cursor.__aexit__ = AsyncMock(return_value=None)

        self.mock_conn = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor

        # For the pool.connection() async context manager, use a helper that yields self.mock_conn:
        async def aenter():
            return self.mock_conn

        async def aexit(exc_type, exc, tb):
            pass

        async_cm = MagicMock()
        async_cm.__aenter__.side_effect = aenter
        async_cm.__aexit__.side_effect = aexit
        self.af.pool = MagicMock()
        self.af.pool.connection.return_value = async_cm

    def _get_async_cm(self, mock_obj):
        # Return a simple asynchronous context manager
        async def aenter():
            return mock_obj

        async def aexit(exc_type, exc_val, tb):
            pass

        cm = MagicMock()
        cm.__aenter__.side_effect = aenter
        cm.__aexit__.side_effect = aexit
        return cm

    async def test_aexit_calls_pool_close(self):
        # Verify that pool.close() is called inside __aexit__
        self.af.pool.close = AsyncMock()
        await self.af.__aexit__(None, None, None)
        self.af.pool.close.assert_called_once()

    async def test_connect_success(self):
        # Simulate the scenario where pool.open() and pool.wait() complete successfully.
        with patch("agefreighter.agefreighter.AsyncConnectionPool") as mock_pool_cls:
            mock_pool_instance = MagicMock()
            mock_pool_instance.open = AsyncMock()
            mock_pool_instance.wait = AsyncMock()
            mock_pool_cls.return_value = mock_pool_instance
            dsn = "dbname=test"
            await self.af.connect(dsn=dsn, max_connections=10, min_connections=1)
            # Verify that the connection pool was created, and open and wait were called.
            mock_pool_instance.open.assert_called_once()
            mock_pool_instance.wait.assert_called_once()

    async def test_connect_pooltimeout(self):
        with patch("asyncio.sleep", new=AsyncMock()):
            with patch(
                "agefreighter.agefreighter.AsyncConnectionPool"
            ) as mock_pool_cls:
                mock_pool_instance = MagicMock()
                mock_pool_instance.open = AsyncMock(side_effect=PoolTimeout("Timeout"))
                mock_pool_instance.wait = AsyncMock()
                mock_pool_cls.return_value = mock_pool_instance
                with self.assertRaises(PoolTimeout):
                    await self.af.connect(dsn="dbname=test")
                self.assertEqual(mock_pool_instance.open.call_count, 3)

    async def test_close_calls_pool_close(self):
        self.af.pool.close = AsyncMock()
        await self.af.close()
        self.af.pool.close.assert_called_once()

    async def test_setUpGraph_existing_create(self):
        # Simulate the case where the graph already exists and create_graph=True.
        row_mock = MagicMock()
        row_mock.count = 1
        self.mock_cursor.execute = AsyncMock()
        self.mock_cursor.fetchone = AsyncMock(return_value=row_mock)
        await self.af.setUpGraph(graph_name="TestGraph", create_graph=True)
        self.assertGreaterEqual(self.mock_cursor.execute.call_count, 3)

    async def test_setUpGraph_existing_no_create(self):
        row_mock = MagicMock()
        row_mock.count = 1
        self.mock_cursor.execute = AsyncMock()
        self.mock_cursor.fetchone = AsyncMock(return_value=row_mock)
        await self.af.setUpGraph(graph_name="TestGraph", create_graph=False)
        self.assertGreaterEqual(self.mock_cursor.execute.call_count, 2)

    async def test_setUpGraph_not_exist_and_create(self):
        self.mock_cursor.execute = AsyncMock()
        self.mock_cursor.fetchone = AsyncMock(return_value=None)
        await self.af.setUpGraph(graph_name="TestGraph", create_graph=True)
        self.assertGreaterEqual(self.mock_cursor.execute.call_count, 1)

    async def test_setUpGraph_not_exist_no_create(self):
        self.mock_cursor.execute = AsyncMock()
        self.mock_cursor.fetchone = AsyncMock(return_value=None)
        with self.assertRaises(ValueError):
            await self.af.setUpGraph(graph_name="TestGraph", create_graph=False)

    async def test_createLabelType_vertex(self):
        self.mock_cursor.execute = AsyncMock()
        await self.af.createLabelType(label_type="vertex", value="TestVertex")
        self.assertGreaterEqual(self.mock_cursor.execute.call_count, 3)

    async def test_createLabelType_edge(self):
        self.mock_cursor.execute = AsyncMock()
        await self.af.createLabelType(label_type="edge", value="TestEdge")
        self.assertGreaterEqual(self.mock_cursor.execute.call_count, 3)

    async def test_executeQuery_success(self):
        self.mock_cursor.execute = AsyncMock()
        query = "SELECT 1;"
        await self.af.executeQuery(query)
        self.mock_cursor.execute.assert_called_with(query)

    async def test_executeQuery_exception(self):
        self.mock_cursor.execute = AsyncMock(side_effect=Exception("TestError"))
        try:
            await self.af.executeQuery("SELECT 1;")
        except Exception:
            self.fail("executeQuery() should not propagate exceptions.")

    async def test_executeWithTasks(self):
        results = []

        async def dummy_target(arg):
            results.append(arg)

        args = ["a", "b", "c"]
        await self.af.executeWithTasks(dummy_target, args)
        self.assertCountEqual(results, args)

    async def test_getFirstId_success(self):
        fake_id = 1234
        self.mock_cursor.execute = AsyncMock()
        self.mock_cursor.fetchone = AsyncMock(return_value=(fake_id,))
        first_id = await self.af.getFirstId(graph_name="mygraph", label_type="vertex")
        self.assertIsInstance(first_id, int)

    async def test_getFirstId_no_row(self):
        self.mock_cursor.execute = AsyncMock()
        self.mock_cursor.fetchone = AsyncMock(return_value=None)
        with self.assertRaises(ValueError):
            await self.af.getFirstId(graph_name="mygraph", label_type="vertex")

    async def test_createGraphFromDataFrame(self):
        src = pd.DataFrame(
            {
                "start": ["1", "2"],
                "sprop": ["a", "b"],
                "end": ["3", "4"],
                "eprop": ["c", "d"],
            }
        )
        self.af.checkKeys = MagicMock()
        self.af.setUpGraph = AsyncMock()
        self.af.createLabelType = AsyncMock()
        self.af.createVertices = AsyncMock()
        self.af.createEdges = AsyncMock()
        await self.af.createGraphFromDataFrame(
            graph_name="mygraph",
            src=src,
            existing_node_ids=[],
            first_chunk=True,
            start_v_label="SV",
            start_id="start",
            start_props=["sprop"],
            edge_type="REL",
            edge_props=["eprop"],
            end_v_label="EV",
            end_id="end",
            end_props=[],
            chunk_size=1,
            direct_loading=False,
            create_graph=True,
            use_copy=False,
        )
        self.af.checkKeys.assert_called_once()
        self.af.setUpGraph.assert_called_once()
        self.assertEqual(self.af.createLabelType.call_count, 3)
        self.assertEqual(self.af.createVertices.call_count, 2)
        self.af.createEdges.assert_called_once()

    # ----- Additional tests to increase coverage for COPY methods and cypher/directed methods -----
    async def test_createVerticesCypher(self):
        # Patch executeWithTasks to record the queries that would be executed.
        self.af.executeWithTasks = AsyncMock()
        # Build a dataframe with one row.
        df = pd.DataFrame({"a": ["val"], "b": ["val2"]})
        self.af.progress = True
        self.af.graph_name = "graph1"
        await self.af.createVerticesCypher(df, "V", chunk_size=1)
        self.af.executeWithTasks.assert_called()

    async def test_createEdgesCypher(self):
        self.af.executeWithTasks = AsyncMock()
        # Build a dataframe with one row containing necessary columns.
        df = pd.DataFrame(
            {
                "start_v_label": ["A"],
                "start_id": ["1"],
                "end_v_label": ["B"],
                "end_id": ["2"],
                "prop": ["x"],
            }
        )
        self.af.progress = True
        await self.af.createEdgesCypher(df, "REL", ["prop"], chunk_size=1)
        self.af.executeWithTasks.assert_called()

    async def test_createVerticesDirectly(self):
        self.af.executeWithTasks = AsyncMock()
        df = pd.DataFrame({"a": ["val"], "b": ["val2"]})
        self.af.progress = True
        self.af.graph_name = "graph1"
        await self.af.createVerticesDirectly(df, "V", chunk_size=1)
        self.af.executeWithTasks.assert_called()

    async def test_createEdgesDirectly(self):
        self.af.executeWithTasks = AsyncMock()
        id_maps = {"A": {"1": "100"}, "B": {"2": "200"}}
        # patch getIdMaps to return a dummy mapping
        self.af.getIdMaps = AsyncMock(return_value=id_maps)
        df = pd.DataFrame(
            {
                "start_v_label": ["A"],
                "start_id": ["1"],
                "end_v_label": ["B"],
                "end_id": ["2"],
                "edge_prop1": ["val"],
            }
        )
        self.af.progress = True
        self.af.graph_name = "graph1"
        await self.af.createEdgesDirectly(df, "REL", ["edge_prop1"], chunk_size=1)
        self.af.executeWithTasks.assert_called()

    async def test_copyChunk_vertices(self):
        # Test copyChunk for vertices (is_edge=False). We simulate copy context.
        fake_copy = AsyncMock()
        fake_copy.write = AsyncMock()

        fake_cursor = MagicMock()
        fake_cursor.__aenter__ = AsyncMock(return_value=fake_cursor)
        fake_cursor.__aexit__ = AsyncMock(return_value=None)
        fake_cursor.copy.return_value.__aenter__.return_value = fake_copy

        fake_conn = MagicMock()
        fake_conn.cursor.return_value = fake_cursor

        async def aenter_conn():
            return fake_conn

        cm = MagicMock()
        cm.__aenter__.side_effect = aenter_conn
        cm.__aexit__.return_value = asyncio.sleep(0)

        self.af.pool.connection.return_value = cm
        # Build a simple df with one row.
        df = pd.DataFrame({"x": ["1"]})
        await self.af.copyChunk(
            chunk=df,
            first_id=10,
            graph_name="graph1",
            label="V",
            id_maps={},  # not used for vertices copy
            is_edge=False,
        )
        fake_copy.write.assert_called()  # check that some write happened

    async def test_copyVertices(self):
        # Patch getFirstId to return a fixed value.
        self.af.getFirstId = AsyncMock(return_value=100)
        fake_copy = AsyncMock()
        fake_copy.write = AsyncMock()

        fake_cursor = MagicMock()
        fake_cursor.__aenter__ = AsyncMock(return_value=fake_cursor)
        fake_cursor.__aexit__ = AsyncMock(return_value=None)
        fake_cursor.copy.return_value.__aenter__.return_value = fake_copy

        fake_cursor.copy.return_value.__aexit__.return_value = None
        fake_conn = MagicMock()
        fake_conn.cursor.return_value = fake_cursor

        async def aenter_conn():
            return fake_conn

        cm = MagicMock()
        cm.__aenter__.side_effect = aenter_conn
        cm.__aexit__.return_value = asyncio.sleep(0)
        self.af.pool.connection.return_value = cm
        df = pd.DataFrame({"col": ["val"]})
        await self.af.copyVertices(df, "V", chunk_size=1)
        fake_copy.write.assert_called()

    async def test_copyEdges(self):
        # Patch getFirstId and getIdMaps to return fixed values.
        self.af.getFirstId = AsyncMock(return_value=200)
        self.af.getIdMaps = AsyncMock(
            return_value={"A": {"1": "101"}, "B": {"2": "202"}}
        )
        fake_copy = AsyncMock()
        fake_copy.write = AsyncMock()
        fake_cursor = MagicMock()
        fake_cursor.__aenter__ = AsyncMock(return_value=fake_cursor)
        fake_cursor.__aexit__ = AsyncMock(return_value=None)
        fake_cursor.copy.return_value.__aenter__.return_value = fake_copy
        fake_conn = MagicMock()
        fake_conn.cursor.return_value = fake_cursor

        async def aenter_conn():
            return fake_conn

        cm = MagicMock()
        cm.__aenter__.side_effect = aenter_conn
        cm.__aexit__.return_value = asyncio.sleep(0)
        self.af.pool.connection.return_value = cm
        df = pd.DataFrame(
            {
                "start_v_label": ["A"],
                "start_id": ["1"],
                "end_v_label": ["B"],
                "end_id": ["2"],
            }
        )
        await self.af.copyEdges(df, "REL", edge_props=[], chunk_size=1)
        fake_copy.write.assert_called()


# ----- Additional tests for Factory or other components can be added here -----

if __name__ == "__main__":
    unittest.main()
