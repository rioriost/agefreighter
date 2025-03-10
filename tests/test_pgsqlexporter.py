import asyncio
import copy
import json
import logging
import os
import unittest
from unittest.mock import AsyncMock, patch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agefreighter.pgsqlexporter import (
    PGSQLExporter,
    ConfigManager,
)


class FakeRow:
    """A fake row object that simulates a namedtuple row with _asdict()."""

    def __init__(self, data):
        self.data = data

    def _asdict(self):
        return self.data

    def __getattr__(self, attr):
        return self.data.get(attr)

    def __getitem__(self, key):
        return self.data[key]

    def items(self):
        return self.data.items()

    def __contains__(self, key):
        return key in self.data

    def __iter__(self):
        return iter(self.data)


class FakeCursor:
    """A fake async cursor for simulating DB responses."""

    def __init__(self, fetchone_result=None, fetchall_result=None):
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.last_query = None
        self.last_params = None

    async def execute(self, query, params=None):
        self.last_query = query
        self.last_params = params

    async def fetchone(self):
        return self.fetchone_result

    async def fetchall(self):
        return self.fetchall_result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class FakeConnection:
    """A fake async connection for simulating a DB connection."""

    def __init__(self, cursor):
        self.cursor_instance = cursor

    def cursor(self, row_factory=None):
        # row_factory is ignored in this fake implementation.
        return self.cursor_instance

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class FakePool:
    """A fake async connection pool."""

    def __init__(self, cursor):
        self.cursor = cursor
        self.closed = False

    # Change this method from async to a normal method.
    def connection(self):
        return FakeConnection(self.cursor)

    async def close(self):
        self.closed = True


# --- Dummy Config JSON for Testing --- #

dummy_config = {
    "edge": {
        "table": "edge_table",
        "start_vertex": {
            "table": "start_table",
            "id": "start_id",
            "label": "Start",
            "props": {},
        },
        "end_vertex": {
            "table": "end_table",
            "id": "end_id",
            "label": "End",
            "props": {},
        },
        "type": "REL_TYPE",
        "props": {},
        "start_id": "start_id",
        "end_id": "end_id",
    }
}

dummy_config_list = {
    "edge": [
        {
            "table": "edge_table",
            "start_vertex": {
                "table": "start_table",
                "id": "start_id",
                "label": "Start",
                "props": {},
            },
            "end_vertex": {
                "table": "end_table",
                "id": "end_id",
                "label": "End",
                "props": {},
            },
            "type": "REL_TYPE",
            "props": {},
        }
    ]
}


# --- Tests for ConfigManager --- #


class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # Fake pool that simulates table exists by returning (True,)
        self.fake_cursor = FakeCursor(fetchone_result=(True,))
        self.fake_pool = FakePool(self.fake_cursor)
        # Prepare temporary file names for config tests
        self.config_filename = "temp_config.json"

    def tearDown(self):
        if os.path.exists(self.config_filename):
            os.remove(self.config_filename)

    def write_temp_config(self, config_data):
        with open(self.config_filename, "w", encoding="utf-8") as f:
            json.dump(config_data, f)
        return self.config_filename

    def test_require_key_success(self):
        manager = ConfigManager("dummy", self.fake_pool)
        config = {"key": "value"}
        value = manager.require_key(config, "key")
        self.assertEqual(value, "value")

    def test_require_key_missing(self):
        manager = ConfigManager("dummy", self.fake_pool)
        config = {"key": "value"}
        with self.assertRaises(ValueError) as ctx:
            manager.require_key(config, "missing")
        self.assertIn("Missing 'missing' key", str(ctx.exception))

    def test_load_config_dict_success(self):
        filename = self.write_temp_config(dummy_config)
        manager = ConfigManager(filename, self.fake_pool)
        config = asyncio.run(manager.load_config())
        self.assertEqual(config, dummy_config)
        # For a dict edge config the parse_result is set to a specific message.
        self.assertEqual(
            manager.parse_result, "Config has single edge and multiple nodes"
        )

    def test_load_config_list_success(self):
        filename = self.write_temp_config(dummy_config_list)
        manager = ConfigManager(filename, self.fake_pool)
        config = asyncio.run(manager.load_config())
        self.assertEqual(config, dummy_config_list)
        # The parse_result could be one of two values.
        self.assertIn(
            manager.parse_result,
            [
                "Config has multiple edges and single node",
                "Config has multiple edges and multiple nodes",
            ],
        )

    def test_load_config_invalid_json(self):
        # Write an invalid JSON file.
        with open(self.config_filename, "w", encoding="utf-8") as f:
            f.write("invalid json")
        manager = ConfigManager(self.config_filename, self.fake_pool)
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(manager.load_config())
        self.assertIn("Invalid JSON format", str(ctx.exception))

    def test_load_config_missing_edge(self):
        bad_config = {}
        filename = self.write_temp_config(bad_config)
        manager = ConfigManager(filename, self.fake_pool)
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(manager.load_config())
        self.assertIn("Missing 'edge' key", str(ctx.exception))

    def test_load_config_table_not_exists(self):
        # Simulate table does not exist by having fetchone return (False,)
        fake_cursor = FakeCursor(fetchone_result=(False,))
        fake_pool = FakePool(fake_cursor)
        filename = self.write_temp_config(dummy_config)
        manager = ConfigManager(filename, fake_pool)
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(manager.load_config())
        self.assertIn("does not exist", str(ctx.exception))


# --- Tests for PGSQLExporter --- #


# We use IsolatedAsyncioTestCase to support async test methods.
class TestPGSQLExporter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Prepare a fake pool with a cursor that returns a count for _count_rows tests.
        self.fake_count_cursor = FakeCursor(fetchone_result=(10,))
        self.fake_pool = FakePool(self.fake_count_cursor)

        with open("dummy_config.json", "w", encoding="utf-8") as f:
            json.dump(dummy_config, f)

        # Create an instance of PGSQLExporter.
        # (We pass dummy values; note that the AgeFreighter base methods will be patched below.)
        self.exporter = PGSQLExporter(
            dsn="dummy_dsn",
            min_connections=1,
            max_connections=1,
            src_dsn="dummy_src_dsn",
            config="dummy_config.json",
            trial=False,
            save_temps=False,
            progress=False,
            graph_name="dummy_graph",
            chunk_size=5,
            log_level=logging.DEBUG,
        )

        # Override AgeFreighter-related methods with AsyncMocks.
        self.exporter.connect = AsyncMock(return_value=self.fake_pool)
        self.exporter.write_csv = AsyncMock(return_value="/tmp/dummy.csv")
        self.exporter.create_label_type = AsyncMock()
        self.exporter.get_first_id = AsyncMock(return_value=1000)
        self.exporter.set_up_graph = AsyncMock()

        # Instead of reading a file, set config_json manually and build vertex configs.
        self.exporter.config_json = copy.deepcopy(dummy_config)
        self.exporter._build_vertex_configs()

        # Set up id_maps for export_edges test.
        self.exporter.id_maps = {"Start": {"s1": 1001}, "End": {"e1": 2001}}

        # For export_edges, override methods that query the DB.
        fake_edge_row = FakeRow({"start_id": "s1", "end_id": "e1", "other": "data"})
        self.exporter._count_rows = AsyncMock(return_value=1)
        self.exporter._fetch_edges_chunk_table = AsyncMock(return_value=[fake_edge_row])
        # For export_nodes, override node fetching methods.
        fake_node_row = FakeRow(
            {
                "start_id": "s1",
                "other": "data",
                "_elementid": "s1",
                "properties": {"start_id": "s1", "other": "data"},
            }
        )
        self.exporter._fetch_nodes_chunk_table = AsyncMock(return_value=[fake_node_row])
        self.exporter._fetch_nodes_by_ids_chunk_table = AsyncMock(
            return_value=[fake_node_row]
        )
        # Also, ensure the config contains candidate keys.
        self.exporter.config_json["edge"]["start_id"] = "start_id"
        self.exporter.config_json["edge"]["end_id"] = "end_id"

    async def test_aenter_aexit(self):
        # Test the async context manager methods.
        with patch.object(self.exporter, "_build_vertex_configs") as mock_build:
            async with self.exporter as exp:
                self.assertIs(exp, self.exporter)
            mock_build.assert_called()

    def test_clean_row(self):
        # Test the _clean_row static method.
        class DummyRow:
            def __init__(self, data):
                self.data = data

            def _asdict(self):
                return self.data

        row = DummyRow({'"key"': '"value"', "num": 123})
        cleaned = PGSQLExporter._clean_row(row)
        self.assertEqual(cleaned, {"key": "value", "num": 123})

    def test_get_labels_dict(self):
        # For a dict config with start_vertex and end_vertex.
        labels = self.exporter.get_labels()
        self.assertCountEqual(labels, ["Start", "End"])

    def test_get_ids_dict(self):
        ids = self.exporter.get_ids()
        self.assertEqual(ids, {"start_id": ["start_id"], "end_id": ["end_id"]})

    def test_get_relationship_types(self):
        types = self.exporter.get_relationship_types()
        self.assertEqual(types, ["REL_TYPE"])

    def test_get_vertex_labels_dict(self):
        labels = self.exporter.get_vertex_labels("REL_TYPE")
        self.assertEqual(labels, ("Start", "End"))

    async def test_count_rows(self):
        # Create a fresh exporter instance that uses a pool with a cursor returning (10,).
        exporter = PGSQLExporter(
            dsn="dummy",
            min_connections=1,
            max_connections=1,
            src_dsn="dummy",
            config="dummy",
            trial=False,
            save_temps=False,
            progress=False,
            graph_name="dummy",
            chunk_size=5,
            log_level=logging.DEBUG,
        )
        exporter.src_con_pool = FakePool(FakeCursor(fetchone_result=(10,)))
        count = await exporter._count_rows("dummy_table")
        self.assertEqual(count, 10)

    async def test_fetch_edges_chunk_table(self):
        # Create a fake cursor that returns two rows.
        fake_rows = [FakeRow({"col": "val1"}), FakeRow({"col": "val2"})]
        exporter = PGSQLExporter(
            dsn="dummy",
            min_connections=1,
            max_connections=1,
            src_dsn="dummy",
            config="dummy",
            trial=False,
            save_temps=False,
            progress=False,
            graph_name="dummy",
            chunk_size=5,
            log_level=logging.DEBUG,
        )
        exporter.src_con_pool = FakePool(FakeCursor(fetchall_result=fake_rows))
        rows = await exporter._fetch_edges_chunk_table("dummy_table", 0, 5)
        # Each row is cleaned and assigned a temporary _elementid.
        self.assertEqual(len(rows), 2)
        self.assertIn("_elementid", rows[0])

    async def test_fetch_nodes_chunk_table(self):
        # Create a fake cursor that returns one row.
        fake_rows = [FakeRow({"start_id": "s1", "other": "data"})]
        exporter = PGSQLExporter(
            dsn="dummy",
            min_connections=1,
            max_connections=1,
            src_dsn="dummy",
            config="dummy",
            trial=False,
            save_temps=False,
            progress=False,
            graph_name="dummy",
            chunk_size=5,
            log_level=logging.DEBUG,
        )
        exporter.src_con_pool = FakePool(FakeCursor(fetchall_result=fake_rows))
        rows = await exporter._fetch_nodes_chunk_table("dummy_table", 0, 5, "start_id")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_elementid"], "s1")

    async def test_fetch_nodes_by_ids_chunk_table_empty(self):
        # Unpatch to use the actual method
        self.exporter._fetch_nodes_by_ids_chunk_table = (
            PGSQLExporter._fetch_nodes_by_ids_chunk_table.__get__(self.exporter)
        )
        result = await self.exporter._fetch_nodes_by_ids_chunk_table(
            "dummy_table", [], "start_id"
        )
        self.assertEqual(result, [])

    async def test_fetch_nodes_by_ids_chunk_table(self):
        fake_rows = [FakeRow({"start_id": "s1", "other": "data"})]
        exporter = PGSQLExporter(
            dsn="dummy",
            min_connections=1,
            max_connections=1,
            src_dsn="dummy",
            config="dummy",
            trial=False,
            save_temps=False,
            progress=False,
            graph_name="dummy",
            chunk_size=5,
            log_level=logging.DEBUG,
        )
        exporter.src_con_pool = FakePool(FakeCursor(fetchall_result=fake_rows))
        rows = await exporter._fetch_nodes_by_ids_chunk_table(
            "dummy_table", ["dummy"], "start_id"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_elementid"], "s1")

    def test_get_edge_table_name(self):
        table_name = self.exporter.get_edge_table_name("REL_TYPE")
        self.assertEqual(table_name, "edge_table")
        with self.assertRaises(ValueError) as ctx:
            self.exporter.get_edge_table_name("NON_EXISTENT")
        self.assertIn("No table found", str(ctx.exception))

    async def test_export_nodes_non_trial(self):
        # trial is False so export_nodes uses _fetch_nodes_chunk_table.
        self.exporter.trial = False
        nodes_args = await self.exporter.export_nodes()
        # Check that the csv path was set (backslashes replaced).
        self.assertIn("Start", nodes_args)
        self.assertEqual(nodes_args["Start"]["csv_path"], "/tmp/dummy.csv")
        # Check that id_maps was populated.
        self.assertIn("s1", self.exporter.id_maps["Start"])

    async def test_export_nodes_trial(self):
        # When trial is True, export_nodes uses _fetch_nodes_by_ids_chunk_table.
        self.exporter.trial = True
        self.exporter.trial_nodes_by_label = {"REL_TYPE": {"Start": ["s1"]}}
        nodes_args = await self.exporter.export_nodes()
        self.assertIn("Start", nodes_args)
        self.assertEqual(nodes_args["Start"]["csv_path"], "/tmp/dummy.csv")

    async def test_export_edges(self):
        edges_args = await self.exporter.export_edges()
        self.assertIn("REL_TYPE", edges_args)
        self.assertEqual(edges_args["REL_TYPE"]["csv_path"], "/tmp/dummy.csv")

    async def test_export(self):
        # Test the main export method.
        self.exporter.trial = True
        self.exporter.list_nodes = AsyncMock()
        nodes_args = {"dummy": {}}
        edges_args = {"dummy": {}}
        self.exporter.export_nodes = AsyncMock(return_value=nodes_args)
        self.exporter.export_edges = AsyncMock(return_value=edges_args)
        await self.exporter.export()
        self.assertEqual(self.exporter.vertices, nodes_args)
        self.assertEqual(self.exporter.edges, edges_args)

    async def test_export_edges_with_warning(self):
        # Test export_edges branch where start_id cannot be resolved.
        self.exporter.id_maps = {"Start": {}}
        # Make the fetch method return a row that cannot resolve candidate keys.
        fake_row = FakeRow(
            {"start_id": "unknown", "end_id": "unknown", "other": "data"}
        )
        self.exporter._fetch_edges_chunk_table = AsyncMock(return_value=[fake_row])
        edges_args = await self.exporter.export_edges()
        # Even if resolution fails (warnings logged), write_csv is still called.
        self.assertIn("REL_TYPE", edges_args)
        self.exporter.write_csv.assert_called()


if __name__ == "__main__":
    unittest.main()
