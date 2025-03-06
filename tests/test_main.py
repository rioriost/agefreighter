#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

# Ensure the src directory is in sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import agefreighter.main as main_module  # Import the module under test


# --- Dummy Exporter Classes for Async Context Management ---
class DummyExporter:
    """A dummy exporter that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def export(self):
        pass

    async def copy(self):
        pass


class DummyFailExporter:
    """A dummy exporter that fails during export()."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def export(self):
        raise Exception("export failed")

    async def copy(self):
        pass


# --- Helper context manager to capture printed output ---
@contextmanager
def captured_stdout():
    import io

    new_out = io.StringIO()
    old = sys.stdout
    sys.stdout = new_out
    try:
        yield sys.stdout
    finally:
        sys.stdout = old


# --- Tests for parse_arguments function ---
class TestParseArguments(unittest.TestCase):
    def test_parse_arguments_defaults(self):
        # The main parser options must come before the subcommand.
        test_args = ["main.py", "--pg-con-str", "dummy_connection", "load"]
        with mock.patch.object(sys, "argv", test_args):
            args = main_module.parse_arguments()
            self.assertEqual(args.graphname, "FROM_NEO4J")
            self.assertEqual(args.pg_con_str, "dummy_connection")
            self.assertEqual(args.pg_min_connections, 4)
            self.assertEqual(args.pg_max_connections, 64)
            self.assertFalse(args.debug)
            self.assertEqual(args.subparser, "load")


# --- Async tests for async_main ---
class TestAsyncMain(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.orig_try_import = getattr(main_module, "try_import", None)

    async def asyncTearDown(self):
        if self.orig_try_import is not None:
            main_module.try_import = self.orig_try_import

    async def test_async_main_invalid_connections(self):
        # Test: pg_min_connections > pg_max_connections.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "--pg-min-connections",
            "10",
            "--pg-max-connections",
            "5",
            "load",
            "--neo4j-uri",
            "dummy",
            "--neo4j-user",
            "dummy",
            "--neo4j-password",
            "dummy",
        ]
        with mock.patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                await main_module.async_main()
            self.assertEqual(cm.exception.code, 1)

    async def test_async_main_missing_pg_con_str(self):
        # Test: PG_CONNECTION_STRING missing.
        test_args = [
            "main.py",
            "--pg-con-str",
            "",
            "load",
            "--neo4j-uri",
            "dummy",
            "--neo4j-user",
            "dummy",
            "--neo4j-password",
            "dummy",
        ]
        with mock.patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                await main_module.async_main()
            self.assertEqual(cm.exception.code, 1)

    async def test_async_main_neo4j_missing_credentials(self):
        # Test: neo4j branch with missing credentials.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "load",
            "--neo4j-uri",
            "",
            "--neo4j-user",
            "",
            "--neo4j-password",
            "",
        ]
        with mock.patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                await main_module.async_main()
            self.assertEqual(cm.exception.code, 1)

    async def test_async_main_neo4j_success(self):
        # Test: neo4j branch with valid credentials.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "load",
            "--neo4j-uri",
            "uri",
            "--neo4j-user",
            "user",
            "--neo4j-password",
            "pass",
        ]
        # Patch the Neo4jExporter to use our dummy exporter.
        with mock.patch("agefreighter.neo4jexporter.Neo4jExporter", DummyExporter):
            with mock.patch.object(sys, "argv", test_args):
                await main_module.async_main()

    async def test_async_main_neo4j_exporter_failure(self):
        # Test: neo4j branch where exporter.export() raises an exception.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "load",
            "--neo4j-uri",
            "uri",
            "--neo4j-user",
            "user",
            "--neo4j-password",
            "pass",
        ]
        with mock.patch("agefreighter.neo4jexporter.Neo4jExporter", DummyFailExporter):
            with mock.patch.object(sys, "argv", test_args):
                with mock.patch("agefreighter.main.sys.exit") as mock_exit:
                    await main_module.async_main()
                    mock_exit.assert_called_once_with(1)

    async def test_async_main_csv_missing_config(self):
        # Test: csv branch with missing config argument.
        test_args = ["main.py", "--pg-con-str", "dummy", "load", "--source-type", "csv"]
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("agefreighter.main.sys.exit") as mock_exit:
                await main_module.async_main()
                self.assertGreaterEqual(mock_exit.call_count, 1)

    async def test_async_main_csv_config_not_exist(self):
        # Test: csv branch with a config file that does not exist.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "load",
            "--source-type",
            "csv",
            "--config",
            "nonexistent.cfg",
        ]
        with mock.patch("os.path.exists", return_value=False):
            with mock.patch.object(sys, "argv", test_args):
                with mock.patch("agefreighter.main.sys.exit") as mock_exit:
                    await main_module.async_main()
                    self.assertGreaterEqual(mock_exit.call_count, 1)

    async def test_async_main_csv_success(self):
        # Test: valid csv branch.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            config_path = tmp.name
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "load",
            "--source-type",
            "csv",
            "--config",
            config_path,
        ]
        with mock.patch("agefreighter.csvexporter.CSVExporter", DummyExporter):
            with mock.patch.object(sys, "argv", test_args):
                await main_module.async_main()
        os.remove(config_path)

    async def test_async_main_cosmosdb_missing(self):
        # Test: cosmosdb branch with missing required parameters.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "load",
            "--source-type",
            "cosmosdb",
        ]
        with mock.patch.dict(
            os.environ,
            {
                "COSMOS_ENDPOINT": "",
                "COSMOS_KEY": "",
                "COSMOS_DATABASE": "",
                "COSMOS_CONTAINER": "",
            },
            clear=True,
        ):
            with mock.patch.object(sys, "argv", test_args):
                with self.assertRaises(SystemExit) as cm:
                    await main_module.async_main()
                self.assertEqual(cm.exception.code, 1)

    async def test_async_main_cosmosdb_success(self):
        # Test: valid cosmosdb branch.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "load",
            "--source-type",
            "cosmosdb",
            "--cosmos-endpoint",
            "ep",
            "--cosmos-key",
            "key",
            "--cosmos-database",
            "db",
            "--cosmos-container",
            "cont",
        ]
        with mock.patch(
            "agefreighter.cosmosnosqlexporter.CosmosNoSQLExporter", DummyExporter
        ):
            with mock.patch.object(sys, "argv", test_args):
                await main_module.async_main()

    async def test_async_main_load_invalid_source(self):
        # Test: load branch with a source type not implemented (e.g. "pgsql").
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "load",
            "--source-type",
            "pgsql",
        ]
        with mock.patch.object(sys, "argv", test_args):
            with mock.patch("agefreighter.main.sys.exit") as mock_exit:
                await main_module.async_main()
                mock_exit.assert_called_once()

    async def test_async_main_no_subcommand(self):
        # Test: no subcommand provided. ArgumentParser should error.
        test_args = ["main.py", "--pg-con-str", "dummy"]
        with mock.patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                await main_module.async_main()


# --- Tests for main() function ---
class TestMainFunction(unittest.TestCase):
    def test_main_runs_without_error(self):
        # Patch async_main to avoid running actual async tasks.
        with mock.patch("agefreighter.main.async_main", return_value=asyncio.sleep(0)):
            with mock.patch("asyncio.run") as mock_run:
                main_module.main()
                mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
