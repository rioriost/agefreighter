#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock
import subprocess

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
            self.assertEqual(args.graphname, "FROM_AGEFREIGHTER")
            self.assertEqual(args.pg_con_str, "dummy_connection")
            self.assertEqual(args.pg_min_connections, 4)
            self.assertEqual(args.pg_max_connections, 64)
            self.assertFalse(args.debug)
            self.assertEqual(args.subparser, "load")


# --- Tests for main module utility functions (lines 21-92) ---
class TestMainUtilityFunctions(unittest.TestCase):
    def test_show_completion_instructions(self):
        from io import StringIO

        captured = StringIO()
        sys.stdout = captured
        try:
            main_module.show_completion_instructions()
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("bash (~/.bashrc):", output)
        self.assertIn("zsh", output)

    def test_check_first_run_creates_marker(self):
        # Use a temporary directory as the "home" so we don't touch the real one.
        with tempfile.TemporaryDirectory() as tmp_home:
            with mock.patch("os.path.expanduser", return_value=tmp_home):
                marker_file = os.path.join(tmp_home, ".agefreighter_first_run")
                if os.path.exists(marker_file):
                    os.remove(marker_file)
                with captured_stdout() as out:
                    main_module.check_first_run()
                self.assertTrue(os.path.exists(marker_file))
                with open(marker_file, "r") as f:
                    content = f.read()
                self.assertEqual(content, "agefreighter has been executed.")
                os.remove(marker_file)

    def test_check_and_install_module_exists(self):
        # Simulate module already exists by patching find_spec to return a non-None value.
        with mock.patch("importlib.util.find_spec", return_value=True):
            # Should return normally without prompting.
            main_module.check_and_install("dummy_module")

    def test_check_and_install_denied(self):
        # Simulate module not found and user denying installation.
        with mock.patch("importlib.util.find_spec", return_value=None):
            with mock.patch("builtins.input", return_value="n"):
                with self.assertRaises(SystemExit) as cm:
                    main_module.check_and_install("dummy_module")
                self.assertEqual(cm.exception.code, 1)

    def test_check_and_install_pip_failure(self):
        # Simulate module not found, user agrees, but pip installation fails.
        with mock.patch("importlib.util.find_spec", return_value=None):
            with mock.patch("builtins.input", return_value="y"):
                error = subprocess.CalledProcessError(1, "cmd", stderr="error")
                with mock.patch("subprocess.run", side_effect=error):
                    with self.assertRaises(SystemExit) as cm:
                        main_module.check_and_install("dummy_module")
                    self.assertEqual(cm.exception.code, 1)

    def test_run_flask(self):
        # Create a dummy app and patch it in the agefreighter.view module.
        dummy_app = mock.MagicMock()
        with mock.patch.dict(
            "sys.modules", {"agefreighter.view": mock.MagicMock(app=dummy_app)}
        ):
            main_module.run_flask(8080, log_level=10)
            dummy_app.logger.setLevel.assert_called_once_with(10)
            dummy_app.run.assert_called_once_with(port=8080)


# --- Async tests for async_main and subcommands ---
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
        # Test: load branch with a source type not implemented.
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
                # Expecting two sys.exit calls due to multiple errors in this branch.
                self.assertEqual(mock_exit.call_count, 2)

    async def test_async_main_no_subcommand(self):
        # Test: no subcommand provided. ArgumentParser should error.
        test_args = ["main.py", "--pg-con-str", "dummy"]
        with mock.patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                await main_module.async_main()

    # --- New tests for subcommands in the later sections (lines 552-681) ---

    async def test_async_main_parse_success(self):
        # Test the "parse" subcommand branch.
        test_args = ["main.py", "--pg-con-str", "dummy", "parse", "MATCH (n) RETURN n"]
        dummy_parser = mock.MagicMock()
        dummy_parser.parse.return_value = "parsed_query"
        with mock.patch("agefreighter.main.check_and_install"):
            with mock.patch(
                "agefreighter.cypherparser.CypherParser", return_value=dummy_parser
            ):
                with captured_stdout() as out:
                    with mock.patch.object(sys, "argv", test_args):
                        await main_module.async_main()
                    output = out.getvalue()
                    self.assertIn("parsed_query", output)

    async def test_async_main_generate_success(self):
        # Test the "generate" subcommand branch.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "generate",
            "--pattern-no",
            "2",
            "--multiplier",
            "3",
        ]

        async def dummy_generator_main(pattern_no, multiplier, log_level):
            self.assertEqual(pattern_no, 2)
            self.assertEqual(multiplier, 3)

        with mock.patch("agefreighter.main.check_and_install"):
            with mock.patch("agefreighter.generator.main", new=dummy_generator_main):
                with mock.patch.object(sys, "argv", test_args):
                    await main_module.async_main()

    async def test_async_main_convert_success(self):
        # Test the "convert" subcommand branch.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "convert",
            "--openai-api-key",
            "key",
            "--gremlin",
            "query",
        ]
        dummy_controller = mock.MagicMock()
        dummy_controller.process = mock.MagicMock()
        with mock.patch("agefreighter.main.check_and_install"):
            with mock.patch(
                "agefreighter.g2c.GremlinConverterController",
                return_value=dummy_controller,
            ):
                with mock.patch.object(sys, "argv", test_args):
                    await main_module.async_main()
                    dummy_controller.process.assert_called_once()

    async def test_async_main_prepare_neo4j_failure(self):
        # Test the "prepare" subcommand for neo4j with missing credentials.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "prepare",
            "--target-type",
            "neo4j",
            "--neo4j-uri",
            "uri",
            "--neo4j-user",
            "",
            "--neo4j-password",
            "",
        ]
        with mock.patch("agefreighter.main.check_and_install"):
            with mock.patch.object(sys, "argv", test_args):
                with self.assertRaises(SystemExit) as cm:
                    await main_module.async_main()
                self.assertEqual(cm.exception.code, 1)

    async def test_async_main_prepare_neo4j_success(self):
        # Test the "prepare" subcommand for neo4j with valid credentials.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "prepare",
            "--target-type",
            "neo4j",
            "--neo4j-uri",
            "uri",
            "--neo4j-user",
            "user",
            "--neo4j-password",
            "pass",
        ]
        dummy_loader = mock.MagicMock()
        dummy_loader.load_data = mock.AsyncMock()
        with mock.patch("agefreighter.main.check_and_install"):
            with mock.patch(
                "agefreighter.csvdatamanager.CsvDataManager",
                return_value=mock.MagicMock(),
            ):
                with mock.patch(
                    "agefreighter.neo4jloader.Neo4jLoader", return_value=dummy_loader
                ):
                    with mock.patch.object(sys, "argv", test_args):
                        await main_module.async_main()
                        dummy_loader.load_data.assert_awaited_once()

    async def test_async_main_prepare_pgsql_success(self):
        # Test the "prepare" subcommand for pgsql.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "prepare",
            "--target-type",
            "pgsql",
            "--src-pg-con-str",
            "src_dummy",
        ]
        dummy_loader = mock.MagicMock()
        dummy_loader.load_data = mock.AsyncMock()
        with mock.patch("agefreighter.main.check_and_install"):
            with mock.patch(
                "agefreighter.csvdatamanager.CsvDataManager",
                return_value=mock.MagicMock(),
            ):
                with mock.patch(
                    "agefreighter.pgsqlloader.PgsqlLoader", return_value=dummy_loader
                ):
                    with mock.patch.object(sys, "argv", test_args):
                        await main_module.async_main()
                        dummy_loader.load_data.assert_awaited_once()

    async def test_async_main_prepare_cosmosdb_success(self):
        # Test the "prepare" subcommand for cosmosdb with valid parameters.
        test_args = [
            "main.py",
            "--pg-con-str",
            "dummy",
            "prepare",
            "--target-type",
            "cosmosdb",
            "--cosmos-gremlin-endpoint",
            "endpoint",
            "--cosmos-key",
            "key",
            "--cosmos-database",
            "db",
            "--cosmos-container",
            "container",
        ]
        dummy_loader = mock.MagicMock()
        dummy_loader.load_data = mock.AsyncMock()
        with mock.patch("agefreighter.main.check_and_install"):
            with mock.patch(
                "agefreighter.csvdatamanager.CsvDataManager",
                return_value=mock.MagicMock(),
            ):
                with mock.patch(
                    "agefreighter.cosmosgremlinloader.CosmosGremlinLoader",
                    return_value=dummy_loader,
                ):
                    with mock.patch.object(sys, "argv", test_args):
                        await main_module.async_main()
                        dummy_loader.load_data.assert_awaited_once()


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
