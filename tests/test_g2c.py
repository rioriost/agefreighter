#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import json
import os
import sys
import unittest
from contextlib import contextmanager
from unittest import mock

# Adjust the import path so that the module under test is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agefreighter.g2c import (
    QueryExtractor,
    CacheManager,
    GremlinToCypherConverter,
    DryRunner,
    GremlinConverterController,
)

# ---------- Dummy Classes for Testing ----------


class DummyChat:
    def __init__(self, content):
        self.content = content
        # Make 'completions' available and reference self so that
        # self.client.chat.completions.create(...) works.
        self.completions = self

    def create(self, messages, model, temperature):
        # Always return a fixed Cypher query for testing.
        dummy_choice = mock.Mock(message=mock.Mock(content="MATCH (n) RETURN n"))
        dummy = mock.Mock(choices=[dummy_choice])
        return dummy


class DummyOpenAI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.chat = DummyChat(content="dummy")


class DummyConnection:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def cursor(self):
        return DummyCursor(should_fail=self.should_fail)


class DummyCursor:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.queries_executed = []
        self._fetched = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def execute(self, query):
        self.queries_executed.append(query)
        # Do not fail for the check query.
        if "SELECT * FROM ag_graph" in query:
            self._fetched = False
        elif self.should_fail:
            raise RuntimeError("Execution failed")

    def fetchone(self):
        return None  # Simulate that the graph does not exist.


class DummyConnectionPool:
    def __init__(self, dsn):
        self.dsn = dsn

    def open(self):
        pass

    def connection(self):
        return DummyConnection()

    def close(self):
        pass


@contextmanager
def captured_stdout():
    new_out = io.StringIO()
    old = sys.stdout
    sys.stdout = new_out
    try:
        yield sys.stdout
    finally:
        sys.stdout = old


# ---------- Test Cases ----------


class TestQueryExtractor(unittest.TestCase):
    def test_extract_python_literal(self):
        code = 'x = "g.foo.query"'
        extractor = QueryExtractor(code, "dummy.py")
        results = extractor.extract()
        self.assertTrue(any("g.foo.query" in lit for _, lit in results))

    def test_extract_gremlin_queries_ast(self):
        code = """
from gremlin_python.driver import client
def test():
    return g.V().has("name", "john")
"""
        extractor = QueryExtractor(code, "dummy.py")
        results = extractor.extract()
        self.assertTrue(len(results) > 0)
        line, query = results[0]
        self.assertTrue(line >= 1)
        self.assertIn("g.V()", query)

    def test_extract_generic_literals(self):
        code = """
// some comment "g.ignore"
var query = "g.someQuery";
"""
        extractor = QueryExtractor(code, "dummy.java")
        results = extractor.extract()
        self.assertEqual(len(results), 1)
        lineno, literal = results[0]
        self.assertIn("g.someQuery", literal)

    def test_tokenize_error(self):
        bad_code = "def foo(:"
        extractor = QueryExtractor(bad_code, "dummy.py")
        with self.assertRaises(SystemExit) as cm:
            extractor._extract_gremlin_from_literals()
        self.assertIn("Error while parsing Python file", str(cm.exception))

    def test_ast_parse_error(self):
        bad_code = "def foo(:"
        extractor = QueryExtractor(bad_code, "dummy.py")
        with self.assertRaises(SystemExit) as cm:
            extractor._extract_gremlin_queries()
        self.assertIn("Error while parsing Python file", str(cm.exception))


class TestCacheManager(unittest.TestCase):
    def setUp(self):
        self.cache_file = os.path.join(os.path.expanduser("~"), ".g2c_cache")
        patcher = mock.patch("agefreighter.g2c.os.path.exists", return_value=True)
        self.addCleanup(patcher.stop)
        self.mock_exists = patcher.start()

        self.initial_cache = {"foo": "bar"}
        self.mock_open = mock.mock_open(read_data=json.dumps(self.initial_cache))
        patcher2 = mock.patch("agefreighter.g2c.open", self.mock_open, create=True)
        self.addCleanup(patcher2.stop)
        patcher2.start()

    def test_load_cache(self):
        cm = CacheManager()
        self.assertEqual(cm.cache, self.initial_cache)

    def test_add_get_search_result(self):
        cm = CacheManager()
        cm.add_search_result("query1", "cypher1")
        result = cm.get_search_result("query1")
        self.assertEqual(result, "cypher1")


class TestGremlinToCypherConverter(unittest.TestCase):
    def setUp(self):
        self.api_key = "dummy_key"
        self.converter = GremlinToCypherConverter(api_key=self.api_key)
        # Replace the real OpenAI client with our dummy.
        self.converter.client = DummyOpenAI(api_key=self.api_key)

    def test_convert_success(self):
        result = self.converter.convert("g.V()")
        self.assertEqual(result, "MATCH (n) RETURN n")

    def test_convert_failure(self):
        # Patch the correct method: completions.create
        with mock.patch.object(
            self.converter.client.chat.completions,
            "create",
            side_effect=RuntimeError("dummy failure"),
        ):
            with mock.patch("agefreighter.g2c.print") as mock_print:
                result = self.converter.convert("g.V()")
                self.assertEqual(result, "")
                mock_print.assert_called_with("Error during conversion: dummy failure")


class TestDryRunner(unittest.TestCase):
    def setUp(self):
        os.environ["PG_CONNECTION_STRING"] = "postgresql://dummy"
        patcher = mock.patch("agefreighter.g2c.ConnectionPool", DummyConnectionPool)
        self.addCleanup(patcher.stop)
        self.mock_connection_pool = patcher.start()

    def test_run_success(self):
        runner = DryRunner(dsn="dummy_dsn", graph_name="dummy_graph")
        msg = runner.run("MATCH (n) RETURN n")
        self.assertIn("[Query executed successfully]", msg)

    def test_run_failure(self):
        pool = DummyConnectionPool("dummy")
        with mock.patch.object(
            pool, "connection", return_value=DummyConnection(should_fail=True)
        ):
            runner = DryRunner(dsn="dummy_dsn", graph_name="dummy_graph")
            runner.pool = pool
            msg = runner.run("MATCH (n) RETURN n")
            self.assertIn("[Error executing query: ", msg)


class TestGremlinConverterController(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            age=True,
            model="dummy-model",
            dryrun=False,
            gremlin="g.V().has('name','john')",
            filepath=None,
            url=None,
            dsn="dummy_dsn",
            graph_name="dummy_graph",
        )
        self.cache_manager_patch = mock.patch("agefreighter.g2c.CacheManager")
        self.converter_patch = mock.patch("agefreighter.g2c.GremlinToCypherConverter")
        self.mock_cache_manager_class = self.cache_manager_patch.start()
        self.mock_converter_class = self.converter_patch.start()

        # Ensure get_search_result returns None to trigger conversion.
        self.mock_cache_manager_class.return_value.get_search_result.return_value = None

        self.mock_converter_instance = mock.Mock()
        self.mock_converter_instance.convert.return_value = "MATCH (n) RETURN n"
        self.mock_converter_class.return_value = self.mock_converter_instance

        self.dryrunner_patch = mock.patch("agefreighter.g2c.DryRunner")
        self.mock_dryrunner_class = self.dryrunner_patch.start()
        self.addCleanup(self.cache_manager_patch.stop)
        self.addCleanup(self.converter_patch.stop)
        self.addCleanup(self.dryrunner_patch.stop)

        self.read_code_patch = mock.patch.object(
            GremlinConverterController,
            "_read_code",
            return_value=(self.args.gremlin, "direct query"),
        )
        self.read_code_patch.start()
        self.addCleanup(self.read_code_patch.stop)

    def _make_controller(self, dryrun=False):
        return GremlinConverterController(
            api_key="dummy_key",
            model=self.args.model,
            dryrun=dryrun,
            dsn=self.args.dsn,
            graph_name=self.args.graph_name,
            gremlin_query=self.args.gremlin,
            filepath=self.args.filepath,
            url=self.args.url,
            log_level=0,
        )

    def test_process_direct_query(self):
        controller = self._make_controller(dryrun=False)
        with captured_stdout() as out:
            controller.process()
        output = out.getvalue()
        self.assertIn("Converted Cypher queries:", output)
        self.assertIn("MATCH (n) RETURN n", output)
        self.mock_converter_instance.convert.assert_called_with(self.args.gremlin)

    def test_format_cypher_failed(self):
        controller = self._make_controller()
        res = controller.format_cypher("")
        self.assertIn("[Failed]", res)

    def test_format_cypher_age_without_dryrun(self):
        controller = self._make_controller(dryrun=False)
        cypher = "MATCH (n) RETURN n.name, n.age"
        res = controller.format_cypher(cypher)
        self.assertIn("cypher('", res)

    def test_format_cypher_age_with_dryrun(self):
        controller = self._make_controller(dryrun=True)
        dummy_runner = mock.Mock()
        dummy_runner.run.return_value = "[Dummy execution]"
        controller.runner = dummy_runner
        cypher = "MATCH (n) RETURN n"
        res = controller.format_cypher(cypher)
        self.assertIn("[Dummy execution]", res)

    def test_format_for_age_and_extract_return_values(self):
        controller = self._make_controller()
        cypher_query = "MATCH (n) RETURN n.age, n.name"
        ret_vals = controller.extract_return_values(cypher_query)
        self.assertTrue(len(ret_vals) > 0)
        formatted = controller.format_for_age(cypher_query)
        self.assertIn("cypher('", formatted)


if __name__ == "__main__":
    unittest.main()
