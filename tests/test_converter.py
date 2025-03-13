#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock


# Adjust the import path so that the module under test is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agefreighter import converter


class DummyCompletions:
    def create(self, messages, model, temperature):
        # Create a dummy completion that returns a fixed Cypher query.
        dummy_message = MagicMock()
        dummy_message.content = "MATCH (n) RETURN n"
        dummy_choice = MagicMock()
        dummy_choice.message = dummy_message
        dummy_completion = MagicMock()
        dummy_completion.choices = [dummy_choice]
        return dummy_completion


class DummyClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = DummyCompletions()


class TestQueryExtractor(unittest.TestCase):
    def test_extract_gremlin_literal(self):
        # A Python code snippet containing a Gremlin literal.
        code = "def f():\n    query = \"g.V().hasLabel('person')\""
        extractor = converter.QueryExtractor(code, "dummy.py")
        results = extractor.extract()
        # Expect one extracted query that starts with "g.V()..."
        self.assertTrue(any("g.V().hasLabel('person')" in q for _, q in results))

    def test_extract_cypher_literal(self):
        # A Python code snippet containing a Cypher literal.
        code = 'def f():\n    query = "MATCH (n) RETURN n"'
        extractor = converter.QueryExtractor(code, "dummy.py")
        results = extractor.extract()
        # Expect one extracted query that starts with "MATCH"
        self.assertTrue(any("MATCH (n) RETURN n" in q for _, q in results))

    def test_extract_generic_cypher_literal(self):
        # For a file with .cypher extension, the extractor should use the generic extractor.
        code = 'Some comment\n"MATCH (n) RETURN n"'
        extractor = converter.QueryExtractor(code, "dummy.cypher")
        results = extractor.extract()
        # The extraction uses regex so the literal should be present.
        self.assertTrue(any("MATCH (n) RETURN n" in q for _, q in results))


class TestGremlinToCypherConverter(unittest.TestCase):
    def test_convert_returns_expected_cypher(self):
        # Create a converter and override its client with our dummy client.
        conv = converter.GremlinToCypherConverter(api_key="dummy_key")
        conv.client = DummyClient()
        gremlin_query = "g.V().hasLabel('person')"
        cypher = conv.convert(gremlin_query)
        # Since our dummy returns a fixed response, we expect:
        self.assertEqual(cypher, "MATCH (n) RETURN n")


class TestCacheManager(unittest.TestCase):
    def setUp(self):
        # Patch _load_cache and _save_cache so no file IO is performed.
        self.patcher_load = patch.object(
            converter.CacheManager, "_load_cache", return_value={}
        )
        self.patcher_save = patch.object(
            converter.CacheManager, "_save_cache", lambda self: None
        )
        self.mock_load = self.patcher_load.start()
        self.mock_save = self.patcher_save.start()
        self.cache_manager = converter.CacheManager()

    def tearDown(self):
        self.patcher_load.stop()
        self.patcher_save.stop()

    def test_add_and_get_search_result(self):
        # Initially, no result should be cached.
        self.assertIsNone(self.cache_manager.get_search_result("test_query"))
        # Add a conversion result.
        self.cache_manager.add_search_result("test_query", "MATCH (n) RETURN n")
        # Now it should be retrievable.
        self.assertEqual(
            self.cache_manager.get_search_result("test_query"), "MATCH (n) RETURN n"
        )


class TestConverterControllerReadCode(unittest.TestCase):
    def test_direct_query_replaces_smart_quotes(self):
        # Test _read_code when a direct query is provided.
        direct_query = "“g.V()”"
        controller = converter.ConverterController(
            query_language="gremlin",
            api_key="dummy",
            model="dummy-model",
            dryrun=False,
            dsn="",
            graph_name="",
            query=direct_query,
            filepath="",
            url="",
            log_level=40,  # Only errors
        )
        code, path = controller._read_code()
        # Smart quotes should have been replaced by standard double quotes.
        self.assertNotIn("“", code)
        self.assertEqual(path, "direct query")

    def test_file_reading(self):
        # Create a temporary file with known content.
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp_file:
            tmp_file.write("print('Hello World')")
            tmp_file_path = tmp_file.name

        try:
            controller = converter.ConverterController(
                query_language="gremlin",
                api_key="dummy",
                model="dummy-model",
                dryrun=False,
                dsn="",
                graph_name="",
                query="",
                filepath=tmp_file_path,
                url="",
                log_level=40,
            )
            code, path = controller._read_code()
            self.assertEqual(code, "print('Hello World')")
            self.assertEqual(path, tmp_file_path)
        finally:
            os.remove(tmp_file_path)


class TestConverterControllerFormatting(unittest.TestCase):
    def setUp(self):
        # Create a dummy ConverterController with a known graph name.
        self.controller = converter.ConverterController(
            query_language="gremlin",
            api_key="dummy",
            model="dummy-model",
            dryrun=False,
            dsn="",
            graph_name="mygraph",
            query="",
            filepath="",
            url="",
            log_level=40,
        )

    def test_format_cypher_failure(self):
        # When the conversion yields an empty string, format_cypher should return the failure message.
        formatted = self.controller.format_cypher("")
        self.assertEqual(formatted, f"{converter.RED}[Failed]{converter.RESET}")

    @patch("agefreighter.converter.CypherParser")
    def test_get_return_values(self, mock_cypher_parser):
        # Set up the dummy parser to return a known result.
        instance = mock_cypher_parser.return_value
        instance.parse.return_value = [("RETURN", ["col1", "col2"])]
        returns = self.controller.get_return_values("MATCH (n) RETURN n")
        self.assertEqual(returns, ["col1", "col2"])

    @patch("agefreighter.converter.CypherParser")
    def test_format_cypher_no_params(self, mock_cypher_parser):
        # Simulate a Cypher query with no parameters and no RETURN clause.
        instance = mock_cypher_parser.return_value
        instance.parse.return_value = []  # No RETURN clause
        query = "MATCH (n) RETURN n"
        formatted = self.controller.format_cypher(query)
        expected = f"SELECT * FROM cypher('mygraph', $$ {query} $$);"
        self.assertEqual(formatted, expected)

    @patch("agefreighter.converter.CypherParser")
    def test_format_cypher_with_return_and_params(self, mock_cypher_parser):
        # Simulate a Cypher query with parameters (detected by a $param pattern)
        # and a RETURN clause.
        instance = mock_cypher_parser.return_value
        instance.parse.return_value = [("RETURN", ["result"])]
        query = "MATCH (n) WHERE n.id = $param RETURN n"
        formatted = self.controller.format_cypher(query)
        # The output should include a stored procedure section, parameter addition and execution.
        self.assertIn("PREPARE cypher_stored_procedure(agtype) AS", formatted)
        self.assertIn("EXECUTE cypher_stored_procedure", formatted)
        self.assertIn("result agtype", formatted)


if __name__ == "__main__":
    unittest.main()
