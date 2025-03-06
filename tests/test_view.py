#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import agefreighter.view as view  # This imports view.py as the module "view"


# ----- Fake Classes for Database Simulation ----- #
class FakeCursor:
    """Fake cursor for simulating database responses."""

    def __init__(self):
        self._data = []
        self.last_query = ""

    def execute(self, query, params=None):
        self.last_query = query
        # Simulate responses based on query content
        if "SELECT 1;" in query:
            self._data = [(1,)]
        elif "FROM cypher(" in query:
            # For execute_query test: return one row with one vertex and one edge.
            vertex_json = (
                json.dumps(
                    {"id": "n1", "label": "Node1", "properties": {"key": "value"}}
                )
                + "::vertex"
            )
            edge_json = (
                json.dumps(
                    {
                        "id": "e1",
                        "label": "Edge1",
                        "start_id": "n1",
                        "end_id": "n2",
                        "properties": {},
                    }
                )
                + "::edge"
            )
            self._data = [(vertex_json, edge_json)]
        elif "SELECT graphid, name, namespace FROM ag_graph" in query:
            self._data = [(1, "graph1", "public")]
        elif "FROM ag_label WHERE graph = %s" in query:
            # Simulate label data for graph id 1.
            if params and params[0] == 1:
                self._data = [
                    ("node_label", "v", "table1"),
                    ("edge_label", "e", "table2"),
                ]
            else:
                self._data = []
        elif query.strip().startswith("SELECT COUNT(*) FROM"):
            # Return counts based on table name
            if "table1" in query:
                self._data = [(10,)]
            elif "table2" in query:
                self._data = [(5,)]
            else:
                self._data = [(0,)]
        else:
            self._data = []

    def fetchone(self):
        return self._data[0] if self._data else None

    def fetchall(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FakeConnection:
    """Fake connection that returns a fake cursor."""

    def __init__(self, *args, **kwargs):
        pass

    def cursor(self):
        return FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FakePool:
    """Fake connection pool that returns a fake connection."""

    def __init__(self, *args, **kwargs):
        pass

    def connection(self):
        return FakeConnection()


# ----- Unit Tests for Utility Classes ----- #
class TestConnectionStringParser(unittest.TestCase):
    """Tests for ConnectionStringParser utility class."""

    def test_parse_valid_connection_string(self):
        conn_str = "host=localhost port=5432 dbname=test user=test password=secret"
        expected = {
            "host": "localhost",
            "port": "5432",
            "dbname": "test",
            "user": "test",
            "password": "secret",
        }
        result = view.ConnectionStringParser.parse(conn_str)
        self.assertEqual(result, expected)

    def test_parse_empty_string(self):
        self.assertEqual(view.ConnectionStringParser.parse(""), {})


class TestCypherQueryFormatter(unittest.TestCase):
    """Tests for CypherQueryFormatter utility class."""

    def test_safe_query_with_limit(self):
        query = "MATCH (n) RETURN n"
        formatted = view.CypherQueryFormatter.format_query("graph1", query)
        self.assertIn("LIMIT", formatted)

        self.assertIn("graph1", formatted)

    def test_query_without_limit_appends_limit(self):
        query = "MATCH (n) RETURN n"
        formatted = view.CypherQueryFormatter.format_query("graph1", query)
        # Check that LIMIT 50 was appended
        self.assertTrue("LIMIT 50" in formatted)

    def test_unsafe_query_raises_value_error(self):
        unsafe_query = "MATCH (n) DELETE n RETURN n"
        with self.assertRaises(ValueError) as context:
            view.CypherQueryFormatter.format_query("graph1", unsafe_query)
        self.assertEqual(str(context.exception), "Unsafe query")

    def test_parameterized_query_raises_value_error(self):
        param_query = "MATCH (n {id: $id}) RETURN n"
        with self.assertRaises(ValueError) as context:
            view.CypherQueryFormatter.format_query("graph1", param_query)
        self.assertEqual(str(context.exception), "Parameterized query")

    def test_query_with_no_return_values_raises_value_error(self):
        query = "MATCH (n)"
        with self.assertRaises(ValueError) as context:
            view.CypherQueryFormatter.format_query("graph1", query)
        self.assertEqual(str(context.exception), "No return values specified")

    def test_extract_return_values(self):
        query = "MATCH (n) RETURN n, n.name AS name"
        returns = view.CypherQueryFormatter.extract_return_values(query)
        # Depending on implementation, at least 'n' should be extracted.
        self.assertTrue("n" in returns)


# ----- Unit Tests for DatabaseManager ----- #
class TestDatabaseManager(unittest.TestCase):
    """Tests for the DatabaseManager class."""

    def setUp(self):
        self.db_manager = view.DatabaseManager()
        # Patch ConnectionPool to use FakePool
        self.pool_patcher = patch(
            "agefreighter.view.ConnectionPool", return_value=FakePool()
        )
        self.pool_patcher.start()

    def tearDown(self):
        self.pool_patcher.stop()

    def test_connect(self):
        connection_info = {
            "host": "localhost",
            "port": "5432",
            "dbname": "testdb",
            "user": "testuser",
            "password": "testpass",
        }
        self.db_manager.connect(connection_info)
        self.assertIsNotNone(self.db_manager.pool)
        self.assertEqual(self.db_manager.connection_info, connection_info)

    def test_execute_query(self):
        # Connect first
        connection_info = {
            "host": "localhost",
            "port": "5432",
            "dbname": "testdb",
            "user": "testuser",
            "password": "testpass",
        }
        self.db_manager.connect(connection_info)
        # Use a safe query that returns a vertex and an edge.
        query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        result = self.db_manager.execute_query("graph1", query)
        # Verify that nodes and edges were extracted.
        self.assertIn("nodes", result)
        self.assertIn("edges", result)
        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["nodes"][0]["id"], "n1")
        self.assertEqual(result["edges"][0]["id"], "e1")

    def test_execute_query_without_connection(self):
        self.db_manager.pool = None
        with self.assertRaises(ValueError):
            self.db_manager.execute_query("graph1", "MATCH (n) RETURN n")

    def test_get_graph_info(self):
        connection_info = {
            "host": "localhost",
            "port": "5432",
            "dbname": "testdb",
            "user": "testuser",
            "password": "testpass",
        }
        self.db_manager.connect(connection_info)
        graph_info = self.db_manager.get_graph_info()
        # Expect one graph with graphid 1, one node label and one edge label.
        self.assertEqual(len(graph_info), 1)
        info = graph_info[0]
        self.assertEqual(info["graphid"], 1)
        self.assertEqual(info["graph_name"], "graph1")
        self.assertEqual(len(info["nodes"]), 1)
        self.assertEqual(len(info["edges"]), 1)
        self.assertEqual(info["nodes"][0]["name"], "node_label")
        self.assertEqual(info["nodes"][0]["count"], 10)
        self.assertEqual(info["edges"][0]["name"], "edge_label")
        self.assertEqual(info["edges"][0]["count"], 5)

    def test_get_graph_info_without_connection(self):
        self.db_manager.pool = None
        with self.assertRaises(ValueError):
            self.db_manager.get_graph_info()


# ----- Unit Tests for Flask Endpoints ----- #
class TestFlaskEndpoints(unittest.TestCase):
    """Tests for the Flask API endpoints."""

    def setUp(self):
        view.app.config["TESTING"] = True
        self.client = view.app.test_client()
        # Default connection info for testing endpoints.
        self.test_connection = {
            "host": "localhost",
            "port": "5432",
            "dbname": "testdb",
            "user": "testuser",
            "password": "testpass",
        }
        # Patch ConnectionPool to use FakePool for endpoints.
        self.pool_patcher = patch(
            "agefreighter.view.ConnectionPool", return_value=FakePool()
        )
        self.pool_patcher.start()
        # Connect the db_manager (assumes connection always succeeds with FakePool).
        try:
            view.db_manager.connect(self.test_connection)
        except Exception:
            pass

    def tearDown(self):
        self.pool_patcher.stop()

    def test_index(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_connect_db_success(self):
        response = self.client.post(
            "/api/connect", json={"connection": self.test_connection}
        )
        data = response.get_json()
        self.assertIn("message", data)
        self.assertEqual(response.status_code, 200)

    def test_connect_db_failure(self):
        # Force failure by patching db_manager.connect to raise an exception.
        with patch.object(
            view.db_manager, "connect", side_effect=Exception("Test error")
        ):
            response = self.client.post(
                "/api/connect", json={"connection": self.test_connection}
            )
            data = response.get_json()
            self.assertIn("error", data)
            self.assertEqual(response.status_code, 500)

    def test_connection_status(self):
        response = self.client.get("/api/connection_status")
        data = response.get_json()
        self.assertIn("status", data)
        self.assertIn("connection_info", data)
        self.assertEqual(response.status_code, 200)

    def test_execute_query_success(self):
        query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        response = self.client.post(
            "/api/execute_query", json={"graph_name": "graph1", "cypher_query": query}
        )
        data = response.get_json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertEqual(response.status_code, 200)

    def test_execute_query_no_query(self):
        response = self.client.post(
            "/api/execute_query", json={"graph_name": "graph1", "cypher_query": ""}
        )
        data = response.get_json()
        self.assertIn("error", data)
        self.assertEqual(response.status_code, 400)

    def test_execute_query_failure(self):
        # Simulate failure by patching execute_query to raise a ValueError.
        with patch.object(
            view.db_manager, "execute_query", side_effect=ValueError("Test query error")
        ):
            response = self.client.post(
                "/api/execute_query",
                json={"graph_name": "graph1", "cypher_query": "MATCH (n) RETURN n"},
            )
            data = response.get_json()
            self.assertIn("error", data)
            self.assertEqual(response.status_code, 400)

    def test_graph_info_success(self):
        response = self.client.get("/api/graph_info")
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(response.status_code, 200)

    def test_graph_info_failure(self):
        # Simulate failure by setting pool to None.
        view.db_manager.pool = None
        response = self.client.get("/api/graph_info")
        data = response.get_json()
        self.assertIn("error", data)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
