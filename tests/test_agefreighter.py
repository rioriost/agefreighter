#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
import unittest
import logging
import sys

import numpy as np
import aiofiles
from psycopg_pool import PoolTimeout
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the class and constants to be tested
from agefreighter.agefreighter import AgeFreighter


# --- Dummy Classes for Async Simulation ---
class DummyRow:
    def __init__(self, count):
        self.count = count


class DummyCopy:
    def __init__(self, query):
        self.query = query
        self.data = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def write(self, data):
        self.data += data


class DummyCursor:
    def __init__(self):
        self.executed_queries = []
        self.fake_row = None  # Allow tests to control the return value.

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def execute(self, query, params=None):
        self.executed_queries.append((query, params))

    async def fetchone(self):
        # Simply return self.fake_row, so if it's None, we simulate no row returned.
        return self.fake_row

    # Make copy a normal method returning an async context manager:
    def copy(self, query):
        return DummyCopy(query)


class DummyConnection:
    def __init__(self, cursor_obj=None):
        self.cursor_obj = cursor_obj if cursor_obj is not None else DummyCursor()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def cursor(self, row_factory=None):
        return self.cursor_obj


class DummyPool:
    def __init__(self):
        # Create a cursor instance that we can inspect.
        self.cursor_obj = DummyCursor()
        self.closed = False

    def connection(self):
        # Return a DummyConnection that uses the shared cursor_obj.
        return DummyConnection(cursor_obj=self.cursor_obj)

    async def close(self):
        self.closed = True


class DummyPoolForConnect:
    """
    Dummy pool for testing the connect() retry logic.
    If always_fail is True the open() method always raises.
    Otherwise, it raises PoolTimeout for the first `fail_attempts` calls.
    """

    def __init__(self, fail_attempts=2, always_fail=False):
        self.attempt = 0
        self.fail_attempts = fail_attempts
        self.always_fail = always_fail

    async def open(self):
        self.attempt += 1
        if self.always_fail or self.attempt <= self.fail_attempts:
            raise PoolTimeout("dummy timeout")

    async def wait(self):
        pass

    async def close(self):
        pass

    async def connection(self):
        return DummyConnection()


# --- Test Suite ---


class TestAgeFreighter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Silence logging during tests.
        logging.disable(logging.CRITICAL)
        self.temp_dirs = []

    def tearDown(self):
        # Clean up any temporary directories created.
        for d in self.temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        logging.disable(logging.NOTSET)

    # Test __init__
    def test_init_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            AgeFreighter(chunk_size=0)

    def test_init_save_temps_creates_output_dir(self):
        instance = AgeFreighter(save_temps=True)
        self.assertTrue(os.path.isdir(instance.output_dir))
        shutil.rmtree(instance.output_dir)

    # Test __del__
    def test_del_removes_temp_files(self):
        # Create a temporary file and assign it to vertices.
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"dummy")
        temp_file.close()
        instance = AgeFreighter(save_temps=False)
        instance.vertices = {"test": {"csv_path": temp_file.name}}
        self.assertTrue(os.path.exists(temp_file.name))
        instance.__del__()
        self.assertFalse(os.path.exists(temp_file.name))

    # Test static methods
    def test_extract_unique_keys(self):
        data = [
            {"properties": {"a": 1, "b": 2}},
            {"properties": {"b": 3, "c": 4}},
            {"properties": {"d": 5}},
        ]
        result = AgeFreighter.extract_unique_keys(data)
        self.assertEqual(result, {"a", "b", "c", "d"})

    def test_quoted_graph_name(self):
        self.assertEqual(AgeFreighter.quoted_graph_name("graph"), "graph")
        self.assertEqual(AgeFreighter.quoted_graph_name("Graph"), '"Graph"')

    # Test set_up_graph (both branches)
    async def test_set_up_graph_existing_no_create(self):
        instance = AgeFreighter(dsn="dummy")
        dummy_cursor = DummyCursor()
        dummy_cursor.fake_row = DummyRow(1)
        dummy_pool = DummyPool()
        dummy_pool.cursor_obj = dummy_cursor
        instance.con_pool = dummy_pool
        await instance.set_up_graph("testgraph", create_graph=False)
        self.assertEqual(instance.graph_name, "testgraph")

    async def test_set_up_graph_existing_with_create(self):
        instance = AgeFreighter(dsn="dummy")
        dummy_cursor = DummyCursor()
        dummy_cursor.fake_row = DummyRow(1)
        dummy_pool = DummyPool()
        dummy_pool.cursor_obj = dummy_cursor
        instance.con_pool = dummy_pool
        await instance.set_up_graph("testgraph", create_graph=True)
        self.assertEqual(instance.graph_name, "testgraph")

    async def test_set_up_graph_not_existing(self):
        instance = AgeFreighter(dsn="dummy")
        dummy_cursor = DummyCursor()
        dummy_cursor.fake_row = DummyRow(0)  # Simulate graph not existing.
        dummy_pool = DummyPool()
        dummy_pool.cursor_obj = dummy_cursor
        instance.con_pool = dummy_pool
        with self.assertRaises(ValueError):
            await instance.set_up_graph("testgraph", create_graph=False)

    # Test create_label_type for vertex, edge, and invalid type.
    async def test_create_label_type_vertex(self):
        instance = AgeFreighter(dsn="dummy")
        dummy_pool = DummyPool()
        instance.con_pool = dummy_pool
        await instance.create_label_type("vertex", "vlabel")
        self.assertTrue(len(dummy_pool.cursor_obj.executed_queries) > 0)

    async def test_create_label_type_edge(self):
        instance = AgeFreighter(dsn="dummy")
        dummy_pool = DummyPool()
        instance.con_pool = dummy_pool
        await instance.create_label_type("edge", "elabel")
        self.assertTrue(len(dummy_pool.cursor_obj.executed_queries) > 0)

    async def test_create_label_type_invalid(self):
        instance = AgeFreighter(dsn="dummy")
        dummy_pool = DummyPool()
        instance.con_pool = dummy_pool
        with self.assertRaises(ValueError):
            await instance.create_label_type("invalid", "label")

    # Test get_first_id
    async def test_get_first_id_success(self):
        instance = AgeFreighter(dsn="dummy")
        dummy_cursor = DummyCursor()
        dummy_cursor.fake_row = (10,)
        dummy_pool = DummyPool()
        dummy_pool.cursor_obj = dummy_cursor
        instance.con_pool = dummy_pool
        result = await instance.get_first_id("testgraph", "label")
        expected = (np.uint64(10) << (32 + 16)) | (
            np.uint64(1) & np.uint64(0x0000FFFFFFFFFFFF)
        )
        self.assertEqual(result, int(expected))

    async def test_get_first_id_no_row(self):
        instance = AgeFreighter(dsn="dummy")
        dummy_cursor = DummyCursor()
        dummy_cursor.fake_row = None
        dummy_pool = DummyPool()
        dummy_pool.cursor_obj = dummy_cursor
        instance.con_pool = dummy_pool
        with self.assertRaises(ValueError):
            await instance.get_first_id("testgraph", "label")

    # Test write_csv
    async def test_write_csv_empty_data(self):
        instance = AgeFreighter()
        result = await instance.write_csv("test", "v", [])
        self.assertEqual(result, "")

    async def test_write_csv_vertex(self):
        # Test writing CSV for vertices.
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = AgeFreighter(output_dir=tmpdir)
            data = [{"id": 1, "properties": {"a": "val"}}]
            file_path = await instance.write_csv("TestLabel", "v", data)
            self.assertTrue(os.path.exists(file_path))
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            self.assertIn("1", content)
            self.assertIn('""a"": ""val""', content)

    async def test_write_csv_edge(self):
        # Test writing CSV for edges.
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = AgeFreighter(output_dir=tmpdir)
            data = [
                {"id": 2, "start_id": 10, "end_id": 20, "properties": {"b": "edge_val"}}
            ]
            file_path = await instance.write_csv("TestEdge", "e", data)
            self.assertTrue(os.path.exists(file_path))
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            self.assertIn("2,10,20", content)
            self.assertIn('""b"": ""edge_val""', content)

    async def test_write_csv_invalid_kind(self):
        instance = AgeFreighter()
        data = [{"id": 1, "properties": {"a": "val"}}]
        with self.assertRaises(ValueError):
            await instance.write_csv("TestLabel", "invalid", data)

    # Test copy() and _copy()
    async def test_copy_missing_parameters(self):
        instance = AgeFreighter()
        with self.assertRaises(ValueError):
            await instance.copy()
        instance.graph_name = "graph"
        with self.assertRaises(ValueError):
            await instance.copy()
        # Now supply vertices with the updated key "next_val" instead of "original_id"
        instance.vertices = {"v": {"csv_path": "dummy", "next_val": 1}}
        with self.assertRaises(ValueError):
            await instance.copy()

    async def test_copy_success(self):
        # Create a temporary CSV file with sample content.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write("col1,col2\n1,2\n")
            tmp_name = tmp.name
        try:
            instance = AgeFreighter(dsn="dummy", graph_name="graph")
            # Use the new key "next_val" in both vertices and edges
            instance.vertices = {"v": {"csv_path": tmp_name, "next_val": 1}}
            instance.edges = {"e": {"csv_path": tmp_name, "next_val": 1}}
            # Use our updated DummyPool.
            instance.con_pool = DummyPool()
            await instance.copy()
            # Now check that some queries were executed.
            self.assertTrue(len(instance.con_pool.cursor_obj.executed_queries) > 0)
        finally:
            os.remove(tmp_name)

    async def test__copy_invalid_kind(self):
        instance = AgeFreighter(dsn="dummy", graph_name="graph")
        dummy_pool = DummyPool()
        instance.con_pool = dummy_pool
        # Pass a valid numeric next_val instead of a string
        with self.assertRaises(ValueError):
            await instance._copy("graph", "dummy.csv", "id", 1, kind="invalid")

    async def test__copy_file_not_found(self):
        instance = AgeFreighter(dsn="dummy", graph_name="graph")
        dummy_pool = DummyPool()
        instance.con_pool = dummy_pool
        # Use a valid numeric next_val and non-existent file to trigger an exception.
        with self.assertRaises(Exception):
            await instance._copy("graph", "non_existent_file.csv", "id", 1, kind="v")

    # Test connect() method
    async def test_connect_success(self):
        instance = AgeFreighter(dsn="dummy")
        # Patch AsyncConnectionPool so that open() fails twice then succeeds.
        with patch(
            "agefreighter.agefreighter.AsyncConnectionPool",
            return_value=DummyPoolForConnect(fail_attempts=2),
        ) as mock_pool:
            await instance.connect(max_connections=10, min_connections=2)
            self.assertIsNotNone(instance.con_pool)

    async def test_connect_failure(self):
        instance = AgeFreighter(dsn="dummy")
        # Patch AsyncConnectionPool so that open() always fails.
        with patch(
            "agefreighter.agefreighter.AsyncConnectionPool",
            return_value=DummyPoolForConnect(always_fail=True),
        ) as mock_pool:
            with self.assertRaises(PoolTimeout):
                await instance.connect(max_connections=10, min_connections=2)

    # Test close() method
    async def test_close(self):
        instance = AgeFreighter(dsn="dummy")
        dummy_pool = DummyPool()
        instance.con_pool = dummy_pool
        await instance.close()
        self.assertTrue(dummy_pool.closed)

    # Test async context manager (__aenter__ and __aexit__)
    async def test_async_context_manager(self):
        instance = AgeFreighter(dsn="dummy")
        instance.connect = AsyncMock()
        # Patch the connection pool's close method instead of instance.close
        instance.con_pool = AsyncMock()
        async with instance as af:
            instance.connect.assert_called_once()
        instance.con_pool.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
