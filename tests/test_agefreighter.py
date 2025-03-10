#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import aiofiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agefreighter.agefreighter import (
    AgeFreighter,
    reconnect_on_failure,
)
from psycopg_pool import PoolTimeout


class DummyCopy:
    async def write(self, data):
        self.data = data


class DummyCopyContextManager:
    async def __aenter__(self):
        self.copy = DummyCopy()
        return self.copy

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummyCursor:
    async def execute(self, query, params=None):
        self.last_query = query
        # Simulate expected queries based on the string content.
        if "SELECT id FROM ag_label" in str(query):
            self.fake_row = (1234,)
        elif "SELECT count(*) FROM ag_graph" in str(query):
            # Fake a row with count attribute.
            FakeRow = type("FakeRow", (), {"count": 1})
            self.fake_row = FakeRow()
        else:
            self.fake_row = None

    async def fetchone(self):
        return self.fake_row

    # Change: Make copy a normal (non-async) method so that it returns an async context manager.
    def copy(self, query):
        return DummyCopyContextManager()


class DummyCursorContextManager:
    async def __aenter__(self):
        return DummyCursor()

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummyConnection:
    def cursor(self, **kwargs):
        return DummyCursorContextManager()


class DummyConnectionContextManager:
    async def __aenter__(self):
        return DummyConnection()

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummyPool:
    def __init__(self):
        self.opened = False

    async def open(self):
        self.opened = True

    async def wait(self):
        pass

    def connection(self):
        return DummyConnectionContextManager()

    async def close(self):
        self.opened = False


class Dummy:
    def __init__(self):
        self.counter = 0
        self.max_attempts = 3
        self.retry_delay = 0.01  # shorten delay for tests

    @reconnect_on_failure
    async def sometimes_fail(self):
        self.counter += 1
        if self.counter < 2:
            raise Exception("fail")
        return "success"


class TestAgeFreighter(unittest.IsolatedAsyncioTestCase):
    async def test_connect_and_close(self):
        # Patch parse_dsn to return a dummy dict with a "host" key.
        with patch.object(
            AgeFreighter, "parse_dsn", return_value={"host": "dummy_host"}
        ):
            # Patch AsyncConnectionPool to return our DummyPool instance.
            with patch(
                "agefreighter.agefreighter.AsyncConnectionPool",
                return_value=DummyPool(),
            ):
                af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
                # Assign the returned pool to af.con_pool
                af.con_pool = await af.connect(
                    af.dsn_w_option,
                    af.max_connections,
                    af.min_connections,
                    af.max_attempts,
                    af.retry_delay,
                )
                self.assertTrue(af.con_pool.opened)
                await af.close()
                self.assertFalse(af.con_pool.opened)

    async def test_get_first_id(self):
        # Test that get_first_id returns the expected computed value.
        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
        af.con_pool = DummyPool()
        first_id = await af.get_first_id("dummy_graph", "dummy_label")
        # Calculation: (1234 << 48) | (1 & 0x0000FFFFFFFFFFFF)
        expected = (1234 << 48) | 1
        self.assertEqual(first_id, expected)

    async def test_set_up_graph_create_existing(self):
        # Test set_up_graph when the graph exists and create_graph is True.
        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
        af.con_pool = DummyPool()
        af.graph_name = "dummy_graph"
        # This should not raise since our dummy cursor simulates count == 1.
        await af.set_up_graph("dummy_graph", create_graph=True)

    async def test_create_label_type_vertex(self):
        # Test creating a vertex label type.
        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
        af.con_pool = DummyPool()
        await af.create_label_type("vertex", "v_label")

    async def test_create_label_type_edge(self):
        # Test creating an edge label type.
        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
        af.con_pool = DummyPool()
        await af.create_label_type("edge", "e_label")

    async def test_create_label_type_invalid(self):
        # Test that an invalid label type eventually raises an exception.
        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
        af.con_pool = DummyPool()
        with self.assertRaises(Exception) as cm:
            await af.create_label_type("invalid", "label")
        self.assertIn("Unsupported label type", str(cm.exception))

    async def test_copy_method_errors(self):
        # Test that copy() raises ValueError if required attributes are missing.
        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
        with self.assertRaises(ValueError):
            await af.copy()
        af.graph_name = "graph"
        with self.assertRaises(ValueError):
            await af.copy()
        af.vertices = {"v": {"csv_path": "dummy", "next_val": 1}}
        with self.assertRaises(ValueError):
            await af.copy()
        af.edges = {"e": {"csv_path": "dummy", "next_val": 1}}

        # Now override _copy_vertices and _copy_edges so that copy() succeeds.
        called = {"vertices": False, "edges": False}

        async def fake_copy_vertices():
            called["vertices"] = True

        async def fake_copy_edges():
            called["edges"] = True

        af._copy_vertices = fake_copy_vertices
        af._copy_edges = fake_copy_edges
        await af.copy()
        self.assertTrue(called["vertices"])
        self.assertTrue(called["edges"])

    async def test_write_csv_normal_and_tab(self):
        # Create a temporary directory for CSV output.
        with tempfile.TemporaryDirectory() as tmpdir:
            af = AgeFreighter(dsn="dummy_dsn", chunk_size=10, output_dir=tmpdir)
            # Create sample vertex data with one property containing a tab.
            vertex_data = [
                {"id": 1, "properties": {"a": "value", "b": "val\tue"}},
                {"id": 2, "properties": {"a": "value2", "b": "value2"}},
            ]
            # Create sample edge data with required keys for edge CSV.
            edge_data = [
                {
                    "id": 1,
                    "start_id": 10,
                    "end_id": 20,
                    "properties": {"a": "value", "b": "val\tue"},
                },
                {
                    "id": 2,
                    "start_id": 30,
                    "end_id": 40,
                    "properties": {"a": "value2", "b": "value2"},
                },
            ]
            # Test for vertex CSV (kind "v")
            csv_path = await af.write_csv("TestLabel", "v", vertex_data)
            self.assertTrue(os.path.exists(csv_path))
            async with aiofiles.open(csv_path, "r", encoding="utf-8") as f:
                content = await f.read()
                self.assertIn("1,", content)
            # Test for edge CSV (kind "e")
            csv_path_edge = await af.write_csv("TestEdge", "e", edge_data)
            self.assertTrue(os.path.exists(csv_path_edge))

    async def test_reconnect_on_failure_decorator(self):
        # Test that the decorator retries and returns success after failures.
        dummy = Dummy()
        result = await dummy.sometimes_fail()
        self.assertEqual(result, "success")
        self.assertEqual(dummy.counter, 2)

    async def test_retry_copy_max_attempts_exit(self):
        # Test that _retry_copy eventually calls sys.exit after repeated failures.
        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
        af.max_attempts = 2

        async def failing_copy(*args, **kwargs):
            raise Exception("copy failure")

        af._copy = failing_copy
        af._recover_label = AsyncMock()

        with self.assertRaises(SystemExit):
            await af._retry_copy("label", {"csv_path": "dummy", "next_val": 1}, "v")

    async def test_async_context_manager(self):
        # Test __aenter__ and __aexit__.
        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
        af.connect = AsyncMock()
        af.con_pool = DummyPool()
        await af.__aenter__()
        await af.__aexit__(None, None, None)

    def test_del_cleanup(self):
        # Test the __del__ cleanup of temporary CSV files.
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)
        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192, save_temps=False)
        af.vertices = {"v": {"csv_path": temp_path}}
        af.edges = {}
        af.__del__()
        self.assertFalse(os.path.exists(temp_path))

    # --- Additional tests for the specified code sections ---

    async def test_connect_pool_timeout(self):
        """Test that connect() retries and eventually raises PoolTimeout."""

        class DummyPoolTimeout:
            async def open(self):
                raise PoolTimeout("Simulated timeout")

            async def wait(self):
                pass

            async def close(self):
                pass

            def connection(self):
                return DummyConnectionContextManager()

        with patch.object(
            AgeFreighter, "parse_dsn", return_value={"host": "dummy_host"}
        ):
            with patch(
                "agefreighter.agefreighter.AsyncConnectionPool",
                return_value=DummyPoolTimeout(),
            ):
                af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
                with self.assertRaises(PoolTimeout):
                    await af.connect(
                        af.dsn_w_option,
                        af.max_connections,
                        af.min_connections,
                        af.max_attempts,
                        af.retry_delay,
                    )

    async def test_connect_generic_exception(self):
        """Test that connect() raises a generic exception immediately."""

        class DummyPoolGeneric:
            async def open(self):
                raise Exception("Generic failure")

            async def wait(self):
                pass

            async def close(self):
                pass

            def connection(self):
                return DummyConnectionContextManager()

        with patch.object(
            AgeFreighter, "parse_dsn", return_value={"host": "dummy_host"}
        ):
            with patch(
                "agefreighter.agefreighter.AsyncConnectionPool",
                return_value=DummyPoolGeneric(),
            ):
                af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
                with self.assertRaises(Exception) as cm:
                    await af.connect(
                        af.dsn_w_option,
                        af.max_connections,
                        af.min_connections,
                        af.max_attempts,
                        af.retry_delay,
                    )
                self.assertIn("Generic failure", str(cm.exception))

    async def test_close_exception_handling(self):
        """Test that close() handles exceptions from con_pool.close() gracefully."""

        class DummyPoolCloseFailure:
            opened = True

            async def close(self):
                raise Exception("Close failure")

        af = AgeFreighter(dsn="dummy_dsn", chunk_size=8192)
        af.con_pool = DummyPoolCloseFailure()
        try:
            await af.close()
        except Exception:
            self.fail("close() should handle exceptions without propagating them.")

    async def test_copy_success(self):
        """Test the _copy method for a successful COPY operation."""
        # Create a temporary CSV file with known content.
        content = "id,data\n1,test\n"
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            csv_path = tmp.name

        af = AgeFreighter(dsn="dummy_dsn", chunk_size=1024)
        af.graph_name = "test_graph"
        af.con_pool = DummyPool()  # Uses our dummy connection and cursor

        # Patch quoted_graph_name to return a properly quoted graph name.
        with patch.object(
            af.__class__, "quoted_graph_name", return_value='"test_graph"'
        ):
            await af._copy("test_graph", csv_path, "test_label", 100, "v")
            # If no exception is raised, the test passes.
            self.assertTrue(True)

        os.remove(csv_path)

    async def test_write_csv_empty(self):
        """Test that write_csv returns an empty string when no data is provided."""
        af = AgeFreighter(
            dsn="dummy_dsn", chunk_size=8192, output_dir=tempfile.gettempdir()
        )
        result = await af.write_csv("empty_label", "v", [])
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
