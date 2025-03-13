#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
import shutil
from datetime import datetime
from unittest.mock import patch, AsyncMock
import aiofiles

# Import our generator module (adjust the import path as needed)
import agefreighter.generator as generator


class TestGeneratorUtils(unittest.TestCase):
    def test_get_timestamp(self):
        ts = generator.get_timestamp()
        # Both keys should be present and equal (since the same datetime is used)
        self.assertIn("available_since", ts)
        self.assertIn("inserted_at", ts)
        self.assertEqual(ts["available_since"], ts["inserted_at"])
        self.assertIsInstance(ts["available_since"], datetime)

    def test_prefixed_props(self):
        # When props is provided.
        props = {"a": 1, "b": 2}
        expected = {"test_a": 1, "test_b": 2}
        self.assertEqual(generator.prefixed_props("test_", props), expected)
        # When props is None.
        self.assertEqual(generator.prefixed_props("test_", None), {})

    def test_generate_base58_dummy_data(self):
        s = generator.generate_base58_dummy_data(16)
        self.assertEqual(len(s), 16)
        base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        for ch in s:
            self.assertIn(ch, base58_chars)


class TestPutCSV(unittest.IsolatedAsyncioTestCase):
    async def test_put_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Instead of using a Node class, we use simple dictionaries.
            node1 = {"id": "1", "a": "val1", "b": "val2"}
            node2 = {"id": "2", "a": "val3", "b": "val4"}
            data = [node1, node2]
            file_path = await generator.put_csv(tmp, "testfile", data)
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            self.assertIn('"id"', content)
            self.assertIn('"a"', content)
            self.assertIn('"b"', content)
            self.assertIn('"1"', content)
            self.assertIn('"val1"', content)


class TestGenerateNodes(unittest.IsolatedAsyncioTestCase):
    async def test_generate_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cls_name, nodes = await generator.generate_nodes("Customer", 3, tmp)
            self.assertEqual(cls_name, "Customer")
            self.assertEqual(len(nodes), 3)
            # Check that CSV file was created (filename is lowercased).
            file_path = os.path.join(tmp, "customer.csv")
            self.assertTrue(os.path.exists(file_path))


class TestGenerateEdges(unittest.IsolatedAsyncioTestCase):
    async def test_generate_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create dummy nodes data for two types.
            # Instead of Node objects, use simple dictionaries.
            node_customer = {"id": "1", "name": "Alice"}
            node_product = {"id": "1", "name": "Widget"}
            nodes_data = {"Customer": [node_customer], "Product": [node_product]}
            # Prepare a property list for an edge.
            prop_list = [{"count": 2, "start": "Customer", "end": "Product"}]
            await generator.generate_edges("Bought", prop_list, nodes_data, tmp)
            # Expected filename: "bought_customer_product.csv" (all lower case)
            file_path = os.path.join(tmp, "bought_customer_product.csv")
            self.assertTrue(os.path.exists(file_path))


class TestGeneratorMain(unittest.IsolatedAsyncioTestCase):
    async def test_main_creates_directory(self):
        # Patch generation functions to avoid heavy data creation.
        with patch(
            "agefreighter.generator.generate_nodes",
            new=AsyncMock(return_value=("TestNode", [])),
        ):
            with patch(
                "agefreighter.generator.generate_edges",
                new=AsyncMock(return_value=None),
            ):
                with patch(
                    "agefreighter.generator.generate_complete_data",
                    new=AsyncMock(return_value=None),
                ):
                    # Patch datetime.now() to return a fixed timestamp.
                    fixed_time = datetime(2025, 1, 1, 0, 0, 0)
                    with patch("agefreighter.generator.datetime") as mock_datetime:
                        mock_datetime.now.return_value = fixed_time
                        await generator.main(pattern_no=1, log_level=40)
                    # Expected directory name.
                    dir_name = "generated_dummy_" + fixed_time.strftime("%Y%m%d_%H%M%S")
                    self.assertTrue(os.path.exists(dir_name))
                    shutil.rmtree(dir_name)


if __name__ == "__main__":
    unittest.main()
