#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import logging
from unittest.mock import patch, AsyncMock
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the module under test.
from agefreighter.cosmosnosqlexporter import CosmosNoSQLExporter


# --- Dummy Cosmos DB Client Classes ---
class DummyContainerClient:
    def __init__(self, items=None, raise_exception=False):
        self.items = items if items is not None else []
        self.raise_exception = raise_exception

    def query_items(self, query, enable_cross_partition_query):
        if self.raise_exception:
            raise Exception("query_items failure")
        return self.items


class DummyDatabaseClient:
    def __init__(self, container_client):
        self._container_client = container_client

    def get_container_client(self, container_name):
        return self._container_client


class DummyCosmosClient:
    def __init__(self, endpoint, credential):
        self.endpoint = endpoint
        self.credential = credential
        # We'll set container_client later via get_database_client.
        self.container_client = None

    def get_database_client(self, database_name):
        # Return a dummy database client with a preset container client.
        return DummyDatabaseClient(self.container_client)


# --- Test Suite for CosmosNoSQLExporter ---
class TestCosmosNoSQLFreighter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Patch CosmosClient in cosmosnosqlexporter so that our dummy is used.
        patcher = patch(
            "agefreighter.cosmosnosqlexporter.CosmosClient", new=DummyCosmosClient
        )
        self.addCleanup(patcher.stop)
        self.mock_cosmos_client = patcher.start()

        # Create a dummy container client we can control.
        self.dummy_container = DummyContainerClient()

        # Make DummyCosmosClient.get_database_client return a client that wraps our dummy container.
        DummyCosmosClient.get_database_client = (
            lambda self_obj, db: DummyDatabaseClient(self.dummy_container)
        )

        # Create an instance of CosmosNoSQLExporter with dummy parameters.
        self.instance = CosmosNoSQLExporter(
            dsn="dummy_dsn",
            min_connections=1,
            max_connections=1,
            cosmos_endpoint="dummy_endpoint",
            cosmos_key="dummy_key",
            cosmos_database="dummy_db",
            cosmos_container="dummy_container",
            trial=False,
            save_temps=False,
            progress=False,
            graph_name="dummy_graph",
            chunk_size=1024,
            log_level=logging.DEBUG,
        )
        # Patch AgeFreighter–inherited methods to avoid real I/O.
        self.instance.create_label_type = AsyncMock(return_value=None)
        self.instance.get_first_id = AsyncMock(return_value=1000)
        self.instance.write_csv = AsyncMock(return_value="dummy.csv")
        # Override _filter_properties so that we can simulate a successful extraction.
        self.instance._filter_properties = lambda doc, exclude_keys: {
            k: v[0]["_value"] for k, v in doc.items() if k not in exclude_keys
        }
        # Patch set_up_graph to be a no‐op.
        self.instance.set_up_graph = AsyncMock(return_value=None)

    async def test_init_success(self):
        # Verify that __init__ sets Cosmos DB client attributes.
        self.assertEqual(self.instance.cosmos_endpoint, "dummy_endpoint")
        self.assertIsNotNone(self.instance.client)
        self.assertIsNotNone(self.instance.container_client)

    async def test_init_failure(self):
        # Test that if CosmosClient.__init__ raises, the exception propagates.
        with patch(
            "agefreighter.cosmosnosqlexporter.CosmosClient",
            side_effect=Exception("init failure"),
        ):
            with self.assertRaises(SystemExit) as cm:
                CosmosNoSQLExporter(
                    dsn="dummy_dsn",
                    min_connections=1,
                    max_connections=1,
                    cosmos_endpoint="bad_endpoint",
                    cosmos_key="bad_key",
                    cosmos_database="bad_db",
                    cosmos_container="bad_container",
                    trial=False,
                    save_temps=False,
                    progress=False,
                    graph_name="dummy_graph",
                    chunk_size=1024,
                    log_level=logging.DEBUG,
                )
            self.assertEqual(cm.exception.code, 1)

    async def test_aenter_aexit(self):
        # Patch connect() to avoid opening a real connection pool.
        self.instance.connect = AsyncMock(return_value=None)
        async with self.instance as inst:
            self.assertIs(inst, self.instance)

    async def test_fetch_edges_trial(self):
        # Test _fetch_edges when trial is True (query uses TOP ...).
        self.instance.trial = True
        dummy_edges = [
            {
                "label": "edge1",
                "_vertexId": "n1",
                "_vertexLabel": "v1",
                "_sink": "n2",
                "_sinkLabel": "v2",
            }
        ]
        self.dummy_container.items = dummy_edges
        result = await self.instance._fetch_edges()
        self.assertEqual(result, dummy_edges)

    async def test_fetch_edges_full(self):
        # Test _fetch_edges when trial is False.
        self.instance.trial = False
        dummy_edges = [
            {
                "label": "edge1",
                "_vertexId": "n1",
                "_vertexLabel": "v1",
                "_sink": "n2",
                "_sinkLabel": "v2",
            }
        ]
        self.dummy_container.items = dummy_edges
        result = await self.instance._fetch_edges()
        self.assertEqual(result, dummy_edges)

    def test_batch_ids(self):
        # Test that _batch_ids splits a set of IDs into batches of the given size.
        ids = set(str(i) for i in range(250))
        batches = self.instance._batch_ids(ids)
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 100)
        self.assertEqual(len(batches[1]), 100)
        self.assertEqual(len(batches[2]), 50)

    async def test_fetch_nodes_batch_success(self):
        # Test _fetch_nodes_batch returns the dummy node list.
        dummy_nodes = [{"id": "n1"}, {"id": "n2"}]
        self.dummy_container.items = dummy_nodes
        result = await self.instance._fetch_nodes_batch(["n1", "n2"])
        self.assertEqual(result, dummy_nodes)

    async def test_fetch_nodes_batch_exception(self):
        # Test that _fetch_nodes_batch raises when query_items fails.
        self.dummy_container.raise_exception = True
        with self.assertRaises(Exception) as cm:
            await self.instance._fetch_nodes_batch(["n1"])
        self.assertIn("query_items failure", str(cm.exception))
        self.dummy_container.raise_exception = False

    async def test_fetch_all_nodes_success(self):
        # Test that _fetch_all_nodes fetches nodes concurrently.
        async def fake_fetch(batch):
            # Return one node per id in the batch.
            return [
                {"id": node_id, "label": "v", "prop": [{"_value": "val"}]}
                for node_id in batch
            ]

        self.instance._fetch_nodes_batch = fake_fetch
        node_ids = {"n1", "n2", "n3"}
        result = await self.instance._fetch_all_nodes(node_ids)
        self.assertEqual(len(result), 3)
        for node_id in node_ids:
            self.assertIn(node_id, result)

    def test_filter_properties_success(self):
        # Test _filter_properties extracts values as expected.
        doc = {
            "id": "n1",
            "prop": [{"_value": "value1"}],
            "extra": [{"_value": "value2"}],
        }
        result = self.instance._filter_properties(doc, exclude_keys=["id"])
        self.assertEqual(result, {"prop": "value1", "extra": "value2"})

    def test_filter_properties_exception(self):
        # Test that _filter_properties raises if a value is not indexable.
        doc = {"id": "n1", "prop": "not a list"}
        with self.assertRaises(Exception):
            self.instance._filter_properties(doc, exclude_keys=["id"])

    async def test_fetch_documents_success(self):
        # Test fetch_documents end-to-end under normal conditions.
        dummy_edges = [
            {
                "label": "edge1",
                "id": "e1",
                "_vertexId": "n1",
                "_vertexLabel": "v1",
                "_sink": "n2",
                "_sinkLabel": "v2",
            }
        ]

        async def fake_fetch_edges():
            return dummy_edges

        self.instance._fetch_edges = fake_fetch_edges

        dummy_nodes = {
            "n1": {"id": "n1", "label": "v1", "prop": [{"_value": "v1_val"}]},
            "n2": {"id": "n2", "label": "v2", "prop": [{"_value": "v2_val"}]},
        }

        async def fake_fetch_all_nodes(node_ids):
            return dummy_nodes

        self.instance._fetch_all_nodes = fake_fetch_all_nodes

        # Override _filter_properties so that it returns a dict including "id" (for mapping).
        self.instance._filter_properties = lambda doc, exclude_keys: {
            "id": doc["id"],
            "prop": doc["prop"][0]["_value"],
        }

        vertex_args, edge_args = await self.instance.fetch_documents()
        # Expect vertex_args to include both "v1" and "v2" and edge_args to include "edge1".
        self.assertIn("v1", vertex_args)
        self.assertIn("v2", vertex_args)
        self.assertIn("edge1", edge_args)

    async def test_fetch_documents_edge_missing_label(self):
        # Test fetch_documents raises ValueError if an edge document is missing its label.
        dummy_edges = [
            {
                "id": "e1",
                "_vertexId": "n1",
                "_vertexLabel": "v1",
                "_sink": "n2",
                "_sinkLabel": "v2",
            }
        ]

        async def fake_fetch_edges():
            return dummy_edges

        self.instance._fetch_edges = fake_fetch_edges
        with self.assertRaises(ValueError) as cm:
            await self.instance.fetch_documents()
        self.assertIn("Edge document does not have a label", str(cm.exception))

    async def test_fetch_documents_node_missing_label(self):
        # Test fetch_documents raises ValueError if a node document is missing its label.
        dummy_edges = [
            {
                "label": "edge1",
                "id": "e1",
                "_vertexId": "n1",
                "_vertexLabel": "v1",
                "_sink": "n2",
                "_sinkLabel": "v2",
            }
        ]

        async def fake_fetch_edges():
            return dummy_edges

        self.instance._fetch_edges = fake_fetch_edges

        # Return a node missing the "label" key.
        dummy_nodes = {
            "n1": {"id": "n1", "prop": [{"_value": "v1_val"}]},
            "n2": {"id": "n2", "label": "v2", "prop": [{"_value": "v2_val"}]},
        }

        async def fake_fetch_all_nodes(node_ids):
            return dummy_nodes

        self.instance._fetch_all_nodes = fake_fetch_all_nodes

        with self.assertRaises(ValueError) as cm:
            await self.instance.fetch_documents()
        self.assertIn("Node document does not have a label", str(cm.exception))

    async def test_fetch_documents_vertex_write_csv_exception(self):
        # Test that an exception in write_csv during vertex processing is propagated.
        dummy_edges = [
            {
                "label": "edge1",
                "id": "e1",
                "_vertexId": "n1",
                "_vertexLabel": "v1",
                "_sink": "n2",
                "_sinkLabel": "v2",
            }
        ]

        async def fake_fetch_edges():
            return dummy_edges

        self.instance._fetch_edges = fake_fetch_edges

        dummy_nodes = {
            "n1": {"id": "n1", "label": "v1", "prop": [{"_value": "v1_val"}]},
            "n2": {"id": "n2", "label": "v2", "prop": [{"_value": "v2_val"}]},
        }

        async def fake_fetch_all_nodes(node_ids):
            return dummy_nodes

        self.instance._fetch_all_nodes = fake_fetch_all_nodes

        self.instance.write_csv = AsyncMock(side_effect=Exception("csv error"))

        with self.assertRaises(Exception) as cm:
            await self.instance.fetch_documents()
        self.assertIn("csv error", str(cm.exception))

    async def test_fetch_documents_edge_missing_mapping(self):
        # Test that if mapping for an edge is missing (KeyError), the edge is skipped.
        dummy_edges = [
            {
                "label": "edge1",
                "id": "e1",
                "_vertexId": "n1",
                "_vertexLabel": "v1",
                "_sink": "n2",
                "_sinkLabel": "v2",
            }
        ]

        async def fake_fetch_edges():
            return dummy_edges

        self.instance._fetch_edges = fake_fetch_edges

        # Return nodes so that _filter_properties (as overridden) does not include an "id" key in its result.
        dummy_nodes = {
            "n1": {"id": "n1", "label": "v1", "prop": [{"_value": "v1_val"}]},
            "n2": {"id": "n2", "label": "v2", "prop": [{"_value": "v2_val"}]},
        }

        async def fake_fetch_all_nodes(node_ids):
            return dummy_nodes

        self.instance._fetch_all_nodes = fake_fetch_all_nodes

        # Override _filter_properties so that it does not include "id" (causing new_map to be empty).
        self.instance._filter_properties = lambda doc, exclude_keys: {
            "prop": doc["prop"][0]["_value"]
        }

        vertex_args, edge_args = await self.instance.fetch_documents()
        # Even though vertex_args is built, in edge processing the mapping lookup will fail and the edge will be skipped,
        # but write_csv will still be called with an (empty) list.
        self.assertIn("edge1", edge_args)
        self.assertEqual(edge_args["edge1"]["csv_path"], "dummy.csv")

    async def test_export(self):
        # Test export method: it should call set_up_graph and fetch_documents,
        # then set self.vertices and self.edges.
        async def fake_fetch_documents():
            return (
                {"v1": {"csv_path": "v1.csv", "original_id": "id", "next_val": "10"}},
                {"edge1": {"csv_path": "e1.csv", "original_id": "id", "next_val": "5"}},
            )

        self.instance.fetch_documents = fake_fetch_documents
        await self.instance.export()
        self.assertEqual(
            self.instance.vertices,
            {
                "v1": {
                    "csv_path": "v1.csv".replace("\\", "\\\\"),
                    "original_id": "id",
                    "next_val": "10",
                }
            },
        )
        self.assertEqual(
            self.instance.edges,
            {
                "edge1": {
                    "csv_path": "e1.csv".replace("\\", "\\\\"),
                    "original_id": "id",
                    "next_val": "5",
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
