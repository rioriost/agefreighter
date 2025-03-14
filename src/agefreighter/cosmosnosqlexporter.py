#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for exporting data from a Cosmos NoSQL database to CSV files and
loading them using AgeFreighter.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Set, Tuple

from azure.cosmos import CosmosClient  # type: ignore
from .agefreighter import AgeFreighter

logging.getLogger("azure.core").setLevel(logging.WARNING)

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CosmosNoSQLExporter(AgeFreighter):
    """
    Exports nodes and edges from a Cosmos NoSQL database into CSV files for AGE COPY import.
    """

    def __init__(
        self,
        dsn: str,
        min_connections: int,
        max_connections: int,
        cosmos_endpoint: str,
        cosmos_key: str,
        cosmos_database: str,
        cosmos_container: str,
        trial: bool,
        no_of_edges_trial: int,
        save_temps: bool,
        progress: bool,
        graph_name: str,
        chunk_size: int,
        log_level: int = logging.INFO,
    ) -> None:
        log.setLevel(log_level)
        super().__init__(
            dsn=dsn,
            min_connections=min_connections,
            max_connections=max_connections,
            save_temps=save_temps,
            progress=progress,
            chunk_size=chunk_size,
            log_level=log_level,
        )
        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_key = cosmos_key
        self.cosmos_database = cosmos_database
        self.cosmos_container = cosmos_container
        self.batch_size = 100
        self.trial = trial
        self.graph_name = graph_name
        self.id_maps: Dict[str, Dict[str, int]] = {}
        self.no_of_edges_trial = no_of_edges_trial

        try:
            self.client = CosmosClient(self.cosmos_endpoint, credential=self.cosmos_key)
            self.database_client = self.client.get_database_client(self.cosmos_database)
            self.container_client = self.database_client.get_container_client(
                self.cosmos_container
            )
            log.info("Connected to Cosmos DB at %s.", self.cosmos_endpoint)
        except Exception as exc:
            log.error("Error connecting to Cosmos DB: %s", exc)
            sys.exit(1)

    async def __aenter__(self) -> "CosmosNoSQLExporter":
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await super().__aexit__(exc_type, exc_value, traceback)

    async def _fetch_edges(self) -> List[Dict[str, Any]]:
        """Fetch all edge documents from Cosmos DB."""
        query = (
            f"SELECT TOP {self.no_of_edges_trial} * FROM c WHERE c._isEdge = true"
            if self.trial
            else "SELECT * FROM c WHERE c._isEdge = true"
        )
        try:
            edges = await asyncio.to_thread(
                lambda: list(
                    self.container_client.query_items(
                        query=query, enable_cross_partition_query=True
                    )
                )
            )
        except Exception as exc:
            log.exception("Failed to fetch edge documents: %s", exc)
            raise
        log.info("Fetched %d edge documents.", len(edges))
        return edges

    def _batch_ids(self, ids: Set[str]) -> List[List[str]]:
        """Split a set of IDs into batches."""
        ids_list = list(ids)
        return [
            ids_list[i : i + self.batch_size]
            for i in range(0, len(ids_list), self.batch_size)
        ]

    async def _fetch_nodes_batch(self, id_batch: List[str]) -> List[Dict[str, Any]]:
        """Fetch node documents for a batch of IDs."""
        id_list_str = ", ".join(f"'{node_id}'" for node_id in id_batch)
        query = f"SELECT * FROM c WHERE c.id IN ({id_list_str})"
        try:
            nodes = await asyncio.to_thread(
                lambda: list(
                    self.container_client.query_items(
                        query=query, enable_cross_partition_query=True
                    )
                )
            )
        except Exception as exc:
            log.exception("Failed to fetch node batch [%s...]: %s", id_batch[:3], exc)
            raise
        return nodes

    async def _fetch_all_nodes(self, node_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch all node documents concurrently and return a mapping
        of node ID to its document.
        """
        batches = self._batch_ids(node_ids)
        tasks = [self._fetch_nodes_batch(batch) for batch in batches]
        try:
            results = await asyncio.gather(*tasks)
        except Exception as exc:
            log.exception("Failed to fetch all node documents: %s", exc)
            raise
        nodes: Dict[str, Dict[str, Any]] = {}
        for batch_nodes in results:
            for node in batch_nodes:
                node_id = node.get("id")
                if node_id:
                    nodes[node_id] = node
        log.info("Fetched %d node documents.", len(nodes))
        return nodes

    def _filter_properties(
        self, document: Dict[str, Any], exclude_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Extract and return properties from the document,
        excluding specified keys.
        """
        try:
            return {
                k: v[0]["_value"] for k, v in document.items() if k not in exclude_keys
            }
        except Exception as exc:
            log.exception(
                "Failed to filter properties from document %s: %s",
                document.get("id", "unknown"),
                exc,
            )
            raise

    async def fetch_documents(
        self,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Fetch documents from Cosmos NoSQL database, group them into vertices and edges,
        write CSV files, and return file path arguments.

        Returns:
            A tuple of two dictionaries: vertex_args and edge_args.
        """
        try:
            # Fetch edge documents.
            edges = await self._fetch_edges()
        except Exception as exc:
            log.exception("Error during fetching edge documents: %s", exc)
            raise

        # Group edge documents by label and collect unique node IDs.
        grouped_edges: Dict[str, List[Dict[str, Any]]] = {}
        node_ids: Set[str] = set()
        for edge in edges:
            edge_label = edge.get("label")
            if not edge_label:
                log.error("Edge document missing label: %s", edge)
                raise ValueError("Edge document does not have a label.")
            grouped_edges.setdefault(edge_label, []).append(edge)
            if edge.get("_vertexId"):
                node_ids.add(edge["_vertexId"])
            if edge.get("_sink"):
                node_ids.add(edge["_sink"])

        # Fetch node documents concurrently.
        try:
            nodes_map = await self._fetch_all_nodes(node_ids)
        except Exception as exc:
            log.exception("Error during fetching node documents: %s", exc)
            raise

        # Group node documents by their label.
        grouped_vertices: Dict[str, List[Dict[str, Any]]] = {}
        for node in nodes_map.values():
            vertex_label = node.get("label")
            if not vertex_label:
                log.error("Node document missing label: %s", node)
                raise ValueError("Node document does not have a label.")
            grouped_vertices.setdefault(vertex_label, []).append(node)

        vertex_args: Dict[str, Dict[str, Any]] = {}
        # Process vertices.
        for label, documents in grouped_vertices.items():
            try:
                await self.create_label_type(label_type="vertex", value=label)
                first_id = await self.get_first_id(self.graph_name, label)
            except Exception as exc:
                log.exception("Error during vertex setup for label %s: %s", label, exc)
                raise

            nodes = []
            for document in documents:
                try:
                    node = {
                        "id": document["id"],
                        **self._filter_properties(
                            document,
                            exclude_keys=[
                                "label",
                                "id",
                                "pk",
                                "_rid",
                                "_self",
                                "_etag",
                                "_attachments",
                                "_ts",
                            ],
                        ),
                    }
                    nodes.append(node)
                except Exception as exc:
                    log.exception("Error processing node document: %s", exc)
                    continue

            # Deduplicate nodes by original ID.
            unique_nodes = {node["id"]: node for node in nodes}
            nodes = list(unique_nodes.values())
            all_data = [
                {"id": idx + first_id, "properties": node}
                for idx, node in enumerate(nodes)
            ]
            try:
                new_map: Dict[str, int] = {}
                for item in all_data:
                    properties = item.get("properties")
                    if isinstance(properties, dict) and "id" in properties:
                        id_val = item.get("id")
                        if isinstance(id_val, (int, str)):
                            new_map[properties["id"]] = int(id_val)
                        else:
                            log.error(
                                "Unexpected type for item['id']: %s", type(id_val)
                            )
                            raise ValueError(
                                f"Expected item['id'] to be int or str convertible to int, got {type(id_val)}"
                            )
                self.id_maps[label] = new_map
                csv_path = await self.write_csv(label, "v", all_data)
            except Exception as exc:
                log.exception(
                    "Error writing CSV for vertices with label %s: %s", label, exc
                )
                raise

            vertex_args[label] = {
                "csv_path": csv_path.replace("\\", "\\\\"),
                "original_id": "id",
                "next_val": str(len(all_data)),
            }

        edge_args: Dict[str, Dict[str, Any]] = {}
        # Process edges.
        for edge_label, documents in grouped_edges.items():
            try:
                await self.create_label_type(label_type="edge", value=edge_label)
                first_id = await self.get_first_id(self.graph_name, edge_label)
            except Exception as exc:
                log.exception(
                    "Error during edge setup for label %s: %s", edge_label, exc
                )
                raise

            edges_list = []
            for idx, document in enumerate(documents):
                try:
                    new_start_id = self.id_maps[document["_vertexLabel"]][
                        document["_vertexId"]
                    ]
                    new_end_id = self.id_maps[document["_sinkLabel"]][document["_sink"]]
                except KeyError as key_exc:
                    log.warning(
                        "Missing mapping for document %s: %s",
                        document.get("id", "unknown"),
                        key_exc,
                    )
                    continue
                try:
                    edge_item = {
                        "id": idx + first_id,
                        "start_id": new_start_id,
                        "end_id": new_end_id,
                        **self._filter_properties(
                            document,
                            exclude_keys=[
                                "label",
                                "id",
                                "_sink",
                                "_sinkLabel",
                                "_sinkPartition",
                                "_vertexId",
                                "_vertexLabel",
                                "_isEdge",
                                "pk",
                                "_rid",
                                "_self",
                                "_etag",
                                "_attachments",
                                "_ts",
                            ],
                        ),
                    }
                    edges_list.append(edge_item)
                except Exception as exc:
                    log.exception(
                        "Error processing edge document %s: %s",
                        document.get("id", "unknown"),
                        exc,
                    )
                    continue

            try:
                csv_path = await self.write_csv(edge_label, "e", edges_list)
            except Exception as exc:
                log.exception(
                    "Error writing CSV for edges with label %s: %s", edge_label, exc
                )
                raise
            edge_args[edge_label] = {
                "csv_path": csv_path.replace("\\", "\\\\"),
                "original_id": "id",
                "next_val": str(len(all_data)),
            }

        return vertex_args, edge_args

    async def export(self) -> None:
        """
        Main export function to process both nodes and edges.
        """
        try:
            await self.set_up_graph(graph_name=self.graph_name, create_graph=True)
            vertex_args, edge_args = await self.fetch_documents()
        except Exception as exc:
            log.exception("Error during export process: %s", exc)
            raise

        self.vertices = vertex_args
        self.edges = edge_args
