#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for exporting data from a Neo4j database to CSV files and loading them using AgeFreighter.
"""

import asyncio
import concurrent.futures
import logging
import sys
from typing import List, Dict, Any, Union

from neo4j import GraphDatabase, Result  # type: ignore
from .agefreighter import AgeFreighter

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Neo4jExporter(AgeFreighter):
    """
    Exports nodes and edges from a Neo4j database into CSV files.
    """

    def __init__(
        self,
        dsn: str,
        min_connections: int,
        max_connections: int,
        uri: str,
        user: str,
        password: str,
        database: str,
        trial: bool,
        no_of_edges_trial: int,
        save_temps: bool,
        progress: bool,
        graph_name: str,
        chunk_size: int,
        log_level: int = logging.INFO,
        **kwargs: Any,
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
        self.uri: str = uri
        self.user: str = user
        self.password: str = password
        self.database: str = database  # optional
        self.graph_name: str = graph_name
        self.trial: bool = trial
        self.trial_nodes_by_label: Dict[str, Dict[str, List[str]]] = {}
        self.id_maps: Dict[str, Dict[str, int]] = {}
        self.max_attempts = 3
        self.retry_delay = 1
        self.no_of_edges_trial = no_of_edges_trial

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            with self.driver.session() as session:
                session.run("RETURN 1").single()
            log.info("Connected to Neo4j at %s.", uri)
        except Exception as e:
            log.error("Error connecting to Neo4j: %s", e)
            sys.exit(1)

    async def __aenter__(self) -> "Neo4jExporter":
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.driver.close()
            log.debug("Closed Neo4j driver.")
        except Exception as e:
            log.error("Error closing Neo4j driver: %s", e)
        await super().__aexit__(exc_type, exc_value, traceback)

    def get_labels(self) -> List[str]:
        """
        Retrieve all node labels from Neo4j.
        """
        try:
            with self.driver.session() as session:
                query = "CALL db.labels() YIELD label RETURN label;"
                result: Result = session.run(query)
                labels = [record["label"] for record in result]
                # Check for nodes with no labels
                query_no_label = (
                    "MATCH (n) WHERE size(labels(n)) = 0 RETURN COUNT(n) AS cnt"
                )
                result_no_label: Result = session.run(query_no_label)
                record = result_no_label.single()
                if record and int(record["cnt"]) > 0:
                    labels.append("NO_LABEL")
            return labels
        except Exception as e:
            log.exception("Error retrieving labels: %s", e)
            return []

    def get_relationship_types(self) -> List[str]:
        """
        Retrieve all relationship types from Neo4j.
        """
        try:
            with self.driver.session() as session:
                query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType;"
                result: Result = session.run(query)
                types = [record["relationshipType"] for record in result]
            return types
        except Exception as e:
            log.exception("Error retrieving relationship types: %s", e)
            return []

    def _count_nodes(self, label: Union[str, List[str]]) -> int:
        """
        Count the number of nodes for a given label.
        """
        try:
            with self.driver.session() as session:
                if isinstance(label, str):
                    if label == "NO_LABEL":
                        query = (
                            "MATCH (n) WHERE size(labels(n)) = 0 RETURN COUNT(n) AS cnt"
                        )
                    else:
                        query = f"MATCH (n:{label}) RETURN COUNT(n) AS cnt"
                else:
                    labels_str = ", ".join(f'"{lbl}"' for lbl in label)
                    query = f"MATCH (n) WHERE ANY(lbl IN labels(n) WHERE lbl IN [{labels_str}]) RETURN COUNT(n) AS cnt"
                result: Result = session.run(query)
                record = result.single()
                return int(record["cnt"]) if record else 0
        except Exception as e:
            log.exception("Error counting nodes for label %s: %s", label, e)
            return 0

    def _count_edges(self, rel_type: str) -> int:
        """
        Count the number of edges for a given relationship type.
        """
        try:
            with self.driver.session() as session:
                query = f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) AS cnt"
                result: Result = session.run(query)
                record = result.single()
                return int(record["cnt"]) if record else 0
        except Exception as e:
            log.exception("Error counting edges for '%s': %s", rel_type, e)
            return 0

    def _fetch_nodes_chunk(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch a chunk of nodes based on the provided Cypher query.
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                data = [
                    {"_elementid": record["n"].element_id, **record["n"]._properties}
                    for record in result
                ]
            return data
        except Exception as e:
            log.exception("Error fetching nodes chunk: %s", e)
            return []

    def _fetch_nodes_by_ids_chunk(
        self, label: str, node_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch nodes for a given label with specific element IDs.
        """
        try:
            with self.driver.session() as session:
                if label == "NO_LABEL":
                    query = "MATCH (n) WHERE size(labels(n)) = 0 AND elementId(n) IN $node_ids RETURN n"
                else:
                    query = (
                        f"MATCH (n:{label}) WHERE elementId(n) IN $node_ids RETURN n"
                    )
                result = session.run(query, node_ids=node_ids)
                data = [
                    {"_elementid": record["n"].element_id, **record["n"]._properties}
                    for record in result
                ]
            return data
        except Exception as e:
            log.exception("Error fetching nodes by IDs for label '%s': %s", label, e)
            return []

    def _fetch_edge_chunk(
        self, rel_type: str, skip: int, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch a chunk of edges for a given relationship type using pagination.
        """
        try:
            with self.driver.session() as session:
                query = f"MATCH (m)-[r:{rel_type}]->(n) SKIP {skip} LIMIT {limit} RETURN m, r, n"
                result = session.run(query)
                data = []
                for record in result:
                    m_labels = (
                        list(record["m"].labels) if record["m"].labels else ["NO_LABEL"]
                    )
                    n_labels = (
                        list(record["n"].labels) if record["n"].labels else ["NO_LABEL"]
                    )
                    for m_label in m_labels:
                        for n_label in n_labels:
                            entry = {
                                "_elementid": record["r"].element_id,
                                "start_id": record["m"].element_id,
                                "start_vertex_type": m_label,
                                "end_id": record["n"].element_id,
                                "end_vertex_type": n_label,
                                **record["r"]._properties,
                            }
                            data.append(entry)
            return data
        except Exception as e:
            log.exception(
                "Error fetching edge chunk for '%s' (skip %d): %s", rel_type, skip, e
            )
            return []

    async def export_nodes(
        self, thread_pool: concurrent.futures.ThreadPoolExecutor
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export nodes from Neo4j to CSV files and return a mapping for AGE import.
        """
        loop = asyncio.get_running_loop()
        vertex_args: Dict[str, Dict[str, Any]] = {}
        combined_nodes: Dict[str, List[str]] = {}

        # In trial mode, combine provided node ID mappings; otherwise, fetch all labels.
        if self.trial:
            for mapping in self.trial_nodes_by_label.values():
                for label, ids in mapping.items():
                    combined_nodes.setdefault(label, []).extend(ids)
            labels = list(combined_nodes.keys())
        else:
            labels = self.get_labels()

        for label in labels:
            try:
                # Create the vertex label in AGE
                await self.create_label_type(label_type="vertex", value=label)
            except Exception as e:
                log.error("Error creating label '%s': %s", label, e)
                sys.exit(1)

            try:
                first_id = await self.get_first_id(self.graph_name, label)
            except Exception as e:
                log.error("Error getting first ID for label '%s': %s", label, e)
                sys.exit(1)

            count = (
                len(combined_nodes.get(label, []))
                if self.trial
                else self._count_nodes(label)
            )
            log.info("Exporting %d nodes for label '%s'.", count, label)
            try:
                if self.trial:
                    tasks = [
                        loop.run_in_executor(
                            thread_pool,
                            self._fetch_nodes_by_ids_chunk,
                            label,
                            combined_nodes[label][skip : skip + int(self.chunk_size)],
                        )
                        for skip in range(0, count, int(self.chunk_size))
                    ]
                else:
                    query_template = (
                        "MATCH (n) WHERE size(labels(n)) = 0 SKIP {skip} LIMIT {limit} RETURN n"
                        if label == "NO_LABEL"
                        else f"MATCH (n:{label}) SKIP {{skip}} LIMIT {{limit}} RETURN n"
                    )
                    tasks = [
                        loop.run_in_executor(
                            thread_pool,
                            self._fetch_nodes_chunk,
                            query_template.format(
                                skip=skip, limit=int(self.chunk_size)
                            ),
                        )
                        for skip in range(0, count, int(self.chunk_size))
                    ]
                chunks = await asyncio.gather(*tasks)
            except Exception as e:
                log.exception(
                    "Error fetching nodes for label '%s': %s from Neo4j", label, e
                )
                sys.exit(1)

            try:
                # Flatten chunks and assign new IDs starting from first_id
                all_data = [
                    {"id": idx + first_id, "properties": item}
                    for sublist in chunks
                    for idx, item in enumerate(sublist)
                ]
                # Build ID mapping for later edge export
                self.id_maps[label] = {}
                for item in all_data:
                    props = item.get("properties")
                    if isinstance(props, dict) and "_elementid" in props:
                        raw_id = item.get("id")
                        if isinstance(raw_id, int):
                            id_value = raw_id
                        elif isinstance(raw_id, str):
                            try:
                                id_value = int(raw_id)
                            except Exception:
                                log.exception(
                                    "Unable to convert id value %s to int.", raw_id
                                )
                                raise
                        else:
                            log.error("Unexpected type for id: %s", type(raw_id))
                            raise ValueError(f"Unsupported type for id: {type(raw_id)}")
                        self.id_maps[label][props["_elementid"]] = id_value
                file_path = await self.write_csv(label, "v", all_data)
                file_path = file_path.replace("\\", "\\\\")
                vertex_args[label] = {
                    "csv_path": file_path,
                    "next_val": str(len(all_data)),
                }
            except Exception as e:
                log.exception("Error exporting nodes for label '%s': %s", label, e)
        return vertex_args

    async def export_edges(
        self, thread_pool: concurrent.futures.Executor
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export edges from Neo4j to CSV files and return a mapping for AGE import.
        """
        loop = asyncio.get_running_loop()
        edge_args: Dict[str, Dict[str, Any]] = {}
        types = self.get_relationship_types()
        for rel_type in types:
            try:
                # Create the edge label in AGE
                await self.create_label_type(label_type="edge", value=rel_type)
            except Exception as e:
                log.exception("Error creating edge type '%s': %s", rel_type, e)
                sys.exit(1)

            try:
                first_id = await self.get_first_id(self.graph_name, rel_type)
            except Exception as e:
                log.exception(
                    "Error getting first ID for edge type '%s': %s", rel_type, e
                )
                sys.exit(1)
            count = self._count_edges(rel_type)
            if self.trial:
                count = min(count, self.no_of_edges_trial)
                limit = min(int(self.chunk_size), count)
            else:
                limit = int(self.chunk_size)
            log.info("Exporting %d edges for relationship type '%s'.", count, rel_type)
            try:
                tasks = [
                    loop.run_in_executor(
                        thread_pool,
                        self._fetch_edge_chunk,
                        rel_type,
                        skip,
                        limit,
                    )
                    for skip in range(0, count, int(self.chunk_size))
                ]
                chunks = await asyncio.gather(*tasks)
                all_data = [
                    {
                        "id": idx + first_id,
                        "start_id": self.id_maps[item["start_vertex_type"]][
                            item["start_id"]
                        ],
                        "end_id": self.id_maps[item["end_vertex_type"]][item["end_id"]],
                        "properties": {
                            k: v
                            for k, v in item.items()
                            if k
                            not in [
                                "start_vertex_type",
                                "start_id",
                                "end_vertex_type",
                                "end_id",
                            ]
                        },
                    }
                    for sublist in chunks
                    for idx, item in enumerate(sublist)
                ]
                file_path = await self.write_csv(rel_type, "e", all_data)
                file_path = file_path.replace("\\", "\\\\")
                edge_args[rel_type] = {
                    "csv_path": file_path,
                    "next_val": str(len(all_data)),
                }
            except KeyError:
                log.error("Edge '%s' must include incomplete data.", rel_type)
            except Exception as e:
                log.exception("Error exporting edges for '%s': %s", rel_type, e)
        return edge_args

    async def list_nodes(self) -> None:
        """
        In trial mode, list nodes per relationship type for sampling.
        """
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as thread_pool:
            attempts = 0
            while True:
                attempts += 1
                try:
                    types = self.get_relationship_types()
                    break
                except Exception as e:
                    if attempts >= self.max_attempts:
                        log.error("Max attempts reached for getting relationship types")
                        sys.exit(1)
                    log.exception("Error getting relationship types: %s", e)
                    await asyncio.sleep(self.retry_delay)
            for rel_type in types:
                try:
                    count = min(self._count_edges(rel_type), self.no_of_edges_trial)
                    limit = min(int(self.chunk_size), count)
                    log.info("Listing nodes for relationship type '%s'.", rel_type)
                    tasks = [
                        loop.run_in_executor(
                            thread_pool,
                            self._fetch_edge_chunk,
                            rel_type,
                            skip,
                            limit,
                        )
                        for skip in range(0, count, limit)
                    ]
                    chunks = await asyncio.gather(*tasks)
                    all_data = [item for sublist in chunks for item in sublist]
                    nodes_by_label: Dict[str, List[str]] = {}
                    for record in all_data:
                        nodes_by_label.setdefault(
                            record["start_vertex_type"], []
                        ).append(record["start_id"])
                        nodes_by_label.setdefault(record["end_vertex_type"], []).append(
                            record["end_id"]
                        )
                    self.trial_nodes_by_label[rel_type] = nodes_by_label
                except Exception as e:
                    log.exception("Error listing nodes for '%s': %s", rel_type, e)

    async def export(self):
        """
        Main export function to process both nodes and edges.
        """
        thread_pool = concurrent.futures.ThreadPoolExecutor()
        attempts = 0
        while True:
            attempts += 1
            try:
                await self.set_up_graph(graph_name=self.graph_name, create_graph=True)
                break
            except Exception as e:
                if attempts >= self.max_attempts:
                    log.error("Max attempts reached for creating graph")
                    sys.exit(1)
                log.exception("Error creating graph: %s", e)
                await asyncio.sleep(self.retry_delay)
        try:
            if self.trial:
                await self.list_nodes()
            nodes_args = await self.export_nodes(thread_pool)
            if not nodes_args:
                log.error("No nodes exported.\nDoes the graph contain nodes?")
                sys.exit(1)
            edges_args = await self.export_edges(thread_pool)
            if not edges_args:
                log.error("No edges exported.\nDoes the graph contain edges?")
                sys.exit(1)
        except Exception as e:
            log.exception("Error during export process: %s", e)
            raise
        finally:
            thread_pool.shutdown()
        self.driver.close()

        self.vertices = nodes_args
        self.edges = edges_args
