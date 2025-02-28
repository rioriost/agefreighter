#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for exporting data from a Neo4j database to CSV files and loading them using AgeFreighter.
"""

import argparse
import asyncio
import concurrent.futures
import logging
import os
import sys
from typing import List, Dict, Any, Set, Union

import aiofiles
from neo4j import GraphDatabase, Result
from agefreighter import Factory

logging.basicConfig(level=logging.INFO)


class Neo4jExporter:
    """
    Class responsible for exporting nodes and edges from a Neo4j database into CSV files.
    """

    def __init__(
        self,
        output_dir: str,
        uri: str,
        user: str,
        password: str,
        database: str,
        trial: bool,
        chunk_size: int,
        progress: bool,
        instance: Factory,
        graph_name: str,
    ):
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            logging.info(
                f"Output directory '{self.output_dir}' does not exist. Creating it."
            )
            os.makedirs(self.output_dir)
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.trial = trial
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")
        self.chunk_size = chunk_size
        self.progress = progress
        self.instance = instance
        self.graph_name = graph_name
        self.trial_nodes_by_label: Dict[
            str, List[str]
        ] = {}  # For trial mode: mapping label -> list of node element IDs
        self.id_maps: Dict[str, Dict[str, int]] = {}

        try:
            # Create a driver instance; the Neo4j driver is thread-safe.
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            logging.exception("Error connecting to Neo4j: %s", e)
            raise

    async def __aenter__(self):
        try:
            dsn = os.environ["PG_CONNECTION_STRING"]
        except KeyError:
            raise ValueError("""PG_CONNECTION_STRING environment variable not set.
            macOS/Linux: export PG_CONNECTION_STRING='host=**..postgres.database.azure.com port=5432 dbname=...'
            Windows:     set PG_CONNECTION_STRING='host=**..postgres.database.azure.com port=5432 dbname=...'
            """)
        # Connect to the AgeFreighter instance using the PG_CONNECTION_STRING environment variable.
        await self.instance.connect(
            dsn=dsn,
            max_connections=64,
            min_connections=4,
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.driver.close()

    def get_labels(self) -> List[str]:
        """
        Retrieve all node labels from the Neo4j database and add "NO_LABEL" if there are nodes without any labels.
        """
        labels = []
        try:
            with self.driver.session() as session:
                query = "CALL db.labels() YIELD label RETURN label;"
                result: Result = session.run(query)
                labels = [record["label"] for record in result]
                # Check for nodes with no label:
                query_no_label = (
                    "MATCH (n) WHERE size(labels(n)) = 0 RETURN COUNT(n) AS cnt"
                )
                result_no_label: Result = session.run(query_no_label)
                record = result_no_label.single()
                if record and int(record["cnt"]) > 0:
                    labels.append("NO_LABEL")
            return labels
        except Exception as e:
            logging.exception("Error retrieving labels: %s", e)
            return []

    def get_relationship_types(self) -> List[str]:
        """
        Retrieve all relationship types from the Neo4j database.
        """
        try:
            with self.driver.session() as session:
                query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType;"
                result: Result = session.run(query)
                types = [record["relationshipType"] for record in result]
            return types
        except Exception as e:
            logging.exception("Error retrieving relationship types: %s", e)
            return []

    def _count_nodes(self, label: Union[str, List[str]]) -> int:
        """
        Count the number of nodes for a given label (or list of labels).
        If label is "NO_LABEL", use a query to count nodes with no labels.
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
                elif isinstance(label, list):
                    labels_str = ", ".join(f'"{l}"' for l in label)
                    query = (
                        f"MATCH (n) WHERE NONE(label IN labels(n) WHERE label IN [{labels_str}]) "
                        f"RETURN COUNT(n) AS cnt"
                    )
                result: Result = session.run(query)
                record = result.single()
                cnt = int(record["cnt"]) if record else 0
            return cnt
        except Exception as e:
            logging.exception("Error counting nodes for label %s: %s", label, e)
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
                cnt = int(record["cnt"]) if record else 0
            return cnt
        except Exception as e:
            logging.exception(
                "Error counting edges for relationship type '%s': %s", rel_type, e
            )
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
            logging.exception("Error fetching nodes chunk: %s", e)
            return []

    def _fetch_nodes_by_ids_chunk(
        self, label: str, node_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch nodes for a given label whose element IDs are in node_ids.
        If label is "NO_LABEL", fetch nodes that have no labels.
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
            logging.exception(
                "Error fetching nodes by IDs for label '%s': %s", label, e
            )
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
                data: List[Dict[str, Any]] = []
                for record in result:
                    # If a node has no label, assign "NO_LABEL"
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
            logging.exception(
                "Error fetching edge chunk for relationship type '%s' (skip %d): %s",
                rel_type,
                skip,
                e,
            )
            return []

    @staticmethod
    def extract_unique_keys(data: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract a set of unique property keys from the provided data.
        """
        unique_keys = set()
        for row in data:
            properties = row.get("properties", {})
            unique_keys.update(properties.keys())
        return unique_keys

    async def export_nodes(
        self, pool: concurrent.futures.Executor
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export nodes from Neo4j to CSV files and return a dictionary of vertex import arguments.
        """
        loop = asyncio.get_running_loop()
        vertex_args: Dict[str, Dict[str, Any]] = {}
        # In trial mode, combine node element IDs from different relationships
        if self.trial_nodes_by_label:
            combined_nodes: Dict[str, List[str]] = {}
            for d in self.trial_nodes_by_label.values():
                for label, ids in d.items():
                    combined_nodes.setdefault(label, []).extend(ids)
            labels = list(combined_nodes.keys())
        else:
            labels = self.get_labels()

        for label in labels:
            try:
                # Create label type in AgeFreighter for the vertex
                await self.instance.createLabelType(label_type="vertex", value=label)
                first_id = await self.instance.getFirstId(self.graph_name, label)
                if self.trial_nodes_by_label:
                    count = len(combined_nodes.get(label, []))
                else:
                    count = self._count_nodes(label)
                logging.info(f"Exporting {count} nodes for label '{label}'")
                if self.trial:
                    tasks = [
                        loop.run_in_executor(
                            pool,
                            self._fetch_nodes_by_ids_chunk,
                            label,
                            combined_nodes[label][skip : skip + self.chunk_size],
                        )
                        for skip in range(0, count, self.chunk_size)
                    ]
                else:
                    # For nodes with no label, use a special query
                    query_template = (
                        f"MATCH (n) WHERE size(labels(n)) = 0 SKIP {{skip}} LIMIT {{limit}} RETURN n"
                        if label == "NO_LABEL"
                        else f"MATCH (n:{label}) SKIP {{skip}} LIMIT {{limit}} RETURN n"
                    )
                    tasks = [
                        loop.run_in_executor(
                            pool,
                            self._fetch_nodes_chunk,
                            query_template.format(skip=skip, limit=self.chunk_size),
                        )
                        for skip in range(0, count, self.chunk_size)
                    ]
                chunks = await asyncio.gather(*tasks)
                # Flatten fetched chunks and assign new IDs starting from first_id
                all_data = [
                    {"id": idx + first_id, "properties": item}
                    for sublist in chunks
                    for idx, item in enumerate(sublist)
                ]
                # Build an ID mapping for use in exporting edges
                self.id_maps[label] = {
                    item["properties"].get("_elementid"): item["id"]
                    for item in all_data
                }
                await self.write_csv(label, "v", all_data)
                # Prepare file path for AgeFreighter import (escape backslashes if needed)
                file_path = os.path.join(
                    self.output_dir, f"{label.lower()}.csv"
                ).replace("\\", "\\\\")
                vertex_args[label] = {
                    "csv_path": file_path,
                    "original_id": "_elementid",
                }
            except Exception as e:
                logging.exception("Error exporting nodes for label '%s': %s", label, e)
        return vertex_args

    async def export_edges(
        self, pool: concurrent.futures.Executor
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export edges from Neo4j to CSV files and return a dictionary of edge import arguments.
        """
        loop = asyncio.get_running_loop()
        edge_args: Dict[str, Dict[str, Any]] = {}
        types = self.get_relationship_types()
        for rel_type in types:
            try:
                await self.instance.createLabelType(label_type="edge", value=rel_type)
                first_id = await self.instance.getFirstId(self.graph_name, rel_type)
                count = self._count_edges(rel_type)
                if self.trial:
                    count = min(count, 100)
                logging.info(
                    f"Exporting {count} edges for relationship type '{rel_type}'"
                )
                tasks = [
                    loop.run_in_executor(
                        pool,
                        self._fetch_edge_chunk,
                        rel_type,
                        skip,
                        self.chunk_size,
                    )
                    for skip in range(0, count, self.chunk_size)
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
                await self.write_csv(rel_type, "e", all_data)
                file_path = os.path.join(
                    self.output_dir, f"{rel_type.lower()}.csv"
                ).replace("\\", "\\\\")
                edge_args[rel_type] = {
                    "csv_path": file_path,
                    "original_id": "_elementid",
                }
            except Exception as e:
                logging.exception(
                    "Error exporting edges for relationship type '%s': %s", rel_type, e
                )
        return edge_args

    async def list_nodes(self) -> None:
        """
        In trial mode, list nodes for each relationship type.
        """
        pool = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_running_loop()
        types = self.get_relationship_types()
        for rel_type in types:
            try:
                count = min(self._count_edges(rel_type), 100)
                logging.info(f"Listing nodes for relationship type '{rel_type}'")
                tasks = [
                    loop.run_in_executor(
                        pool,
                        self._fetch_edge_chunk,
                        rel_type,
                        skip,
                        self.chunk_size,
                    )
                    for skip in range(0, count, self.chunk_size)
                ]
                chunks = await asyncio.gather(*tasks)
                all_data = [item for sublist in chunks for item in sublist]

                nodes_by_label: Dict[str, List[str]] = {}
                for record in all_data:
                    nodes_by_label.setdefault(record["start_vertex_type"], []).append(
                        record["start_id"]
                    )
                    nodes_by_label.setdefault(record["end_vertex_type"], []).append(
                        record["end_id"]
                    )
                self.trial_nodes_by_label[rel_type] = nodes_by_label
            except Exception as e:
                logging.exception(
                    "Error listing nodes for relationship type '%s': %s", rel_type, e
                )
        pool.shutdown()

    async def write_csv(
        self, label: str, kind: str, data: List[Dict[str, Any]]
    ) -> None:
        """
        Write exported data to a CSV file.
        'kind' specifies:
          - 'v' for vertex (node) export
          - 'e' for edge export
        """
        if not data:
            logging.info(f"No data to write for '{label}'.")
            return

        def format_kv(key: str, value: Any) -> str:
            # Escape tab characters for compatibility with agtype
            wk = str(value).replace("\t", "\\t")
            return f'""{key}"": ""{wk}""'

        file_path = os.path.join(self.output_dir, f"{label.lower()}.csv")
        headers = self.extract_unique_keys(data)
        try:
            async with aiofiles.open(file_path, "w", encoding="utf-8", newline="") as f:
                if kind == "e":  # Edge export
                    for row in data:
                        line = ", ".join(
                            format_kv(h, row["properties"].get(h, ""))
                            for h in headers
                            if row["properties"].get(h, "")
                        )
                        await f.write(
                            f'{row["id"]},{row["start_id"]},{row["end_id"]},"{{{line}}}"\n'
                        )
                elif kind == "v":  # Node export
                    for row in data:
                        line = ", ".join(
                            format_kv(h, row["properties"].get(h, ""))
                            for h in headers
                            if row["properties"].get(h, "")
                        )
                        await f.write(f'{row["id"]},"{{{line}}}"\n')
                else:
                    raise ValueError(f"Unsupported kind: {kind}")
            logging.info(f"Exported {len(data)} records to {file_path}")
        except Exception as e:
            logging.exception("Error writing CSV for label '%s': %s", label, e)

    async def export(self) -> Dict[str, Dict[str, Any]]:
        """
        Main export function to process both nodes and edges.
        Returns a dictionary containing import arguments for both vertices and edges.
        """
        # Use a ThreadPoolExecutor for I/O-bound tasks (avoiding pickling issues)
        pool = concurrent.futures.ThreadPoolExecutor()
        try:
            await self.instance.setUpGraph(
                graph_name=self.graph_name, create_graph=True
            )
            if self.trial:
                await self.list_nodes()
            nodes_args = await self.export_nodes(pool)
            edges_args = await self.export_edges(pool)
        except Exception as e:
            logging.exception("Error during export process: %s", e)
            raise
        finally:
            pool.shutdown()
        self.driver.close()
        return {"nodes": nodes_args, "edges": edges_args}


class AgeLoader:
    """
    Class responsible for loading graph data into AgeFreighter.
    """

    def __init__(
        self,
        graph_name: str,
        vertices: Dict[str, Any],
        edges: Dict[str, Any],
        chunk_size: int,
        progress: bool,
        instance: Factory,
    ):
        self.graph_name = graph_name
        self.vertices = vertices
        self.edges = edges
        self.chunk_size = chunk_size
        self.progress = progress
        self.instance = instance

    async def load(self) -> None:
        """
        Initiate the data loading process.
        """
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        await self._load()

    async def _load(self) -> None:
        """
        Load the graph data into AgeFreighter using the provided configuration.
        """
        try:
            await self.instance.copy(
                graph_name=self.graph_name,
                vertices=self.vertices,
                edges=self.edges,
                chunk_size=self.chunk_size,
                progress=self.progress,
            )
        except Exception as e:
            logging.exception("Error loading data into AgeFreighter: %s", e)
            raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export data from Neo4j to CSV and load into Apache AGE."
    )
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--uri",
        type=str,
        default="bolt://localhost:7687",
        help="The URI of the Neo4j database",
    )
    parser.add_argument(
        "--user", type=str, default="neo4j", help="The username of the Neo4j database"
    )
    parser.add_argument(
        "--password",
        type=str,
        default="neo4jpass",
        help="The password of the Neo4j database",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="neo4j",
        help="The database of the Neo4j database",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        default=False,
        help="Extract only 100 edges per relationship type",
    )
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size")
    parser.add_argument(
        "--progress", action="store_true", default=True, help="Show progress"
    )
    parser.add_argument(
        "--graphname",
        type=str,
        default="FROM_NEO4J",
        help="Name of the graph to be embedded in 'importer.py'",
    )
    return parser.parse_args()


async def main() -> None:
    """
    Main function to export data from Neo4j and load it using AgeFreighter.
    """
    args = parse_arguments()
    # Adjust chunk size based on trial mode
    args.chunk_size = 100 if args.trial else 10000

    # Create an AgeFreighter instance
    af = Factory.create_instance("AgeFreighter")

    try:
        async with Neo4jExporter(
            output_dir=args.output_dir,
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            trial=args.trial,
            chunk_size=args.chunk_size,
            progress=args.progress,
            graph_name=args.graphname,
            instance=af,
        ) as exporter:
            result = await exporter.export()
    except Exception as e:
        logging.exception("An error occurred during export: %s", e)
        return

    try:
        al = AgeLoader(
            graph_name=args.graphname,
            vertices=result["nodes"],
            edges=result["edges"],
            chunk_size=args.chunk_size,
            progress=args.progress,
            instance=af,
        )
        await al.load()
    except Exception as e:
        logging.exception("An error occurred during loading: %s", e)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.exception("Unhandled exception in main: %s", e)
