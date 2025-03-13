#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for exporting PostgreSQL data for AGE COPY import using AgeFreighter.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, cast, Tuple

from psycopg_pool import AsyncConnectionPool
from psycopg.sql import SQL, Identifier
from psycopg.rows import namedtuple_row

from .agefreighter import AgeFreighter

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ConfigManager:
    """
    Loads and validates a JSON configuration file.
    """

    def __init__(
        self,
        config_path: str,
        src_con_pool: AsyncConnectionPool,
        log_level: int = logging.INFO,
    ) -> None:
        log.setLevel(log_level)
        self.config_path: str = config_path
        self.src_con_pool: AsyncConnectionPool = src_con_pool
        self.config_json: Dict[str, Any] = {}
        self.parse_result: Optional[str] = None

    def require_key(self, config: Dict[str, Any], key: str, context: str = "") -> Any:
        if key not in config:
            ctx = f" ({context})" if context else ""
            raise ValueError(
                f"Missing '{key}' key in config file {self.config_path}{ctx}"
            )
        return config[key]

    async def load_config(self) -> Dict[str, Any]:
        tables: List[str] = []

        # Load JSON configuration
        try:
            with open(os.path.abspath(self.config_path), "r", encoding="utf-8") as f:
                config_json = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in config file {self.config_path}")

        edge_config = self.require_key(config_json, "edge", "root level")

        if isinstance(edge_config, dict):
            self.parse_result = "Config has single edge and multiple nodes"
            start_vertex = self.require_key(edge_config, "start_vertex", "edge config")
            end_vertex = self.require_key(edge_config, "end_vertex", "edge config")
            tables.extend(
                [
                    self.require_key(edge_config, "table", "edge config"),
                    self.require_key(start_vertex, "table", "start_vertex config"),
                    self.require_key(end_vertex, "table", "end_vertex config"),
                ]
            )
            for vertex in (start_vertex, end_vertex):
                self.require_key(vertex, "id", "vertex config")
                self.require_key(vertex, "label", "vertex config")
                self.require_key(vertex, "props", "vertex config")
            self.require_key(edge_config, "type", "edge config")
            self.require_key(edge_config, "props", "edge config")
            self.require_key(edge_config, "start_vertex", "edge config")
            self.require_key(edge_config, "end_vertex", "edge config")
        elif isinstance(edge_config, list):
            for ec in edge_config:
                self.require_key(ec, "type", "edge config")
                self.require_key(ec, "props", "edge config")
                if "vertex" in ec:
                    self.parse_result = "Config has multiple edges and single node"
                    tables.append(self.require_key(ec, "table", "edge config"))
                    vertex = self.require_key(ec, "vertex", "edge config")
                    self.require_key(vertex, "id", "vertex config")
                    self.require_key(vertex, "label", "vertex config")
                    self.require_key(vertex, "props", "vertex config")
                else:
                    self.parse_result = "Config has multiple edges and multiple nodes"
                    start_vertex = self.require_key(ec, "start_vertex", "edge config")
                    end_vertex = self.require_key(ec, "end_vertex", "edge config")
                    tables.extend(
                        [
                            self.require_key(ec, "table", "edge config"),
                            self.require_key(
                                start_vertex, "table", "start_vertex config"
                            ),
                            self.require_key(end_vertex, "table", "end_vertex config"),
                        ]
                    )
                    for vertex in (start_vertex, end_vertex):
                        self.require_key(vertex, "id", "vertex config")
                        self.require_key(vertex, "label", "vertex config")
                        self.require_key(vertex, "props", "vertex config")
        else:
            raise ValueError(f"Invalid configuration structure in {self.config_path}")

        if not tables:
            raise ValueError("No tables provided")

        async with self.src_con_pool.connection() as con:
            async with con.cursor() as cur:
                for table in tables:
                    await cur.execute(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                        (table,),
                    )
                    exists = await cur.fetchone()
                    if exists is None or not exists[0]:
                        raise ValueError(f"Table {table} does not exist")
        return config_json


class PGSQLExporter(AgeFreighter):
    """
    Exports nodes and edges from PostgreSQL tables into new CSV files for AGE COPY import.
    """

    def __init__(
        self,
        dsn: str,
        min_connections: int,
        max_connections: int,
        src_dsn: str,
        config: str,
        trial: bool,
        save_temps: bool,
        progress: bool,
        graph_name: str,
        chunk_size: int,
        log_level: int = logging.INFO,
    ) -> None:
        super().__init__(
            dsn=dsn,
            min_connections=min_connections,
            max_connections=max_connections,
            save_temps=save_temps,
            progress=progress,
            chunk_size=chunk_size,
            log_level=log_level,
        )
        log.setLevel(log_level)
        self.src_dsn = src_dsn
        self.config: str = config
        self.config_json: Dict[str, Any] = {}
        self.graph_name: str = graph_name
        self.trial: bool = trial
        self.trial_nodes_by_label: Dict[str, Dict[str, List[str]]] = {}
        self.id_maps: Dict[str, Dict[Any, int]] = {}
        self.no_of_edges_trial = 100
        self.log_level = log_level
        self.vertex_configs: Dict[str, Dict[str, Any]] = {}

    async def __aenter__(self) -> "PGSQLExporter":
        await super().__aenter__()
        log.debug("Entering async context for PGSQLExporter.")
        self.src_con_pool = await self.connect(
            dsn=self.src_dsn,
            max_connections=self.max_connections,
            min_connections=self.min_connections,
            max_attempts=self.max_attempts,
            retry_delay=self.retry_delay,
            **self.extra_kwargs,
        )
        config_manager = ConfigManager(
            config_path=self.config,
            src_con_pool=self.src_con_pool,
            log_level=self.log_level,
        )
        self.config_json = await config_manager.load_config()
        self._build_vertex_configs()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc_type:
            log.error("Exception in PGSQLFreighter context: %s", exc)
        if self.src_con_pool:
            try:
                await self.src_con_pool.close()
            except Exception as e:
                log.error("Error closing source connection pool: %s", e)
        log.debug("Exiting PGSQLFreighter context.")
        await super().__aexit__(exc_type, exc, tb)

    @staticmethod
    def _clean_row(row: Any) -> Dict[str, Any]:
        return {
            key.strip('"'): value.strip('"') if isinstance(value, str) else value
            for key, value in row._asdict().items()
        }

    def get_labels(self) -> List[str]:
        labels = set()
        edge_config = self.config_json["edge"]
        if isinstance(edge_config, dict):
            if "vertex" in edge_config:
                labels.add(edge_config["vertex"]["label"])
            else:
                labels.add(edge_config["start_vertex"]["label"])
                labels.add(edge_config["end_vertex"]["label"])
        elif isinstance(edge_config, list):
            for ec in edge_config:
                if "vertex" in ec:
                    labels.add(ec["vertex"]["label"])
                else:
                    labels.add(ec["start_vertex"]["label"])
                    labels.add(ec["end_vertex"]["label"])
        return list(labels)

    def get_ids(self) -> Dict[str, List[str]]:
        edge_config = self.config_json["edge"]
        if isinstance(edge_config, dict):
            return {
                "start_id": [edge_config["start_id"]],
                "end_id": [edge_config["end_id"]],
            }
        elif isinstance(edge_config, list):
            return {
                "start_id": [ec["start_id"] for ec in edge_config],
                "end_id": [ec["end_id"] for ec in edge_config],
            }
        return {}

    def get_relationship_types(self) -> List[str]:
        edge_config = self.config_json["edge"]
        types: List[str] = []
        if isinstance(edge_config, dict):
            edge_type = edge_config.get("type")
            if isinstance(edge_type, str):
                types.append(edge_type)
        elif isinstance(edge_config, list):
            for ec in edge_config:
                edge_type = ec.get("type")
                if isinstance(edge_type, str):
                    types.append(edge_type)
        return list(dict.fromkeys(types))

    def get_vertex_labels(self, rel_type: str) -> Tuple[str, Optional[str]]:
        edge_config = self.config_json["edge"]
        if isinstance(edge_config, dict):
            if edge_config.get("type") != rel_type:
                raise ValueError(
                    f"No vertex labels found for relationship type {rel_type}"
                )
            return (
                edge_config["start_vertex"]["label"],
                edge_config["end_vertex"]["label"],
            )
        elif isinstance(edge_config, list):
            for ec in edge_config:
                if ec.get("type") == rel_type:
                    if "vertex" in ec:
                        return (ec["vertex"]["label"], None)
                    return (ec["start_vertex"]["label"], ec["end_vertex"]["label"])
        raise ValueError(f"No vertex labels found for relationship type {rel_type}")

    async def _count_rows(self, table_name: str) -> int:
        assert self.src_con_pool is not None, (
            "Source connection pool is not initialized"
        )
        async with self.src_con_pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                await cur.execute(
                    SQL("SELECT COUNT(*) FROM {table_name}").format(
                        table_name=Identifier(table_name)
                    )
                )
                result = await cur.fetchone()
                if result is None:
                    return 0
                # Assuming result is a tuple, return its first element.
                return result[0]

    async def _fetch_edges_chunk_table(
        self, table_name: str, skip: int, chunk_size: int
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        assert self.src_con_pool is not None, (
            "Source connection pool is not initialized"
        )
        async with self.src_con_pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                await cur.execute(
                    SQL(
                        "SELECT * FROM {table_name} LIMIT {limit} OFFSET {skip}"
                    ).format(
                        table_name=Identifier(table_name), limit=chunk_size, skip=skip
                    )
                )
                rows = await cur.fetchall()
                for idx, row in enumerate(rows):
                    cleaned = self._clean_row(row)
                    cleaned["_elementid"] = idx + 1  # temporary element id
                    results.append(cleaned)
        return results

    async def _fetch_nodes_chunk_table(
        self, table_name: str, skip: int, chunk_size: int, id_column: str
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        assert self.src_con_pool is not None, (
            "Source connection pool is not initialized"
        )
        async with self.src_con_pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                await cur.execute(
                    SQL(
                        "SELECT * FROM {table_name} LIMIT {limit} OFFSET {skip}"
                    ).format(
                        table_name=Identifier(table_name), limit=chunk_size, skip=skip
                    )
                )
                rows = await cur.fetchall()
                for row in rows:
                    cleaned = self._clean_row(row)
                    cleaned["_elementid"] = getattr(row, id_column)
                    results.append(cleaned)
        return results

    async def _fetch_nodes_by_ids_chunk_table(
        self, table_name: str, node_ids: List[str], id_column: str
    ) -> List[Dict[str, Any]]:
        # If no node IDs, return immediately to avoid syntax errors.
        if not node_ids:
            return []

        results: List[Dict[str, Any]] = []
        assert self.src_con_pool is not None, (
            "Source connection pool is not initialized"
        )
        async with self.src_con_pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                values = ",".join([str(nid) for nid in node_ids])
                await cur.execute(
                    'SELECT * FROM "{table_name}" WHERE {id_column} IN ({values})'.format(
                        table_name=table_name, id_column=id_column, values=values
                    )
                )
                rows = await cur.fetchall()
                for i, row in enumerate(rows):
                    cleaned = self._clean_row(row)
                    cleaned["_elementid"] = getattr(row, id_column)
                    results.append(cleaned)
        return results

    def get_edge_table_name(self, rel_type: str) -> str:
        edge_config = self.config_json["edge"]
        if isinstance(edge_config, dict):
            if edge_config.get("type") == rel_type:
                return edge_config["table"]
        elif isinstance(edge_config, list):
            for ec in edge_config:
                if ec.get("type") == rel_type:
                    return ec["table"]
        raise ValueError(f"No table found for relationship type {rel_type}")

    def _build_vertex_configs(self) -> None:
        vertex_configs: Dict[str, Any] = {}
        edge_config = self.config_json["edge"]

        if isinstance(edge_config, dict):
            # Single edge configuration (Pattern 1 and Pattern 2)
            if "start_vertex" in edge_config and "end_vertex" in edge_config:
                start_vc = edge_config["start_vertex"]
                end_vc = edge_config["end_vertex"]

                if "table" not in start_vc:
                    start_vc["table"] = edge_config["table"]
                if "table" not in end_vc:
                    end_vc["table"] = edge_config["table"]

                if start_vc["label"] == end_vc["label"]:
                    # Pattern 1: both vertices share the same label
                    vertex_configs[start_vc["label"]] = [start_vc, end_vc]
                else:
                    # Pattern 2: different labels, store as single dict per label
                    vertex_configs[start_vc["label"]] = start_vc
                    vertex_configs[end_vc["label"]] = end_vc
            elif "vertex" in edge_config:
                # In case there is a "vertex" key at the top level
                vc = edge_config["vertex"]
                if "table" not in vc:
                    vc["table"] = edge_config["table"]
                vertex_configs[vc["label"]] = vc
            else:
                raise ValueError("Invalid edge config structure")
        elif isinstance(edge_config, list):
            # Multiple edge configurations (Pattern 3 and Pattern 4)
            for ec in edge_config:
                if "vertex" in ec:
                    # Pattern 3: each edge has a "vertex" key
                    vc = ec["vertex"]
                    if "table" not in vc:
                        vc["table"] = ec["table"]
                    label = vc["label"]
                    if label not in vertex_configs:
                        vertex_configs[label] = vc
                    else:
                        existing = vertex_configs[label]
                        if isinstance(existing, list):
                            if vc not in existing:
                                existing.append(vc)
                        else:
                            if existing != vc:
                                vertex_configs[label] = [existing, vc]
                else:
                    # Pattern 4: each edge has "start_vertex" and "end_vertex"
                    for key in ["start_vertex", "end_vertex"]:
                        vc = ec[key]
                        if "table" not in vc:
                            vc["table"] = ec["table"]
                        label = vc["label"]
                        if label not in vertex_configs:
                            vertex_configs[label] = vc
                        else:
                            existing = vertex_configs[label]
                            if isinstance(existing, list):
                                if vc not in existing:
                                    existing.append(vc)
                            else:
                                if existing != vc:
                                    vertex_configs[label] = [existing, vc]
        else:
            raise ValueError("Invalid edge config type")

        self.vertex_configs = vertex_configs

    async def export_nodes(self) -> Dict[str, Dict[str, Any]]:
        """
        Export nodes from PostgreSQL to new CSV files and return mapping for AGE import.
        """
        vertex_args: Dict[str, Dict[str, Any]] = {}

        # Determine labels based on trial mode or full export
        if self.trial:
            # Wrap keys in list to ensure a list of strings.
            labels = list(list(self.trial_nodes_by_label.values())[0].keys())
        else:
            labels = list(self.vertex_configs.keys())

        for label in labels:
            try:
                await self.create_label_type(label_type="vertex", value=label)
                first_id = await self.get_first_id(self.graph_name, label)
                print(self.vertex_configs[label])
                if isinstance(self.vertex_configs[label], list):
                    vertex_config_list = cast(
                        List[Dict[str, Any]], self.vertex_configs[label]
                    )
                    tables = [vc["table"] for vc in vertex_config_list]
                    id_columns = [vc["id"] for vc in vertex_config_list]
                    properties = [vc["props"] for vc in vertex_config_list]
                else:
                    tables = [self.vertex_configs[label]["table"]]
                    id_columns = [self.vertex_configs[label]["id"]]
                    properties = [self.vertex_configs[label]["props"]]

                nodes: List[Dict[str, Any]] = []
                if self.trial:
                    for node_id_list in list(self.trial_nodes_by_label.values()):
                        node_ids = node_id_list[label]
                        for table, id_column in zip(tables, id_columns):
                            nodes.extend(
                                await self._fetch_nodes_by_ids_chunk_table(
                                    table, node_ids, id_column
                                )
                            )
                else:
                    for table, id_column in zip(tables, id_columns):
                        count = await self._count_rows(table)
                        tasks = [
                            self._fetch_nodes_chunk_table(
                                table, skip, int(self.chunk_size), id_column
                            )
                            for skip in range(0, count, int(self.chunk_size))
                        ]
                        for chunk in await asyncio.gather(*tasks):
                            nodes.extend(chunk)

                # Deduplicate nodes by their original ID.
                unique_nodes = {node["_elementid"]: node for node in nodes}
                nodes = list(unique_nodes.values())

                all_data = [
                    {
                        "id": idx + first_id,
                        "properties": {
                            k: v
                            for k, v in node.items()
                            if k in properties or k == "_elementid"
                        },
                    }
                    for idx, node in enumerate(nodes)
                ]

                self.id_maps[label] = {
                    cast(Dict[str, Any], item["properties"])["_elementid"]: cast(
                        int, item["id"]
                    )
                    for item in all_data
                    if "_elementid" in cast(Dict[str, Any], item["properties"])
                }

                file_path = await self.write_csv(label, "v", all_data)
                vertex_args[label] = {
                    "csv_path": file_path.replace("\\", "\\\\"),
                    "original_id": "_elementid",
                    "next_val": str(len(all_data)),
                }
            except Exception as exc:
                log.exception("Error exporting nodes for label '%s': %s", label, exc)
        return vertex_args

    async def export_edges(self) -> Dict[str, Dict[str, Any]]:
        """
        Export edges from PostgreSQL to new CSV files and return mapping for AGE import.
        """
        edge_args: Dict[str, Dict[str, Any]] = {}
        rel_types = self.get_relationship_types()

        for rel_type in rel_types:
            try:
                edge_table_name = self.get_edge_table_name(rel_type)
                await self.create_label_type(label_type="edge", value=rel_type)
                first_id = await self.get_first_id(self.graph_name, rel_type)
                count = await self._count_rows(edge_table_name)
                if self.trial:
                    count = min(count, 100)
                    limit = min(count, int(self.chunk_size))
                else:
                    limit = int(self.chunk_size)
                log.info(
                    "Exporting %d edges for relationship type '%s'.", count, rel_type
                )
                tasks = [
                    self._fetch_edges_chunk_table(edge_table_name, skip, limit)
                    for skip in range(0, count, limit)
                ]
                chunks = await asyncio.gather(*tasks)
                all_data: List[Dict[str, Any]] = []
                start_vertex_label, end_vertex_label = self.get_vertex_labels(rel_type)
                ids = self.get_ids()

                for sublist in chunks:
                    for idx, item in enumerate(sublist):
                        # Resolve start_id using candidate keys
                        new_start_id = None
                        for candidate in ids["start_id"]:
                            try:
                                new_start_id = self.id_maps[start_vertex_label][
                                    item[candidate]
                                ]
                                break
                            except KeyError:
                                continue
                        if new_start_id is None:
                            log.warning(
                                "Could not resolve start_id for record: %s", item
                            )
                            continue

                        # Resolve end_id using candidate keys
                        new_end_id = None
                        if end_vertex_label:
                            for candidate in ids["end_id"]:
                                try:
                                    new_end_id = self.id_maps[end_vertex_label][
                                        item[candidate]
                                    ]
                                    break
                                except KeyError:
                                    continue
                        else:
                            for candidate in ids["end_id"]:
                                try:
                                    new_end_id = self.id_maps[start_vertex_label][
                                        item[candidate]
                                    ]
                                    break
                                except KeyError:
                                    continue
                        if new_end_id is None:
                            log.warning("Could not resolve end_id for record: %s", item)
                            continue

                        edge_data = {
                            "id": idx + first_id,
                            "start_id": new_start_id,
                            "end_id": new_end_id,
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
                        all_data.append(edge_data)

                file_path = await self.write_csv(rel_type, "e", all_data)
                edge_args[rel_type] = {
                    "csv_path": file_path.replace("\\", "\\\\"),
                    "original_id": "_elementid",
                    "next_val": str(len(all_data)),
                }
            except Exception as exc:
                log.exception("Error exporting edges for '%s': %s", rel_type, exc)
        return edge_args

    async def list_nodes(self) -> None:
        """
        In trial mode, list nodes per relationship type by sampling edge rows.
        """
        for rel_type in self.get_relationship_types():
            try:
                edge_table_name = self.get_edge_table_name(rel_type)
                count = min(
                    await self._count_rows(edge_table_name),
                    self.no_of_edges_trial,
                )
                limit = min(int(self.chunk_size), count)
                log.info("Listing nodes for relationship type '%s'.", rel_type)
                tasks = [
                    self._fetch_edges_chunk_table(edge_table_name, skip, limit)
                    for skip in range(0, count, limit)
                ]
                chunks = await asyncio.gather(*tasks)
                all_data = [item for sublist in chunks for item in sublist]
                id_dict = self.get_ids()
                nodes_by_label: Dict[str, List[str]] = {}
                for node_label in self.vertex_configs.keys():
                    nodes_by_label[node_label] = []
                    for key_list in id_dict.values():
                        for candidate in key_list:
                            for record in all_data:
                                v = record.get(candidate, None)
                                if v:
                                    nodes_by_label[node_label].append(v)
                self.trial_nodes_by_label[rel_type] = nodes_by_label
            except Exception as exc:
                log.exception("Error listing nodes for '%s': %s", rel_type, exc)

    async def export(self) -> None:
        """
        Main export function to process both nodes and edges.
        """
        try:
            await self.set_up_graph(graph_name=self.graph_name, create_graph=True)
            if self.trial:
                await self.list_nodes()
            nodes_args = await self.export_nodes()
            if not nodes_args:
                log.error(
                    "No nodes exported.\nDoes the PostgreSQL contain data for nodes?"
                )
                sys.exit(1)
            edges_args = await self.export_edges()
            if not edges_args:
                log.error(
                    "No edges exported.\nDoes the PostgreSQL contain data for edges?"
                )
                sys.exit(1)
        except Exception as exc:
            log.exception("Error during export process: %s", exc)
            raise

        self.vertices = nodes_args
        self.edges = edges_args
