#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for exporting CSV data for AGE COPY import using AgeFreighter.
"""

import asyncio
import concurrent.futures
import csv
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, cast

from .agefreighter import AgeFreighter

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ConfigManager:
    """
    Loads and validates a JSON configuration file.
    """

    def __init__(self, config_path: str, log_level: int = logging.INFO) -> None:
        """
        :param config_path: Path to the configuration file.
        """
        log.setLevel(log_level)
        self.config_path: str = config_path
        self.config_json: Dict[str, Any] = {}
        self.parse_result: Optional[str] = None

    def require_key(self, config: Dict[str, Any], key: str, context: str = "") -> Any:
        """
        Ensure that the specified key exists in the config dictionary.

        :param config: Dictionary to validate.
        :param key: Required key.
        :param context: Additional context for error message.
        :return: The value associated with the key.
        :raises ValueError: If key is missing.
        """
        if key not in config:
            ctx = f" ({context})" if context else ""
            raise ValueError(
                f"Missing '{key}' key in config file {self.config_path}{ctx}"
            )
        return config[key]

    def load_config(self) -> Dict[str, Any]:
        """
        Load the JSON configuration file, validate required keys,
        and ensure all CSV paths exist.

        :return: Validated configuration dictionary.
        :raises ValueError: If JSON is invalid, keys are missing, or CSV files do not exist.
        """
        csv_paths: List[str] = []

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
            csv_paths.append(self.require_key(edge_config, "csv_path", "edge config"))
            csv_paths.append(
                self.require_key(start_vertex, "csv_path", "start_vertex config")
            )
            csv_paths.append(
                self.require_key(end_vertex, "csv_path", "end_vertex config")
            )
            for vertex in (start_vertex, end_vertex):
                self.require_key(vertex, "id", "vertex config")
                self.require_key(vertex, "label", "vertex config")
                self.require_key(vertex, "props", "vertex config")
            self.require_key(edge_config, "type", "edge config")
            self.require_key(edge_config, "props", "edge config")
        elif isinstance(edge_config, list):
            for ec in edge_config:
                self.require_key(ec, "type", "edge config")
                self.require_key(ec, "props", "edge config")
                if "vertex" in ec:
                    self.parse_result = "Config has multiple edges and single node"
                    csv_paths.append(self.require_key(ec, "csv_path", "edge config"))
                    vertex = self.require_key(ec, "vertex", "edge config")
                    self.require_key(vertex, "id", "vertex config")
                    self.require_key(vertex, "label", "vertex config")
                    self.require_key(vertex, "props", "vertex config")
                else:
                    self.parse_result = "Config has multiple edges and multiple nodes"
                    start_vertex = self.require_key(ec, "start_vertex", "edge config")
                    end_vertex = self.require_key(ec, "end_vertex", "edge config")
                    csv_paths.append(self.require_key(ec, "csv_path", "edge config"))
                    csv_paths.append(
                        self.require_key(
                            start_vertex, "csv_path", "start_vertex config"
                        )
                    )
                    csv_paths.append(
                        self.require_key(end_vertex, "csv_path", "end_vertex config")
                    )
                    for vertex in (start_vertex, end_vertex):
                        self.require_key(vertex, "id", "vertex config")
                        self.require_key(vertex, "label", "vertex config")
                        self.require_key(vertex, "props", "vertex config")
        else:
            raise ValueError(f"Invalid configuration structure in {self.config_path}")

        if not csv_paths:
            raise ValueError("No CSV paths provided")

        for path in csv_paths:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                raise ValueError(f"File {abs_path} does not exist")

        log.info(self.parse_result)
        log.info(self.config_json)
        return config_json


class CSVExporter(AgeFreighter):
    """
    Exports nodes and edges from CSV files into new CSV files for AGE COPY import.
    """

    def __init__(
        self,
        dsn: str,
        min_connections: int,
        max_connections: int,
        config: str,
        trial: bool,
        no_of_edges_trial: int,
        save_temps: bool,
        progress: bool,
        graph_name: str,
        chunk_size: int,
        log_level: int = logging.INFO,
        **kwargs: Any,
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
        self.config: str = config
        self.config_json: Dict[str, Any] = {}
        self.graph_name: str = graph_name
        self.trial: bool = trial
        self.trial_nodes_by_label: Dict[str, Dict[str, List[str]]] = {}
        self.id_maps: Dict[str, Dict[str, int]] = {}
        self.no_of_edges_trial = no_of_edges_trial

        config_manager = ConfigManager(self.config, log_level=log_level)
        self.config_json = config_manager.load_config()

    async def __aenter__(self) -> "CSVExporter":
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await super().__aexit__(exc_type, exc_value, traceback)

    @staticmethod
    def _clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove extraneous double quotes from keys and string values.
        """
        return {
            key.strip('"'): value.strip('"') if isinstance(value, str) else value
            for key, value in row.items()
        }

    def get_labels(self) -> List[str]:
        """
        Retrieve all node labels from the configuration.
        """
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

    def get_relationship_types(self) -> List[str]:
        """
        Retrieve all relationship types from the configuration.
        """
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
        # Deduplicate while preserving order
        return list(dict.fromkeys(types))

    def _count_nodes_csv(self, csv_path: str) -> int:
        """
        Count the number of rows in the CSV file.
        """
        abs_path = os.path.abspath(csv_path)
        with open(abs_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            return sum(1 for _ in reader)

    def _fetch_nodes_chunk_csv(
        self, csv_path: str, skip: int, chunk_size: int, vertex_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Fetch a chunk of nodes from the CSV file, adding an '_elementid'
        field based on vertex_config["id"].
        """
        results: List[Dict[str, Any]] = []
        abs_path = os.path.abspath(csv_path)
        with open(abs_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for i, row in enumerate(reader):
                if i < skip:
                    continue
                if len(results) >= chunk_size:
                    break
                cleaned = self._clean_row(row)
                cleaned["_elementid"] = cleaned[vertex_config["id"]]
                results.append(cleaned)
        return results

    def _fetch_nodes_by_ids_chunk_csv(
        self, csv_path: str, node_ids: List[str], vertex_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Fetch nodes from the CSV whose original ID is in node_ids.
        """
        results: List[Dict[str, Any]] = []
        abs_path = os.path.abspath(csv_path)
        with open(abs_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                cleaned = self._clean_row(row)
                if cleaned[vertex_config["id"]] in node_ids:
                    cleaned["_elementid"] = cleaned[vertex_config["id"]]
                    results.append(cleaned)
        return results

    def get_edge_csv_path(self, rel_type: str) -> str:
        """
        Retrieve the CSV file path for edges of the given relationship type.
        """
        edge_config = self.config_json["edge"]
        if isinstance(edge_config, dict):
            if edge_config.get("type") == rel_type:
                return edge_config["csv_path"]
        elif isinstance(edge_config, list):
            for ec in edge_config:
                if ec.get("type") == rel_type:
                    return ec["csv_path"]
        raise ValueError(f"No CSV path found for relationship type {rel_type}")

    def _count_edges(self, rel_type: str) -> int:
        """
        Count the total number of edge rows in the CSV.
        """
        csv_path = self.get_edge_csv_path(rel_type)
        abs_path = os.path.abspath(csv_path)
        with open(abs_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            return sum(1 for _ in reader)

    def _fetch_edge_chunk_csv(
        self, rel_type: str, skip: int, chunk_size: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch a chunk of edge rows for the given relationship type.
        """
        results: List[Dict[str, Any]] = []
        csv_path = self.get_edge_csv_path(rel_type)
        abs_path = os.path.abspath(csv_path)
        with open(abs_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for i, row in enumerate(reader):
                if i < skip:
                    continue
                if len(results) >= chunk_size:
                    break
                results.append(self._clean_row(row))
        return results

    def _build_vertex_configs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build a mapping from vertex labels to their configurations.
        """
        vertex_configs: Dict[str, List[Dict[str, Any]]] = {}
        edge_config = self.config_json["edge"]
        if isinstance(edge_config, dict):
            for key in ["start_vertex", "end_vertex"]:
                vc = edge_config[key]
                if "csv_path" not in vc:
                    vc["csv_path"] = edge_config["csv_path"]
                vertex_configs.setdefault(vc["label"], []).append(vc)
        elif isinstance(edge_config, list):
            for ec in edge_config:
                if "vertex" in ec:
                    vc = ec["vertex"]
                    if "csv_path" not in vc:
                        vc["csv_path"] = ec["csv_path"]
                    vertex_configs.setdefault(vc["label"], []).append(vc)
                else:
                    for key in ["start_vertex", "end_vertex"]:
                        vc = ec[key]
                        if "csv_path" not in vc:
                            vc["csv_path"] = ec["csv_path"]
                        vertex_configs.setdefault(vc["label"], []).append(vc)
        return vertex_configs

    async def export_nodes(
        self, thread_pool: concurrent.futures.ThreadPoolExecutor
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export nodes from CSV files to new CSV files and return mapping for AGE import.
        """
        loop = asyncio.get_running_loop()
        vertex_args: Dict[str, Dict[str, Any]] = {}
        combined_nodes: Dict[str, List[str]] = {}

        vertex_configs = self._build_vertex_configs()

        # Determine labels based on trial mode or full export
        if self.trial:
            for mapping in self.trial_nodes_by_label.values():
                for label, ids in mapping.items():
                    combined_nodes.setdefault(label, []).extend(ids)
            labels = list(combined_nodes.keys())
        else:
            labels = list(vertex_configs.keys())

        for label in labels:
            try:
                await self.create_label_type(label_type="vertex", value=label)
                first_id = await self.get_first_id(self.graph_name, label)
                nodes: List[Dict[str, Any]] = []

                if self.trial:
                    for vc in vertex_configs[label]:
                        csv_path = vc["csv_path"]
                        node_ids = combined_nodes[label]
                        nodes.extend(
                            self._fetch_nodes_by_ids_chunk_csv(csv_path, node_ids, vc)
                        )
                else:
                    for vc in vertex_configs[label]:
                        csv_path = vc["csv_path"]
                        count = self._count_nodes_csv(csv_path)
                        tasks = [
                            loop.run_in_executor(
                                thread_pool,
                                self._fetch_nodes_chunk_csv,
                                csv_path,
                                skip,
                                int(self.chunk_size),
                                vc,
                            )
                            for skip in range(0, count, int(self.chunk_size))
                        ]
                        chunks = await asyncio.gather(*tasks)
                        for chunk in chunks:
                            nodes.extend(chunk)

                # Deduplicate nodes by their original ID.
                unique_nodes = {node["_elementid"]: node for node in nodes}
                nodes = list(unique_nodes.values())

                all_data = [
                    {"id": idx + first_id, "properties": node}
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
                # Escape backslashes if needed.
                vertex_args[label] = {
                    "csv_path": file_path.replace("\\", "\\\\"),
                    "original_id": "_elementid",
                    "next_val": str(len(all_data)),
                }
            except Exception as exc:
                log.exception("Error exporting nodes for label '%s': %s", label, exc)
        return vertex_args

    async def export_edges(
        self, thread_pool: concurrent.futures.Executor
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export edges from CSV files to new CSV files and return mapping for AGE import.
        """
        loop = asyncio.get_running_loop()
        edge_args: Dict[str, Dict[str, Any]] = {}
        rel_types = self.get_relationship_types()

        for rel_type in rel_types:
            try:
                await self.create_label_type(label_type="edge", value=rel_type)
                first_id = await self.get_first_id(self.graph_name, rel_type)
                count = self._count_edges(rel_type)
                if self.trial:
                    count = min(count, 100)
                    limit = min(count, int(self.chunk_size))
                else:
                    limit = int(self.chunk_size)
                log.info(
                    "Exporting %d edges for relationship type '%s'.", count, rel_type
                )

                tasks = [
                    loop.run_in_executor(
                        thread_pool,
                        self._fetch_edge_chunk_csv,
                        rel_type,
                        skip,
                        limit,
                    )
                    for skip in range(0, count, limit)
                ]
                chunks = await asyncio.gather(*tasks)
                all_data: List[Dict[str, Any]] = []

                for sublist in chunks:
                    for idx, item in enumerate(sublist):
                        try:
                            new_start_id = self.id_maps[item["start_vertex_type"]][
                                item["start_id"]
                            ]
                            new_end_id = self.id_maps[item["end_vertex_type"]][
                                item["end_id"]
                            ]
                        except KeyError:
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
        In trial mode, list nodes per relationship type by sampling edge CSV data.
        """
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as thread_pool:
            for rel_type in self.get_relationship_types():
                try:
                    count = min(self._count_edges(rel_type), self.no_of_edges_trial)
                    limit = min(int(self.chunk_size), count)
                    log.info("Listing nodes for relationship type '%s'.", rel_type)
                    tasks = [
                        loop.run_in_executor(
                            thread_pool,
                            self._fetch_edge_chunk_csv,
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
                except Exception as exc:
                    log.exception("Error listing nodes for '%s': %s", rel_type, exc)

    async def export(self) -> None:
        """
        Main export function to process both nodes and edges.
        """
        thread_pool = concurrent.futures.ThreadPoolExecutor()
        try:
            await self.set_up_graph(graph_name=self.graph_name, create_graph=True)
            if self.trial:
                await self.list_nodes()
            nodes_args = await self.export_nodes(thread_pool)
            if not nodes_args:
                log.error("No nodes exported.\nDoes the CSV contain nodes?")
                sys.exit(1)
            edges_args = await self.export_edges(thread_pool)
            if not edges_args:
                log.error("No edges exported.\nDoes the CSV contain edges?")
                sys.exit(1)
        except Exception as exc:
            log.exception("Error during export process: %s", exc)
            raise
        finally:
            thread_pool.shutdown()
        self.vertices = nodes_args
        self.edges = edges_args
