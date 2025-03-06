#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime
import logging
import os
import platform
import sys
import tempfile
from typing import Dict, Any, Union, List, Set, Optional

import numpy as np
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from psycopg.sql import SQL, Identifier, Literal
from psycopg.rows import namedtuple_row
import aiofiles

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


class AgeFreighter:
    """
    Provides methods to manage a PostgreSQL connection pool,
    set up the AGE graph, and copy CSV data using PostgreSQL's COPY protocol.
    """

    def __init__(
        self,
        dsn: str = "",
        max_connections: int = 64,
        min_connections: int = 4,
        save_temps: bool = False,
        progress: bool = True,
        output_dir: Optional[str] = None,
        chunk_size: int = 8192,
        graph_name: str = "",
        log_level: int = logging.INFO,
        **kwargs,
    ) -> None:
        log.setLevel(log_level)
        log.debug("Initializing AgeFreighter.")
        self.max_connections: int = max_connections
        self.min_connections: int = min_connections
        self.con_pool: Union[None, AsyncConnectionPool] = None
        self.save_temps: bool = save_temps
        self.progress: bool = progress
        self.output_dir: Optional[str] = output_dir
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")
        self.chunk_size: int = chunk_size
        self.graph_name: str = graph_name

        self.vertices: Dict[str, Any] = {}
        self.edges: Dict[str, Any] = {}
        self.dsn = dsn
        self.dsn_w_option = (
            dsn + " options='-c search_path=ag_catalog,\"$user\",public'"
        )
        self.extra_kwargs: Dict[str, Any] = kwargs
        self.max_attempts = 3
        self.retry_delay = 3

        # Create output directory if save_temps flag is True
        if self.save_temps:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(os.getcwd(), f"exported_{timestamp}")
            if not os.path.exists(self.output_dir):
                log.info(
                    "Output directory '%s' does not exist. Creating it.",
                    self.output_dir,
                )
                try:
                    os.makedirs(self.output_dir)
                except Exception as e:
                    log.error(
                        "Error creating output directory '%s': %s", self.output_dir, e
                    )
                    raise

    def __del__(self):
        # Cleanup temporary CSV files if save_temps flag is not set
        if not self.save_temps:
            try:
                # Merge vertices and edges dictionaries
                combined = {**self.vertices, **self.edges}
                for label, csv_spec in combined.items():
                    csv_path = csv_spec.get("csv_path")
                    if csv_path and os.path.exists(csv_path):
                        log.info("Deleting CSV file: %s", csv_path)
                        os.remove(csv_path)
            except Exception as e:
                log.error("Error during cleanup of temporary files: %s", e)

    async def __aenter__(self) -> "AgeFreighter":
        log.debug("Entering async context for AgeFreighter.")
        await self.connect(
            max_connections=self.max_connections,
            min_connections=self.min_connections,
            **self.extra_kwargs,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc_type:
            log.error("Exception in AgeFreighter context: %s", exc)
        if self.con_pool:
            try:
                await self.con_pool.close()
            except Exception as e:
                log.error("Error closing connection pool: %s", e)
        log.debug("Exiting AgeFreighter context.")

    async def connect(
        self, max_connections: int = 64, min_connections: int = 4, **kwargs
    ) -> None:
        """
        Open a connection pool with retry logic.
        """
        log.debug("Opening connection pool in connect().")
        MAX_RETRIES = 3
        DELAY_SEC = 5

        # Increase file descriptor limits for Unix systems if needed
        if platform.system() in ["Darwin", "Linux"]:
            try:
                import resource

                current_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (8192, current_limit[1]))
            except Exception as e:
                log.error("Error setting resource limits: %s", e)
                # Not critical; proceed without raising

        self.con_pool = AsyncConnectionPool(
            self.dsn_w_option,
            max_size=max_connections,
            min_size=min_connections,
            open=False,
            timeout=600,
            **kwargs,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await self.con_pool.open()
                await self.con_pool.wait()
                log.info("Connection pool opened successfully.")
                break
            except PoolTimeout as e:
                log.error("Pool timeout on attempt %d: %s", attempt, e)
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(DELAY_SEC)
            except Exception as e:
                log.error("Error opening connection pool on attempt %d: %s", attempt, e)
                raise

    async def close(self) -> None:
        """
        Close the connection pool.
        """
        log.debug("Closing connection pool.")
        if self.con_pool:
            try:
                await self.con_pool.close()
            except Exception as e:
                log.error("Error closing connection pool: %s", e)

    async def _recover_label(self, label: str) -> None:
        """
        Recover the label / type relation
        """
        log.info(f"Recovering label / type relation {label}.")
        assert self.con_pool is not None, "Connection pool is not initialized."
        async with self.con_pool.connection() as conn:
            async with conn.cursor() as cur:
                log.info(f"Truncating vertex label {label}.")
                await cur.execute(f'TRUNCATE "{self.graph_name}"."{label}";')
                log.info(f"Setting sequence for vertex label {label}.")
                await cur.execute(
                    f'SELECT setval(\'"{self.graph_name}"."{label}_id_seq"\', 1, false)'
                )

    async def copy(self) -> None:
        """
        Copy vertices and edges from CSV files into the AGE graph.
        """
        if not self.graph_name:
            raise ValueError("graph_name must be specified")
        if not self.vertices:
            raise ValueError("vertices must be specified")
        if not self.edges:
            raise ValueError("edges must be specified")

        for vertex_label, csv_spec in self.vertices.items():
            log.info("Copying vertex '%s' using CSV spec: %s", vertex_label, csv_spec)
            attempts = 0
            while True:
                try:
                    attempts += 1
                    await self._copy(
                        graph_name=self.graph_name,
                        csv_path=csv_spec["csv_path"],
                        label_name=vertex_label,
                        next_val=csv_spec["next_val"],
                        kind="v",
                    )
                    break
                except Exception:
                    if attempts > self.max_attempts:
                        log.error(
                            "Max attempts reached for copying vertex '%s'", vertex_label
                        )
                        sys.exit(1)
                    log.exception("Failed to copy vertex '%s'", vertex_label)
                    await self._recover_label(label=vertex_label)
                    await asyncio.sleep(self.retry_delay)

        for edge_type, csv_spec in self.edges.items():
            log.info("Copying edge '%s' using CSV spec: %s", edge_type, csv_spec)
            attempts = 0
            while True:
                try:
                    attempts += 1
                    await self._copy(
                        graph_name=self.graph_name,
                        csv_path=csv_spec["csv_path"],
                        label_name=edge_type,
                        next_val=csv_spec["next_val"],
                        kind="e",
                    )
                    break
                except Exception:
                    if attempts > self.max_attempts:
                        log.error(
                            "Max attempts reached for copying edge '%s'", edge_type
                        )
                        sys.exit(1)
                    log.exception("Failed to copy edge '%s'", edge_type)
                    await asyncio.sleep(self.retry_delay)
                    await self._recover_label(label=edge_type)

    async def _copy(
        self,
        graph_name: str,
        csv_path: str,
        label_name: str,
        next_val: int,
        kind: str,
    ) -> None:
        """
        Execute the COPY command to load CSV data.
        """
        log.debug("Starting COPY for file '%s' into label '%s'.", csv_path, label_name)
        if kind not in ["v", "e"]:
            raise ValueError(f"Invalid kind: {kind}")
        graph_name_quoted = self.quoted_graph_name(graph_name)
        if kind == "v":
            query = f'COPY {graph_name_quoted}."{label_name}" FROM STDIN (FORMAT CSV)'
        else:
            query = f'COPY {graph_name_quoted}."{label_name}" (id, start_id, end_id, properties) FROM STDIN (FORMAT CSV)'
        assert self.con_pool is not None, "Connection pool is not initialized."
        async with self.con_pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    async with cur.copy(query) as copy:
                        async with aiofiles.open(
                            csv_path, "r", encoding="utf-8"
                        ) as file:
                            while True:
                                data = await file.read(self.chunk_size)
                                if not data:
                                    break
                                await copy.write(data)
                    log.info("Finished COPY for '%s'.", csv_path)
                except Exception as e:
                    log.error("Error during COPY from file '%s': %s", csv_path, e)
                    raise
                query = f'SELECT setval(\'"{self.graph_name}"."{label_name}_id_seq"\', {next_val}, true)'
                await cur.execute(query)
                await cur.execute("COMMIT")

    @staticmethod
    def extract_unique_keys(data: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract a set of unique property keys from a list of records.
        """
        unique_keys = set()
        for row in data:
            properties = row.get("properties", {})
            unique_keys.update(properties.keys())
        return unique_keys

    @staticmethod
    def quoted_graph_name(graph_name: str) -> str:
        """
        Quote the graph name if needed.
        """
        log.debug("Quoting graph name: %s", graph_name)
        return f'"{graph_name}"' if graph_name.lower() != graph_name else graph_name

    async def set_up_graph(self, graph_name: str, create_graph: bool = False) -> None:
        """
        Create or validate the graph in the PostgreSQL database.
        """
        log.debug("Setting up graph '%s'.", graph_name)
        # Remove quotes from Identifier string if present.
        self.graph_name = Identifier(graph_name).as_string().strip('"')
        assert self.con_pool is not None, "Connection pool is not initialized."
        async with self.con_pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                try:
                    await cur.execute(SQL("CREATE EXTENSION IF NOT EXISTS age CASCADE"))
                    # Use parameterized SQL to avoid injection
                    await cur.execute(
                        SQL("SELECT count(*) FROM ag_graph WHERE name = {name}").format(
                            name=Literal(self.graph_name)
                        )
                    )
                    row = await cur.fetchone()
                    exists = row is not None and row.count == 1
                    if exists:
                        log.debug("Graph '%s' already exists.", self.graph_name)
                        if create_graph:
                            log.debug(
                                "Dropping and recreating graph '%s'.",
                                self.graph_name,
                            )
                            await cur.execute(
                                SQL("SELECT drop_graph({name}, true)").format(
                                    name=Literal(self.graph_name)
                                )
                            )
                            await cur.execute(
                                SQL("SELECT create_graph({name})").format(
                                    name=Literal(self.graph_name)
                                )
                            )
                        else:
                            log.debug("Using existing graph '%s'.", self.graph_name)
                    else:
                        log.debug("Graph '%s' does not exist.", self.graph_name)
                        if create_graph:
                            log.debug("Creating graph '%s'.", self.graph_name)
                            await cur.execute(
                                SQL("SELECT create_graph({name})").format(
                                    name=Literal(self.graph_name)
                                )
                            )
                        else:
                            raise ValueError(
                                f"Graph '{self.graph_name}' doesn't exist. Set create_graph=True."
                            )
                except Exception as e:
                    log.error("Error setting up graph '%s': %s", self.graph_name, e)
                    raise

    async def create_label_type(self, label_type: str, value: str) -> None:
        """
        Create a vertex or edge label type in the AGE graph.
        """
        log.debug("Creating %s label type with value '%s'.", label_type, value)
        assert self.con_pool is not None, "Connection pool is not initialized."
        async with self.con_pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    if label_type == "vertex":
                        log.debug("Creating vertex label '%s'.", value)
                        await cur.execute(
                            SQL("SELECT create_vlabel({schema}, {label});").format(
                                schema=Literal(self.graph_name),
                                label=Literal(value),
                            )
                        )
                        log.debug("Creating indices for vertex '%s'.", value)
                        await cur.execute(
                            SQL(
                                "CREATE INDEX ON {schema}.{label} USING GIN (properties);"
                            ).format(
                                schema=Identifier(self.graph_name),
                                label=Identifier(value),
                            )
                        )
                        await cur.execute(
                            SQL(
                                "CREATE INDEX ON {schema}.{label} USING BTREE (id);"
                            ).format(
                                schema=Identifier(self.graph_name),
                                label=Identifier(value),
                            )
                        )
                    elif label_type == "edge":
                        log.debug("Creating edge label '%s'.", value)
                        await cur.execute(
                            SQL("SELECT create_elabel({schema}, {label});").format(
                                schema=Literal(self.graph_name),
                                label=Literal(value),
                            )
                        )
                        log.debug("Creating indices for edge '%s'.", value)
                        await cur.execute(
                            SQL("CREATE INDEX ON {schema}.{label} (start_id);").format(
                                schema=Identifier(self.graph_name),
                                label=Identifier(value),
                            )
                        )
                        await cur.execute(
                            SQL("CREATE INDEX ON {schema}.{label} (end_id);").format(
                                schema=Identifier(self.graph_name),
                                label=Identifier(value),
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported label type: {label_type}")
                except Exception as e:
                    log.error(
                        "Error creating label type '%s' with value '%s': %s",
                        label_type,
                        value,
                        e,
                    )
                    raise

    async def drop_label_type(self, label_type: str, value: str) -> None:
        """
        Drop a vertex or edge label type in the AGE graph.
        """
        log.debug("Dropping %s label type with value '%s'.", label_type, value)
        assert self.con_pool is not None, "Connection pool is not initialized."
        async with self.con_pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    if label_type == "vertex":
                        log.debug("Dropping vertex label '%s'.", value)
                        await cur.execute(
                            SQL("SELECT drop_label({schema}, {label});").format(
                                schema=Literal(self.graph_name),
                                label=Literal(value),
                            )
                        )
                    elif label_type == "edge":
                        log.debug("Dropping edge label '%s'.", value)
                        await cur.execute(
                            SQL("SELECT drop_label({schema}, {label});").format(
                                schema=Literal(self.graph_name),
                                label=Literal(value),
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported label type: {label_type}")
                except Exception as e:
                    log.error(
                        "Error creating label type '%s' with value '%s': %s",
                        label_type,
                        value,
                        e,
                    )
                    raise

    async def get_first_id(self, graph_name: str, label_type: str) -> int:
        """
        Get the first ID for a given label type.
        """
        quoted_graph = self.quoted_graph_name(graph_name)
        assert self.con_pool is not None, "Connection pool is not initialized."
        async with self.con_pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    relation = f'{quoted_graph}."{label_type}"'
                    query = SQL(
                        "SELECT id FROM ag_label WHERE relation = {}::regclass;"
                    ).format(Literal(relation))
                    await cur.execute(query)
                    row = await cur.fetchone()
                    if row is None:
                        raise ValueError("No row returned; check your query or data.")
                    ENTRY_ID_BITS = 32 + 16
                    ENTRY_ID_MASK = np.uint64(0x0000FFFFFFFFFFFF)
                    first_id = ((np.uint64(row[0])) << ENTRY_ID_BITS) | (
                        np.uint64(1) & ENTRY_ID_MASK
                    )
                    return int(first_id)
                except Exception as e:
                    log.error(
                        "Error getting first ID for label '%s': %s", label_type, e
                    )
                    raise

    async def write_csv(self, label: str, kind: str, data: List[Dict[str, Any]]) -> str:
        """
        Write exported data to CSV. Uses different formats for nodes and edges.
        If self.output_dir is None, a temporary file is created.
        Additionally, if any row contains a tab (chr(9)) in any property value,
        that row is also written to an extra CSV file in a directory named
        "tab_replaced_yyyymmdd_hhmmss" under the current working directory.
        A message is printed if such tab-containing rows are found.

        Returns:
            The file path of the main CSV file.
        """
        if not data:
            log.info("No data to write for '%s'.", label)
            return ""

        def format_kv(key: str, value: Any) -> str:
            # Escape tab characters in the property value.
            safe_value = str(value).replace("\t", "\\t")
            return f'""{key}"": ""{safe_value}""'

        # Determine the main CSV file path.
        if self.output_dir:
            normal_file_path = os.path.join(self.output_dir, f"{label.lower()}.csv")
        else:
            normal_file_path = tempfile.NamedTemporaryFile(delete=False).name

        # Prepare variables for tab-replaced rows.
        tab_file_path: Optional[str] = None
        tab_file = None
        tab_printed = False  # Flag to ensure we print the message only once

        # Extract headers from the data.
        headers = self.extract_unique_keys(data)

        try:
            async with aiofiles.open(
                normal_file_path, "w", encoding="utf-8", newline=""
            ) as normal_f:
                BATCH_SIZE = 10000

                normal_lines: List[str] = []
                tab_lines: List[str] = []

                for row in data:
                    # Cache the properties dictionary to avoid repeated lookups.
                    props = row.get("properties", {})

                    # Build formatted key/value parts once.
                    formatted_parts = [
                        format_kv(h, props.get(h, ""))
                        for h in headers
                        if props.get(h, "")
                    ]
                    line = ", ".join(formatted_parts)

                    # Build the CSV line based on the kind.
                    if kind == "e":
                        csv_line = f'{row["id"]},{row["start_id"]},{row["end_id"]},"{{{line}}}"\n'
                    elif kind == "v":
                        csv_line = f'{row["id"]},"{{{line}}}"\n'
                    else:
                        raise ValueError(f"Unsupported kind: {kind}")

                    normal_lines.append(csv_line)

                    # Check if any property value in this row contains a tab.
                    if any("\t" in str(props.get(h, "")) for h in headers):
                        tab_lines.append(csv_line)

                    # When we reach the batch size, write to the files and reset the lists.
                    if len(normal_lines) >= BATCH_SIZE:
                        await normal_f.write("".join(normal_lines))
                        normal_lines = []
                    if tab_lines and len(tab_lines) >= BATCH_SIZE:
                        if tab_file is None:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            tab_dir = os.path.join(
                                os.getcwd(), f"tab_replaced_{timestamp}"
                            )
                            os.makedirs(tab_dir, exist_ok=True)
                            tab_file_path = os.path.join(
                                tab_dir, f"{label.lower()}_tab_replaced.csv"
                            )
                            tab_file = await aiofiles.open(
                                tab_file_path, "w", encoding="utf-8", newline=""
                            )
                        await tab_file.write("".join(tab_lines))
                        tab_lines = []

                # Write any remaining lines
                if normal_lines:
                    await normal_f.write("".join(normal_lines))
                if tab_lines:
                    if tab_file is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        tab_dir = os.path.join(os.getcwd(), f"tab_replaced_{timestamp}")
                        os.makedirs(tab_dir, exist_ok=True)
                        tab_file_path = os.path.join(
                            tab_dir, f"{label.lower()}_tab_replaced.csv"
                        )
                        tab_file = await aiofiles.open(
                            tab_file_path, "w", encoding="utf-8", newline=""
                        )
                    await tab_file.write("".join(tab_lines))

            log.info("Exported %d records to %s.", len(data), normal_file_path)
        except Exception as e:
            log.exception("Error writing CSV for '%s': %s", label, e)
            raise
        finally:
            # If a tab file was used, close it and print the message once.
            if tab_file is not None:
                await tab_file.close()
                if not tab_printed:
                    print(
                        f"{RED}{BOLD}Tab replacement occurred. Additional rows have been output to: {tab_file_path}{RESET}"
                    )
                    tab_printed = True

        # Replace backslashes for compatibility if needed.
        return normal_file_path.replace("\\", "\\\\")
