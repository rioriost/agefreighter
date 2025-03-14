#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime
import functools
import logging
import os
import platform
import sys
import tempfile
from typing import Dict, Any, List, Set, Optional

import numpy as np
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from psycopg.sql import SQL, Identifier, Literal
from psycopg.rows import namedtuple_row
import aiofiles

# Configure logging; default to INFO (overridable via --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Console formatting constants
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def reconnect_on_failure(func):
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        attempts = 0
        while attempts < self.max_attempts:
            try:
                return await func(self, *args, **kwargs)
            except ValueError:
                # Immediately re-raise ValueError (non-transient)
                raise
            except Exception as e:
                attempts += 1
                log.error("Error in %s: %s (attempt %d)", func.__name__, e, attempts)
                # (Connection-recovery logic omitted for brevity)
                await asyncio.sleep(self.retry_delay)
        log.error("Max attempts reached in %s", func.__name__)
        raise Exception(f"Max attempts reached in {func.__name__}")

    return wrapper


class AgeFreighter:
    """
    Manages a PostgreSQL connection pool, sets up an AGE graph,
    and copies CSV data using PostgreSQL's COPY protocol.
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
        self.dsn = dsn
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.chunk_size = chunk_size
        self.graph_name = graph_name
        self.save_temps = save_temps
        self.progress = progress
        self.output_dir = output_dir
        self.con_pool: Optional[AsyncConnectionPool] = None
        self.vertices: Dict[str, Any] = {}
        self.edges: Dict[str, Any] = {}
        self.extra_kwargs: Dict[str, Any] = kwargs
        self.max_attempts = 3
        self.retry_delay = 3

        # Append search_path options for AGE
        self.dsn_w_option = (
            f"{dsn} options='-c search_path=ag_catalog,\"$user\",public'"
        )

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
        # Cleanup temporary CSV files if not preserving temps.
        if not self.save_temps:
            try:
                combined = {**self.vertices, **self.edges}
                for csv_spec in combined.values():
                    csv_path = csv_spec.get("csv_path")
                    if csv_path and os.path.exists(csv_path):
                        log.info("Deleting CSV file: %s", csv_path)
                        os.remove(csv_path)
            except Exception as e:
                log.error("Error during cleanup of temporary files: %s", e)

    async def __aenter__(self) -> "AgeFreighter":
        log.debug("Entering async context for AgeFreighter.")
        self.con_pool = await self.connect(
            dsn=self.dsn_w_option,
            max_connections=self.max_connections,
            min_connections=self.min_connections,
            max_attempts=self.max_attempts,
            retry_delay=self.retry_delay,
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
        self,
        dsn: str,
        max_connections: int,
        min_connections: int,
        max_attempts: int,
        retry_delay: int,
        **kwargs,
    ) -> AsyncConnectionPool:
        """
        Open a connection pool with retry logic.
        """
        parsed_dsn = self.parse_dsn(dsn)
        log.debug("Opening connection pool for %s.", parsed_dsn["host"])

        # Increase file descriptor limits for Unix systems if needed
        if platform.system() in ["Darwin", "Linux"]:
            try:
                import resource

                current_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (8192, current_limit[1]))
            except Exception as e:
                log.error("Error setting resource limits: %s", e)

        con_pool = AsyncConnectionPool(
            dsn,
            max_size=max_connections,
            min_size=min_connections,
            open=False,
            timeout=7200,
            **kwargs,
        )

        for attempt in range(1, max_attempts + 1):
            try:
                await con_pool.open()
                await con_pool.wait()
                log.info(
                    "Connection pool for %s opened successfully.", parsed_dsn["host"]
                )
                break
            except PoolTimeout as e:
                log.error(
                    "Connection pool for %s timeout on attempt %d: %s",
                    parsed_dsn["host"],
                    attempt,
                    e,
                )
                if attempt == max_attempts:
                    raise
                await asyncio.sleep(retry_delay)
            except Exception as e:
                log.error(
                    "Error opening connection pool for %s on attempt %d: %s",
                    parsed_dsn["host"],
                    attempt,
                    e,
                )
                raise

        return con_pool

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

    @staticmethod
    def parse_dsn(dsn: str) -> dict:
        """
        Parse a DSN string into an AST.
        """
        try:
            return dict(item.split("=", 1) for item in dsn.split())
        except ValueError:
            raise ValueError("Invalid DSN format")

    @reconnect_on_failure
    async def _recover_label(self, label: str) -> None:
        """
        Recover the label/type relation.
        """
        log.info("Recovering label/type relation '%s'.", label)
        pool = self.con_pool
        assert pool is not None, "Connection pool is not initialized."
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                log.info("Truncating label '%s'.", label)
                await cur.execute(
                    SQL("TRUNCATE {graph_name}.{label};").format(
                        graph_name=Identifier(self.graph_name),
                        label=Identifier(label),
                    )
                )
                log.info("Resetting sequence for label '%s'.", label)
                await cur.execute(
                    f'SELECT setval(\'"{self.graph_name}"."{label}_id_seq"\', 1, false)'
                )

    async def _retry_copy(
        self, label: str, csv_spec: Dict[str, Any], kind: str
    ) -> None:
        """
        Helper to retry a COPY operation for a given label.
        """
        attempts = 0
        while attempts < self.max_attempts:
            try:
                attempts += 1
                await self._copy(
                    graph_name=self.graph_name,
                    csv_path=csv_spec["csv_path"],
                    label_name=label,
                    next_val=csv_spec["next_val"],
                    kind=kind,
                )
                return
            except Exception:
                log.exception("Failed to copy '%s' (attempt %d)", label, attempts)
                if attempts >= self.max_attempts:
                    log.error("Max attempts reached for copying '%s'", label)
                    sys.exit(1)
                await self._recover_label(label=label)
                await asyncio.sleep(self.retry_delay)

    @reconnect_on_failure
    async def _copy_vertices(self) -> None:
        """
        Copy all vertex CSV files into the AGE graph.
        """
        for vertex_label, csv_spec in self.vertices.items():
            log.info("Copying vertex '%s' using CSV spec: %s", vertex_label, csv_spec)
            await self._retry_copy(vertex_label, csv_spec, kind="v")

    @reconnect_on_failure
    async def _copy_edges(self) -> None:
        """
        Copy all edge CSV files into the AGE graph.
        """
        for edge_type, csv_spec in self.edges.items():
            log.info("Copying edge '%s' using CSV spec: %s", edge_type, csv_spec)
            await self._retry_copy(edge_type, csv_spec, kind="e")

    @reconnect_on_failure
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
        # graph_name_quoted = self.quoted_graph_name(graph_name)
        if kind == "v":
            query = SQL(
                "COPY {graph_name}.{label_name} FROM STDIN (FORMAT CSV)"
            ).format(
                graph_name=Identifier(graph_name), label_name=Identifier(label_name)
            )
        else:
            query = SQL(
                "COPY {graph_name}.{label_name} (id, start_id, end_id, properties) FROM STDIN (FORMAT CSV)"
            ).format(
                graph_name=Identifier(graph_name), label_name=Identifier(label_name)
            )

        pool = self.con_pool
        assert pool is not None, "Connection pool is not initialized."
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    async with cur.copy(query) as copy:
                        async with aiofiles.open(
                            csv_path, "r", encoding="utf-8"
                        ) as file:
                            while data := await file.read(self.chunk_size):
                                await copy.write(data)
                    log.info("Finished COPY for '%s'.", csv_path)
                except Exception as e:
                    log.error("Error during COPY from file '%s': %s", csv_path, e)
                    raise
                # Reset sequence after COPY
                await cur.execute(
                    f'SELECT setval(\'"{self.graph_name}"."{label_name}_id_seq"\', {next_val}, true)'
                )

                await cur.execute(SQL("COMMIT"))

    @staticmethod
    def extract_unique_keys(data: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract unique property keys from a list of records.
        """
        unique_keys = set()
        for row in data:
            unique_keys.update(row.get("properties", {}).keys())
        return unique_keys

    @staticmethod
    def quoted_graph_name(graph_name: str) -> str:
        """
        Return the properly quoted graph name.
        """
        log.debug("Quoting graph name: %s", graph_name)
        return f'"{graph_name}"' if graph_name.lower() != graph_name else graph_name

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

        await self._copy_vertices()
        await self._copy_edges()

    @reconnect_on_failure
    async def get_first_id(self, graph_name: str, label_type: str) -> int:
        """
        Get the first ID for a given label type.
        """
        quoted_graph = self.quoted_graph_name(graph_name)
        pool = self.con_pool
        assert pool is not None, "Connection pool is not initialized."
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                relation = f'{quoted_graph}."{label_type}"'
                await cur.execute(
                    SQL(
                        "SELECT id FROM ag_label WHERE relation = {}::regclass;"
                    ).format(Literal(relation))
                )
                row = await cur.fetchone()
                if row is None:
                    raise ValueError("No row returned; check your query or data.")
                ENTRY_ID_BITS = 32 + 16
                ENTRY_ID_MASK = np.uint64(0x0000FFFFFFFFFFFF)
                first_id = ((np.uint64(row[0])) << ENTRY_ID_BITS) | (
                    np.uint64(1) & ENTRY_ID_MASK
                )
                return int(first_id)

    @reconnect_on_failure
    async def set_up_graph(self, graph_name: str, create_graph: bool = False) -> None:
        """
        Create or validate the graph in the PostgreSQL database.
        """
        log.debug("Setting up graph '%s'.", graph_name)
        # Remove surrounding quotes if present.
        self.graph_name = Identifier(graph_name).as_string().strip('"')
        pool = self.con_pool
        assert pool is not None, "Connection pool is not initialized."
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                try:
                    await cur.execute(SQL("CREATE EXTENSION IF NOT EXISTS age CASCADE"))
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
                                "Dropping and recreating graph '%s'.", self.graph_name
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

    @reconnect_on_failure
    async def create_label_type(self, label_type: str, value: str) -> None:
        """
        Create a vertex or edge label type in the AGE graph.
        """
        log.debug("Creating %s label type with value '%s'.", label_type, value)
        pool = self.con_pool
        assert pool is not None, "Connection pool is not initialized."
        async with pool.connection() as conn:
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

    async def write_csv(self, label: str, kind: str, data: List[Dict[str, Any]]) -> str:
        """
        Write exported data to a CSV file.
        For nodes (kind "v") and edges (kind "e") different formats are used.
        If any row contains a tab, it is also written to an extra CSV file.

        Returns:
            The main CSV file path.
        """
        if not data:
            log.info("No data to write for '%s'.", label)
            return ""

        def format_kv(key: str, value: Any) -> str:
            safe_value = str(value).replace("\t", "\\t").replace('"', '\\""')
            return f'""{key}"": ""{safe_value}""'

        normal_file_path = (
            os.path.join(self.output_dir, f"{label.lower()}.csv")
            if self.output_dir
            else tempfile.NamedTemporaryFile(delete=False).name
        )

        tab_file_path: Optional[str] = None
        tab_file = None
        tab_printed = False
        headers = self.extract_unique_keys(data)
        BATCH_SIZE = 10000

        try:
            async with aiofiles.open(
                normal_file_path, "w", encoding="utf-8", newline=""
            ) as normal_f:
                normal_lines: List[str] = []
                tab_lines: List[str] = []
                for row in data:
                    props = row.get("properties", {})
                    formatted_parts = [
                        format_kv(h, props.get(h, ""))
                        for h in headers
                        if props.get(h, "")
                    ]
                    line = ", ".join(formatted_parts)
                    if kind == "e":
                        csv_line = f'{row["id"]},{row["start_id"]},{row["end_id"]},"{{{line}}}"\n'
                    elif kind == "v":
                        csv_line = f'{row["id"]},"{{{line}}}"\n'
                    else:
                        raise ValueError(f"Unsupported kind: {kind}")
                    normal_lines.append(csv_line)
                    if any("\t" in str(props.get(h, "")) for h in headers):
                        tab_lines.append(csv_line)
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
            if tab_file is not None:
                await tab_file.close()
                if not tab_printed and tab_file_path:
                    print(
                        f"{RED}{BOLD}Tab replacement occurred. Additional rows have been output to: {tab_file_path}{RESET}"
                    )
                    tab_printed = True

        return normal_file_path.replace("\\", "\\\\")
