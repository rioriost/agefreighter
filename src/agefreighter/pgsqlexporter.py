#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for exporting data from a Cosmos NoSQL database to CSV files and
loading them using AgeFreighter.
"""

import logging
from typing import Any, Dict

from .agefreighter import AgeFreighter

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class PGSQLExporter(AgeFreighter):
    """
    Exports nodes and edges from a Cosmos NoSQL database into CSV files for AGE COPY import.
    """

    def __init__(
        self,
        dsn: str,
        min_connections: int,
        max_connections: int,
        src_dsn: str,
        trial: bool,
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
        self.src = src_dsn
        self.trial = trial
        self.no_of_edges_trial = 100
        self.graph_name = graph_name
        self.id_maps: Dict[str, Dict[str, int]] = {}

    async def __aenter__(self) -> "PGSQLExporter":
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await super().__aexit__(exc_type, exc_value, traceback)

    async def export(self) -> None:
        """
        Main export function to process both nodes and edges.
        """
        try:
            await self.set_up_graph(graph_name=self.graph_name, create_graph=True)
            vertex_args: dict[str, str] = {}
            edge_args: dict[str, str] = {}
        except Exception as exc:
            log.exception("Error during export process: %s", exc)
            raise

        self.vertices = vertex_args
        self.edges = edge_args
