#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import time
from typing import List

from agefreighter.csvdatamanager import CsvDataManager
import pandas as pd
import psycopg as pg

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class PgsqlLoader:
    """
    Load CSV data into PostgreSQL.
    """

    def __init__(
        self, csv_manager: CsvDataManager, src_dsn: str, log_level: int = logging.INFO
    ) -> None:
        log.setLevel(log_level)
        self.csv_manager = csv_manager
        self.src_dsn = src_dsn

    async def load_data(self) -> None:
        log.info("Loading CSV to PGSQL")

        start_time = time.time()
        schema = "public"
        src_tables = {"start": "Customer", "end": "Product", "edges": "BOUGHT"}

        df = self.csv_manager.get_dataframe()

        # Prepare data and types for each table
        data_frames: List[pd.DataFrame] = [
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        ]
        types: List[List[str]] = [[], [], []]

        # Start table (Customer)
        data_frames[0] = df[
            ["CustomerID", "Name", "Address", "Email", "Phone"]
        ].drop_duplicates()
        data_frames[0].insert(0, "CustomerSerial", range(1, len(data_frames[0]) + 1))
        types[0] = ["SERIAL", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT"]

        # End table (Product)
        data_frames[1] = df[
            ["ProductID", "Phrase", "SKU", "Price", "Color", "Size", "Weight"]
        ].drop_duplicates()
        data_frames[1].insert(0, "ProductSerial", range(1, len(data_frames[1]) + 1))
        types[1] = ["SERIAL", "TEXT", "TEXT", "TEXT", "REAL", "TEXT", "TEXT", "INT"]

        # Edges table (BOUGHT)
        data_frames[2] = df[["CustomerID", "ProductID"]].copy()
        data_frames[2].insert(0, "BoughtSerial", range(1, len(data_frames[2]) + 1))
        types[2] = ["SERIAL", "TEXT", "TEXT"]

        with pg.connect(self.src_dsn) as conn:
            with conn.cursor() as cur:
                for (table_key, table_name), df_data, col_types in zip(
                    src_tables.items(), data_frames, types
                ):
                    cur.execute(f'DROP TABLE IF EXISTS {schema}."{table_name}"')
                    columns = ", ".join(
                        [f'"{col}" {tp}' for col, tp in zip(df_data.columns, col_types)]
                    )
                    cur.execute(f'CREATE TABLE {schema}."{table_name}" ({columns})')
                    query = (
                        f'COPY {schema}."{table_name}" FROM STDIN (FORMAT TEXT, FREEZE)'
                    )
                    with cur.copy(query) as copy:
                        copy_data = "\n".join(
                            "\t".join(map(str, row))
                            for row in df_data.itertuples(index=False)
                        )
                        copy.write(copy_data)
                    # Create indexes based on table type
                    if table_key == "edges":
                        cur.execute(
                            f'CREATE INDEX ON {schema}."{table_name}"("CustomerID")'
                        )
                        cur.execute(
                            f'CREATE INDEX ON {schema}."{table_name}"("ProductID")'
                        )
                    elif table_key == "start":
                        cur.execute(
                            f'CREATE INDEX ON {schema}."{table_name}"("CustomerID")'
                        )
                    elif table_key == "end":
                        cur.execute(
                            f'CREATE INDEX ON {schema}."{table_name}"("ProductID")'
                        )
                cur.execute("COMMIT")
        self._show_time(start_time, sys._getframe().f_code.co_name)

    @staticmethod
    def _show_time(start_time: float, message: str) -> None:
        elapsed = time.time() - start_time if start_time else 0.0
        print(f"Time for {message}: {elapsed:.2f} seconds")
