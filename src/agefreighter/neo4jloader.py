#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import time

from agefreighter.csvdatamanager import CsvDataManager
from neo4j import AsyncGraphDatabase

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Neo4jLoader:
    """
    Load CSV data into Neo4j.
    """

    def __init__(
        self,
        csv_manager: CsvDataManager,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        log_level: int = logging.INFO,
    ) -> None:
        log.setLevel(log_level)
        self.csv_manager = csv_manager
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        log.info("Neo4jLoader initialized")

    async def load_data(self) -> None:
        log.info("Loading CSV to Neo4j")

        start_time = time.time()
        BATCH_SIZE = 1000
        df = self.csv_manager.get_dataframe()

        # Get unique start and end node information
        unique_starts = df[
            ["start_id", "start_vertex_type", "Name", "Address", "Email", "Phone"]
        ].drop_duplicates()
        unique_ends = df[
            ["end_id", "end_vertex_type", "SKU", "Price", "Color", "Size", "Weight"]
        ].drop_duplicates()

        start_label = unique_starts.iloc[0]["start_vertex_type"]
        end_label = unique_ends.iloc[0]["end_vertex_type"]

        async with AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        ) as driver:
            async with driver.session() as session:
                # Clear the database
                await session.run("MATCH (a)-[r]->() DELETE a, r")
                await session.run("MATCH (a) DELETE a")
                # Manage indices
                await session.run(f"DROP INDEX {start_label}_index_id IF EXISTS")
                await session.run(f"DROP INDEX {end_label}_index_id IF EXISTS")
                await session.run(
                    f"CREATE INDEX {start_label}_index_id FOR (n:{start_label}) ON (n.CustomerID)"
                )
                await session.run(
                    f"CREATE INDEX {end_label}_index_id FOR (n:{end_label}) ON (n.ProductID)"
                )

                # Create start nodes in batches
                for idx in range(0, len(unique_starts), BATCH_SIZE):
                    batch = unique_starts.iloc[idx : idx + BATCH_SIZE]
                    starts = [
                        {
                            start_label: row["start_vertex_type"],
                            "CustomerID": row["start_id"],
                            "Name": row["Name"],
                            "Address": row["Address"],
                            "Email": row["Email"],
                            "Phone": row["Phone"],
                        }
                        for _, row in batch.iterrows()
                    ]
                    query = (
                        f"UNWIND $starts AS row "
                        f"CREATE (a:{start_label} {{CustomerID: row.CustomerID, Name: row.Name, "
                        f"Address: row.Address, Email: row.Email, Phone: row.Phone}}) "
                        f"SET a += row"
                    )
                    await session.run(query, starts=starts)

                # Create end nodes in batches
                for idx in range(0, len(unique_ends), BATCH_SIZE):
                    batch = unique_ends.iloc[idx : idx + BATCH_SIZE]
                    ends = [
                        {
                            end_label: row["end_vertex_type"],
                            "ProductID": row["end_id"],
                            "SKU": row["SKU"],
                            "Price": row["Price"],
                            "Color": row["Color"],
                            "Size": row["Size"],
                            "Weight": row["Weight"],
                        }
                        for _, row in batch.iterrows()
                    ]
                    query = (
                        f"UNWIND $ends AS row "
                        f"CREATE (f:{end_label} {{ProductID: row.ProductID, SKU: row.SKU, "
                        f"Price: row.Price, Color: row.Color, Size: row.Size, Weight: row.Weight}}) "
                        f"SET f += row"
                    )
                    await session.run(query, ends=ends)

                # Create edges in batches
                for idx in range(0, len(df), BATCH_SIZE):
                    batch = df.iloc[idx : idx + BATCH_SIZE]
                    edges = [
                        {"from": row["start_id"], "to": row["end_id"]}
                        for _, row in batch.iterrows()
                    ]
                    query = (
                        f"UNWIND $edges AS row "
                        f"MATCH (from:{start_label} {{CustomerID: row.from}}) "
                        f"MATCH (to:{end_label} {{ProductID: row.to}}) "
                        f"CREATE (from)-[r:BOUGHT]->(to) "
                        f"SET r += row"
                    )
                    await session.run(query, edges=edges)
        self._show_time(start_time, sys._getframe().f_code.co_name)

    @staticmethod
    def _show_time(start_time: float, message: str) -> None:
        elapsed = time.time() - start_time if start_time else 0.0
        print(f"Time for {message}: {elapsed:.2f} seconds")
