#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import concurrent.futures
import logging
import sys
import time
from typing import Any, Dict

from agefreighter.csvdatamanager import CsvDataManager
from gremlin_python.driver import client, serializer  # type: ignore

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CosmosGremlinLoader:
    """
    Load CSV data into Cosmos DB using the Gremlin API.
    """

    def __init__(
        self,
        csv_manager: CsvDataManager,
        cosmos_gremlin_endpoint: str,
        cosmos_key: str,
        cosmos_database: str,
        cosmos_container: str,
        log_level: int = logging.INFO,
    ) -> None:
        log.setLevel(log_level)
        self.csv_manager = csv_manager
        self.cosmos_gremlin_endpoint = cosmos_gremlin_endpoint
        self.cosmos_key = cosmos_key
        self.cosmos_database = cosmos_database
        self.cosmos_container = cosmos_container
        log.info("CosmosGremlinLoader initialized")

    def execute_gremlin_query(self, g_client: client.Client, query: str) -> None:
        """
        Execute a Gremlin query with retry logic.
        """
        retries = 0
        initial_wait = 1
        while retries < 5:
            try:
                future = g_client.submitAsync(query)
                result = future.result()
                log.debug(f"Gremlin query result: {result.all().result()}")
                return
            except Exception:
                wait_time = initial_wait * (2**retries)
                log.warning(
                    f"Query failed (attempt {retries + 1}). Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
                retries += 1
        raise Exception("Max retries exceeded for Gremlin query")

    async def load_data(self) -> None:
        log.info("Loading CSV to Cosmos DB via Gremlin API")

        start_time = time.time()
        COSMOS_USERNAME = "/dbs/db1/colls/transaction"
        COSMOS_PKEY = "pk"

        LOGICAL_PARTITION_SIZE = 20 * 1024 * 1024 * 1024  # 20GB
        AVERAGE_SIZE_OF_DOCUMENT = 512  # 512 bytes
        num_of_docs_per_partition = LOGICAL_PARTITION_SIZE // AVERAGE_SIZE_OF_DOCUMENT
        num_of_pk = 1
        MAX_OPERATOR_DEPTH = 400

        try:
            g_client = client.Client(
                url=self.cosmos_gremlin_endpoint,
                traversal_source="g",
                username=COSMOS_USERNAME,
                password=self.cosmos_key,
                message_serializer=serializer.GraphSONSerializersV2d0(),
                timeout=600,
            )
        except Exception as e:
            print(f"Failed to connect to Gremlin server: {e}")
            return

        df = self.csv_manager.get_dataframe()
        df.drop_duplicates(inplace=True)

        vertex_columns: Dict[str, Any] = {
            "Customer": ["CustomerID", "Name", "Address", "Email", "Phone"],
            "Product": [
                "ProductID",
                "Phrase",
                "SKU",
                "Price",
                "Color",
                "Size",
                "Weight",
            ],
        }

        total_docs = 0
        max_workers = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for vertex_type, cols in vertex_columns.items():
                vertices = df[cols].drop_duplicates()
                # Escape single quotes in string fields
                vertices = vertices.applymap(
                    lambda x: x.replace("'", r"\'") if isinstance(x, str) else x
                )
                if vertex_type == "Customer":
                    tmp_query = """.addV('Customer')
                        .property('Name', '{Name}')
                        .property('CustomerID', '{CustomerID}')
                        .property('Address', '{Address}')
                        .property('Email', '{Email}')
                        .property('Phone', '{Phone}')
                        .property('{pk}', '{num_of_pk}')"""
                elif vertex_type == "Product":
                    tmp_query = """.addV('Product')
                        .property('Phrase', '{Phrase}')
                        .property('ProductID', '{ProductID}')
                        .property('SKU', '{SKU}')
                        .property('Price', '{Price}')
                        .property('Color', '{Color}')
                        .property('Size', '{Size}')
                        .property('Weight', '{Weight}')
                        .property('{pk}', '{num_of_pk}')"""
                chunk_size = int(MAX_OPERATOR_DEPTH / (len(cols) + 2))
                for i, chunk in enumerate(
                    CsvDataManager.get_chunks(vertices, chunk_size)
                ):
                    log.info(
                        f"Creating '{vertex_type}' vertices: {len(chunk)} records."
                    )
                    if len(chunk.columns) == 5:
                        query = "g" + "".join(
                            [
                                tmp_query.format(
                                    Name=row["Name"],
                                    CustomerID=row["CustomerID"],
                                    Address=row["Address"],
                                    Email=row["Email"],
                                    Phone=row["Phone"],
                                    pk=COSMOS_PKEY,
                                    num_of_pk=num_of_pk,
                                )
                                for _, row in chunk.iterrows()
                            ]
                        )
                    elif len(chunk.columns) == 7:
                        query = "g" + "".join(
                            [
                                tmp_query.format(
                                    Phrase=row["Phrase"],
                                    ProductID=row["ProductID"],
                                    SKU=row["SKU"],
                                    Price=row["Price"],
                                    Color=row["Color"],
                                    Size=row["Size"],
                                    Weight=row["Weight"],
                                    pk=COSMOS_PKEY,
                                    num_of_pk=num_of_pk,
                                )
                                for _, row in chunk.iterrows()
                            ]
                        )
                    futures.append(
                        executor.submit(self.execute_gremlin_query, g_client, query)
                    )
                    total_docs += len(chunk)
                    if total_docs % num_of_docs_per_partition == 0:
                        num_of_pk += 1
            concurrent.futures.wait(futures)

        # Create edges (BOUGHT relationships) using a larger thread pool
        max_workers = 1024
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, row in enumerate(df.itertuples(index=False), start=1):
                log.info(f"Creating 'BOUGHT' edge: {i}")
                edge_query = (
                    f"g.V().has('CustomerID', '{row.CustomerID}')"
                    f".addE('BOUGHT').to(g.V().has('ProductID', '{row.ProductID}'))"
                )
                futures.append(
                    executor.submit(self.execute_gremlin_query, g_client, edge_query)
                )
            concurrent.futures.wait(futures)

        g_client.close()
        self._show_time(start_time, sys._getframe().f_code.co_name)

    @staticmethod
    def _show_time(start_time: float, message: str) -> None:
        elapsed = time.time() - start_time if start_time else 0.0
        print(f"Time for {message}: {elapsed:.2f} seconds")
