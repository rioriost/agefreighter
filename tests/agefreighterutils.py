#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import logging
import os
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AgeFreighterUtils:
    def __init__(self):
        self.data_dir = "../data/transaction/"
        self.base_file = "customer_product_bought"
        self.csv_file = f"{self.data_dir}{self.base_file}.csv"
        self.avro_file = f"{self.data_dir}{self.base_file}.avro"
        self.parquet_file = f"{self.data_dir}{self.base_file}.parquet"
        self.pickle_file = f"{self.data_dir}{self.base_file}.pickle"

    def __del__(self):
        log.info("Deleting AgeUtils")

    def showTime(self, start_time: float = 0.0, message: str = ""):
        """
        Show the time
        """
        message = "Time for " + message + ": {proc_time:.2f}"
        if start_time == 0.0:
            print(message.format(proc_time=0.0))
        else:
            print(message.format(proc_time=time.time() - start_time))

    async def loadCSVtoNeo4j(self) -> None:
        """
        Load CSV to Neo4j
        """
        log.info("Loading CSV to Neo4j")
        from neo4j import AsyncGraphDatabase

        try:
            n4j_uri = os.environ["NEO4J_URI"]
            n4j_user = os.environ["NEO4J_USER"]
            n4j_password = os.environ["NEO4J_PASSWORD"]
        except KeyError:
            print(
                "Please set the environment variables NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"
            )
            return

        start_time = time.time()
        BATCH_SIZE = 1000
        df = pd.read_csv(self.csv_file)
        uniq_starts = df[
            ["CustomerID", "start_vertex_type", "Name", "Address", "Email", "Phone"]
        ].drop_duplicates()
        uniq_ends = df[
            ["ProductID", "end_vertex_type", "SKU", "Price", "Color", "Size", "Weight"]
        ].drop_duplicates()

        start_name = uniq_starts.iloc[0].start_vertex_type
        end_name = uniq_ends.iloc[0].end_vertex_type

        async with AsyncGraphDatabase.driver(
            n4j_uri, auth=(n4j_user, n4j_password)
        ) as driver:
            async with driver.session() as session:
                # clear the database
                await session.run("MATCH (a)-[r]->() DELETE a, r")
                await session.run("MATCH (a) DELETE a")
                await session.run(f"DROP INDEX {start_name}_index_id IF EXISTS")
                await session.run(f"DROP INDEX {end_name}_index_id IF EXISTS")
                await session.run(
                    f"CREATE INDEX {start_name}_index_id FOR (n:{start_name}) ON (n.CustomerID)"
                )
                await session.run(
                    f"CREATE INDEX {end_name}_index_id FOR (n:{end_name}) ON (n.ProductID)"
                )
                # create start nodes
                for idx in range(0, len(uniq_starts), BATCH_SIZE):
                    starts = [
                        {
                            start_name: start_vertex_type,
                            "CustomerID": customerid,
                            "Name": name,
                            "Address": address,
                            "Email": email,
                            "Phone": phone,
                        }
                        for i, (
                            start_vertex_type,
                            customerid,
                            name,
                            address,
                            email,
                            phone,
                        ) in enumerate(
                            zip(
                                uniq_starts["start_vertex_type"][
                                    idx : idx + BATCH_SIZE
                                ].tolist(),
                                uniq_starts["CustomerID"][
                                    idx : idx + BATCH_SIZE
                                ].tolist(),
                                uniq_starts["Name"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_starts["Address"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_starts["Email"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_starts["Phone"][idx : idx + BATCH_SIZE].tolist(),
                            )
                        )
                    ]
                    await session.run(
                        f"""UNWIND $starts AS row
                        CREATE (a:{start_name} {{CustomerID: row.CustomerID, Name: row.name, Address: row.address, Email: row.email, Phone: row.phone}})
                        SET a += row""",
                        starts=starts,
                    )
                # create end nodes
                for idx in range(0, len(uniq_ends), BATCH_SIZE):
                    ends = [
                        {
                            end_name: end_vertex_type,
                            "ProductID": productid,
                            "SKU": sku,
                            "Price": price,
                            "Color": color,
                            "Size": size,
                            "Weight": weight,
                        }
                        for i, (
                            end_vertex_type,
                            productid,
                            sku,
                            price,
                            color,
                            size,
                            weight,
                        ) in enumerate(
                            zip(
                                uniq_ends["end_vertex_type"][
                                    idx : idx + BATCH_SIZE
                                ].tolist(),
                                uniq_ends["ProductID"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_ends["SKU"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_ends["Price"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_ends["Color"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_ends["Size"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_ends["Weight"][idx : idx + BATCH_SIZE].tolist(),
                            )
                        )
                    ]
                    await session.run(
                        f"""UNWIND $ends AS row
                        CREATE (f:{end_name} {{ProductID: row.ProductID, SKU: row.SKU, Price: row.Price, Color: row.Color, Size: row.Size, Weight: row.Weight}})
                        SET f += row""",
                        ends=ends,
                    )
                # create edges
                for idx in range(0, len(df), BATCH_SIZE):
                    edges = [
                        {
                            "from": start_id,
                            "to": end_id,
                        }
                        for i, (start_id, end_id) in enumerate(
                            zip(
                                df["CustomerID"][idx : idx + BATCH_SIZE].tolist(),
                                df["ProductID"][idx : idx + BATCH_SIZE].tolist(),
                            )
                        )
                    ]
                    await session.run(
                        f"""UNWIND $edges AS row
                        MATCH (from:{start_name} {{CustomerID: row.from}})
                        MATCH (to:{end_name} {{ProductID: row.to}})
                        CREATE (from)-[r:BOUGHT]->(to)
                        SET r += row""",
                        edges=edges,
                    )
        self.showTime(start_time, sys._getframe().f_code.co_name)

    async def loadCSVtoPGSQL(self) -> None:
        """
        Load CSV to PGSQL
        """
        log.info("Loading CSV to PGSQL")
        import psycopg as pg

        try:
            con_string = os.environ["SRC_PG_CONNECTION_STRING"]
        except KeyError:
            print("Please set the environment variables SRC_PG_CONNECTION_STRING")
            return

        start_time = time.time()
        schema = "public"
        src_tables = {"start": "Customer", "end": "Product", "edges": "BOUGHT"}

        df = pd.read_csv(self.csv_file)

        datum = [None, None, None]
        types = [None, None, None]

        datum[0] = df[
            ["CustomerID", "Name", "Address", "Email", "Phone"]
        ].drop_duplicates()
        datum[0].insert(0, "CustomerSerial", range(1, len(datum[0]) + 1))
        types[0] = ["SERIAL", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT"]

        datum[1] = df[
            ["ProductID", "Phrase", "SKU", "Price", "Color", "Size", "Weight"]
        ].drop_duplicates()
        datum[1].insert(0, "ProductSerial", range(1, len(datum[1]) + 1))
        types[1] = ["SERIAL", "TEXT", "TEXT", "TEXT", "REAL", "TEXT", "TEXT", "INT"]

        datum[2] = df[["CustomerID", "ProductID"]]
        datum[2].insert(0, "BoughtSerial", range(1, len(datum[2]) + 1))
        types[2] = ["SERIAL", "TEXT", "TEXT"]

        with pg.connect(con_string) as conn:
            with conn.cursor() as cur:
                for idx, ((table_k, table_v), data, type) in enumerate(
                    zip(src_tables.items(), datum, types)
                ):
                    cur.execute(f'DROP TABLE IF EXISTS {schema}."{table_v}"')
                    cols = ",".join(
                        [
                            f'"{col}"' + " " + tp
                            for _, (col, tp) in enumerate(zip(data.columns, type))
                        ]
                    )
                    cur.execute(f'CREATE TABLE {schema}."{table_v}" ({cols})')
                    query = (
                        f'COPY {schema}."{table_v}" FROM STDIN (FORMAT TEXT, FREEZE)'
                    )
                    with cur.copy(query) as copy:
                        copy.write(
                            "\n".join(
                                [
                                    "\t".join(map(str, row))
                                    for row in data.itertuples(index=False)
                                ]
                            )
                        )
                    if table_k == "edges":
                        cur.execute(
                            f'CREATE INDEX ON {schema}."{table_v}"("CustomerID")'
                        )
                        cur.execute(
                            f'CREATE INDEX ON {schema}."{table_v}"("ProductID")'
                        )
                    elif table_k == "start":
                        cur.execute(
                            f'CREATE INDEX ON {schema}."{table_v}"("CustomerID")'
                        )
                    elif table_k == "end":
                        cur.execute(
                            f'CREATE INDEX ON {schema}."{table_v}"("ProductID")'
                        )
                cur.execute("COMMIT")
        self.showTime(start_time, sys._getframe().f_code.co_name)

    def getChunks(self, df: pd.DataFrame = None, chunk_size: int = 0) -> pd.DataFrame:
        """
        Get the DataFrame in chunks.

        Args:
            df (DataFrame): The DataFrame to get the edges from.
            chunk_size (int): The size of the chunks to get the edges in.

        Returns:
            DataFrame: chunk of the DataFrame
        """
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i : i + chunk_size].copy()

    async def deleteAllVertices(self, g_client) -> None:
        """
        Delete all vertices

        Args:
            g_client (gremlin_python.driver.client.Client): Gremlin client
        """
        count = 0

        try:
            future_count = g_client.submitAsync("g.V().count()")
            results = await asyncio.wrap_future(future_count)
            count = results.all().result()[0]
        except Exception as e:
            log.info(f"Failed to count all vertices: {e}")

        if count != 0:
            log.info(f"Deleting vertices,: {count}")
            while True:
                try:
                    future_drop = g_client.submitAsync("g.V().drop()")
                    results = await asyncio.wrap_future(future_drop)
                    break
                except Exception as e:
                    log.info(
                        f"Failed to drop all vertices and edges and continued to do...: {e}"
                    )
                    continue

    def executeGremlinQuery(self, g_client, query: str, i) -> None:
        """
        Execute Gremlin query

        Args:
            g_client (gremlin_python.driver.client.Client): Gremlin client
            query (str): Gremlin query
        """
        retries = 0
        INITIAL_WAIT_TIME = 1
        while retries < 5:
            try:
                future = g_client.submitAsync(query)
                results = future.result()
                log.debug(f"results: {results.all().result()}")
                return
            except Exception:
                wait_time = INITIAL_WAIT_TIME * (2**retries)
                log.warning(
                    f"Request rate too large. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
                retries += 1
            finally:
                raise Exception("Max retries exceeded")

    async def loadCSVtoCosmosGremlin(self) -> None:
        """
        Load CSV to Cosmos Gremlin
        """
        # edge are automaticaly located in the same partirion as the 'from' node
        # AVERAGE_SIZE_OF_DOCUMENT includes the estimated size of the edge document
        log.info("Loading CSV to Cosmos DB via Gremlin API")

        from gremlin_python.driver import client, serializer
        import concurrent.futures
        import nest_asyncio

        nest_asyncio.apply()

        try:
            cosmos_gremlin_endpoint = os.environ["COSMOS_GREMLIN_ENDPOINT"]
            cosmos_gremlin_key = os.environ["COSMOS_GREMLIN_KEY"]
        except KeyError:
            print(
                "Please set the environment variables COSMOS_GREMLIN_ENDPOINT / COSMOS_GREMLIN_KEY"
            )
            raise KeyError

        start_time = time.time()
        COSMOS_USERNAME = "/dbs/db1/colls/transaction"
        COSMOS_PKEY = "pk"

        LOGICAL_PARTITION_SIZE = 20 * 1024 * 1024 * 1024  # 20GB
        AVERAGE_SIZE_OF_DOCUMENT = 512  # 512bytes
        NUM_OF_DOCUMENTS_PER_PARTITON = (
            LOGICAL_PARTITION_SIZE // AVERAGE_SIZE_OF_DOCUMENT
        )
        num_of_pk = 1
        MAXIMUM_OPERATOR_DEPTH = 400

        try:
            g_client = client.Client(
                url=cosmos_gremlin_endpoint,
                traversal_source="g",
                username=COSMOS_USERNAME,
                password=cosmos_gremlin_key,
                message_serializer=serializer.GraphSONSerializersV2d0(),
                timeout=600,
            )
        except Exception as e:
            print(f"Failed to connect to Gremlin server: {e}")
            return

        # Can not wait here. 'await' doesn't work.
        # log.info("Dropping all vertices and edges")
        # await self.deleteAllVertices(g_client)

        # With 40,000 R/U
        max_workers = 4

        df = pd.read_csv(self.csv_file)
        df.drop_duplicates(inplace=True)

        columns = {
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

        total_num_of_documents = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for key, cols in columns.items():
                vertices = df[cols].drop_duplicates()
                vertices = vertices.map(
                    lambda x: x.replace("'", r"\'") if isinstance(x, str) else x
                )
                if key == "Customer":
                    tmp_query = """.addV('Customer')
                    .property('Name', '{Name}')
                    .property('CustomerID', '{CustomerID}')
                    .property('Address', '{Address}')
                    .property('Email', '{Email}')
                    .property('Phone', '{Phone}')
                    .property('{pk}', '{num_of_pk}')"""
                elif key == "Product":
                    tmp_query = """.addV('Product')
                    .property('Phrase', '{Phrase}')
                    .property('ProductID', '{ProductID}')
                    .property('SKU', '{SKU}')
                    .property('Price', '{Price}')
                    .property('Color', '{Color}')
                    .property('Size', '{Size}')
                    .property('Weight', '{Weight}')
                    .property('{pk}', '{num_of_pk}')"""
                chunk_size = int(MAXIMUM_OPERATOR_DEPTH / (len(cols) + 2))
                for i, chunk in enumerate(self.getChunks(vertices, chunk_size)):
                    log.info(f"Creating '{key}' {len(chunk)} vertices.")
                    if len(chunk.columns) == 5:
                        query = "g" + "".join(
                            [
                                tmp_query.format(
                                    Name=name,
                                    CustomerID=customerid,
                                    Address=address,
                                    Email=email,
                                    Phone=phone,
                                    pk=COSMOS_PKEY,
                                    num_of_pk=num_of_pk,
                                )
                                for idx, (
                                    customerid,
                                    name,
                                    address,
                                    email,
                                    phone,
                                ) in chunk.iterrows()
                            ]
                        )
                    elif len(chunk.columns) == 7:
                        query = "g" + "".join(
                            [
                                tmp_query.format(
                                    Phrase=phrase,
                                    ProductID=productid,
                                    SKU=sku,
                                    Price=price,
                                    Color=color,
                                    Size=size,
                                    Weight=weight,
                                    pk=COSMOS_PKEY,
                                    num_of_pk=num_of_pk,
                                )
                                for idx, (
                                    productid,
                                    phrase,
                                    sku,
                                    price,
                                    color,
                                    size,
                                    weight,
                                ) in chunk.iterrows()
                            ]
                        )
                    futures.append(
                        executor.submit(self.executeGremlinQuery, g_client, query, i)
                    )
                    total_num_of_documents += len(chunk)
                    if total_num_of_documents % NUM_OF_DOCUMENTS_PER_PARTITON == 0:
                        num_of_pk += 1
            concurrent.futures.wait(futures)

        # With 40,000 R/U
        max_workers = 1024

        # can not avoid cross-partition query when the total size of documents
        #   exceeds the maximum size of logical partition, 20GB,
        #   because actor and film are in different partition
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, row in enumerate(df.itertuples(index=False), start=1):
                log.info(f"Creating 'BOUGHT' edges: {i}")
                query = f"g.V().has('CustomerID', '{row.CustomerID}').addE('BOUGHT').to(g.V().has('ProductID', '{row.ProductID}'))"
                futures.append(
                    executor.submit(self.executeGremlinQuery, g_client, query, i)
                )
            concurrent.futures.wait(futures)

        g_client.close()
        self.showTime(start_time, sys._getframe().f_code.co_name)

    async def convertCSVtoAvro(self) -> None:
        """
        Convert CSV to Avro
        """
        log.info("Converting CSV to Avro")

        import fastavro as fa

        start_time = time.time()
        columns = [
            "id",
            "CustomerID",
            "start_vertex_type",
            "Name",
            "Address",
            "Email",
            "Phone",
            "ProductID",
            "end_vertex_type",
            "Phrase",
            "SKU",
            "Price",
            "Color",
            "Size",
            "Weight",
        ]
        dtype_dict = {
            "id": "int64",
            "CustomerID": "int64",
            "start_vertex_type": "object",
            "Name": "object",
            "Address": "object",
            "Email": "object",
            "Phone": "object",
            "ProductID": "int64",
            "end_vertex_type": "object",
            "Phrase": "object",
            "SKU": "object",
            "Price": "float64",
            "Color": "object",
            "Size": "object",
            "Weight": "int64",
        }
        records = pd.read_csv(self.csv_file, usecols=columns, dtype=dtype_dict).to_dict(
            orient="records"
        )
        schema = {
            "type": "record",
            "name": self.base_file,
            "fields": [
                {"name": "id", "type": "int"},
                {"name": "CustomerID", "type": "int"},
                {"name": "start_vertex_type", "type": "string"},
                {"name": "Name", "type": "string"},
                {"name": "Address", "type": "string"},
                {"name": "Email", "type": "string"},
                {"name": "Phone", "type": "string"},
                {"name": "ProductID", "type": "int"},
                {"name": "end_vertex_type", "type": "string"},
                {"name": "Phrase", "type": "string"},
                {"name": "SKU", "type": "string"},
                {"name": "Price", "type": "float"},
                {"name": "Color", "type": "string"},
                {"name": "Size", "type": "string"},
                {"name": "Weight", "type": "int"},
            ],
        }
        parsed_schema = fa.parse_schema(schema)
        with open(self.avro_file, "wb") as f:
            fa.writer(f, parsed_schema, records)
        self.showTime(start_time, sys._getframe().f_code.co_name)

    async def convertCSVtoParquet(self) -> None:
        """
        Convert CSV to Parquet
        """
        log.info("Converting CSV to Parquet")
        os.system("uv add fastparquet")

        start_time = time.time()
        pd.read_csv(self.csv_file).to_parquet(self.parquet_file)
        self.showTime(start_time, sys._getframe().f_code.co_name)
        os.system("uv remove fastparquet")

    async def convertCSVtoNetworkXPickle(self) -> None:
        """
        Convert CSV to NetworkX and save as pickle
        """
        log.info("Converting CSV to NetworkX and to save as pickle")

        import networkx as nx
        import pickle

        start_time = time.time()
        df = pd.read_csv(self.csv_file)
        starts = df[
            ["CustomerID", "Name", "Address", "Email", "Phone"]
        ].drop_duplicates()
        starts = starts.reset_index(drop=True)
        starts.index += 1
        ends = df[
            ["ProductID", "Phrase", "SKU", "Price", "Color", "Size", "Weight"]
        ].drop_duplicates()
        ends = ends.reset_index(drop=True)
        ends.index += len(starts) + 1

        G = nx.DiGraph()

        [
            G.add_node(
                idx,
                customerid=row["CustomerID"],
                label="Customer",
                name=row["Name"],
                address=row["Address"],
                email=row["Email"],
                phone=row["Phone"],
            )
            for idx, row in starts.iterrows()
        ]
        [
            G.add_node(
                idx,
                productid=row["ProductID"],
                label="Product",
                phrase=row["Phrase"],
                sku=row["SKU"],
                price=row["Price"],
                color=row["Color"],
                size=row["Size"],
                weight=row["Weight"],
            )
            for idx, row in ends.iterrows()
        ]
        [
            G.add_edge(
                starts[starts["CustomerID"] == row["CustomerID"]].index.tolist()[0],
                ends[ends["ProductID"] == row["ProductID"]].index.tolist()[0],
                label="BOUGHT",
            )
            for idx, row in df.iterrows()
        ]

        with open(self.pickle_file, "wb") as f:
            pickle.dump(G, f)

        self.showTime(start_time, sys._getframe().f_code.co_name)


async def main():
    utils = AgeFreighterUtils()
    await utils.loadCSVtoNeo4j()

    await utils.loadCSVtoPGSQL()

    await utils.loadCSVtoCosmosGremlin()

    await utils.convertCSVtoAvro()

    await utils.convertCSVtoParquet()

    await utils.convertCSVtoNetworkXPickle()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
