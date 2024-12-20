#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import logging
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AgeFreighterUtils:
    def __init__(self):
        self.name = "AgeUtils"
        self.author = "Rio Fujita"
        self.data_dir = "../data/"
        self.csv_file = f"{self.data_dir}actorfilms2.csv"
        self.avro_file = f"{self.data_dir}actorfilms2.avro"
        self.parquet_file = f"{self.data_dir}actorfilms2.parquet"
        self.pickle_file = f"{self.data_dir}actorfilms2.pickle"

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
        import os
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
        uniq_actors = df[["ActorID", "Actor"]].drop_duplicates()
        uniq_films = df[["FilmID", "Film", "Year", "Votes", "Rating"]].drop_duplicates()

        async with AsyncGraphDatabase.driver(
            n4j_uri, auth=(n4j_user, n4j_password)
        ) as driver:
            async with driver.session() as session:
                # clear the database
                await session.run("MATCH (a)-[r]->() DELETE a, r")
                await session.run("MATCH (a) DELETE a")
                await session.run("DROP INDEX actor_index_id IF EXISTS")
                await session.run("DROP INDEX film_index_id IF EXISTS")
                await session.run(
                    "CREATE INDEX actor_index_id FOR (n:Actor) ON (n.ActorID)"
                )
                await session.run(
                    "CREATE INDEX film_index_id FOR (n:Film) ON (n.FilmID)"
                )
                # create actor nodes
                for idx in range(0, len(uniq_actors), BATCH_SIZE):
                    actors = [
                        {"Actor": actor, "ActorID": actorid}
                        for i, (actor, actorid) in enumerate(
                            zip(
                                uniq_actors["Actor"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_actors["ActorID"][idx : idx + BATCH_SIZE].tolist(),
                            )
                        )
                    ]
                    await session.run(
                        """UNWIND $actors AS row
                        CREATE (a:Actor)
                        SET a += row""",
                        actors=actors,
                    )
                # create film nodes
                for idx in range(0, len(uniq_films), BATCH_SIZE):
                    films = [
                        {
                            "Film": film,
                            "FilmID": filmid,
                            "Year": year,
                            "Votes": votes,
                            "Rating": rating,
                        }
                        for i, (film, filmid, year, votes, rating) in enumerate(
                            zip(
                                uniq_films["Film"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_films["FilmID"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_films["Year"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_films["Votes"][idx : idx + BATCH_SIZE].tolist(),
                                uniq_films["Rating"][idx : idx + BATCH_SIZE].tolist(),
                            )
                        )
                    ]
                    await session.run(
                        """UNWIND $films AS row
                        CREATE (f:Film)
                        SET f += row""",
                        films=films,
                    )
                # create edges
                for idx in range(0, len(df), BATCH_SIZE):
                    acted_ins = [
                        {
                            "from": actorid,
                            "to": filmid,
                            "Genre": genre,
                            "Director": director,
                        }
                        for i, (actorid, filmid, genre, director) in enumerate(
                            zip(
                                df["ActorID"][idx : idx + BATCH_SIZE].tolist(),
                                df["FilmID"][idx : idx + BATCH_SIZE].tolist(),
                                df["Genre"][idx : idx + BATCH_SIZE].tolist(),
                                df["Director"][idx : idx + BATCH_SIZE].tolist(),
                            )
                        )
                    ]
                    await session.run(
                        """UNWIND $acted_ins AS row
                        MATCH (from:Actor {ActorID: row.from})
                        MATCH (to:Film {FilmID: row.to})
                        CREATE (from)-[r:ACTED_IN {Genre:row.genre, Director:row.director}]->(to)
                        SET r += row""",
                        acted_ins=acted_ins,
                    )
        self.showTime(start_time, sys._getframe().f_code.co_name)

    async def loadCSVtoPGSQL(self) -> None:
        """
        Load CSV to PGSQL
        """
        log.info("Loading CSV to PGSQL")
        import os
        import psycopg as pg

        try:
            con_string = os.environ["SRC_PG_CONNECTION_STRING"]
        except KeyError:
            print("Please set the environment variables SRC_PG_CONNECTION_STRING")
            return

        start_time = time.time()
        schema = "public"
        src_tables = {"start": "Actor", "end": "Film", "edges": "ACTED_IN"}

        df = pd.read_csv(self.csv_file)

        datum = [None, None, None]
        types = [None, None, None]

        datum[0] = df[["ActorID", "Actor"]].drop_duplicates()
        datum[0].insert(0, "ActorSerial", range(1, len(datum[0]) + 1))
        types[0] = ["SERIAL", "TEXT", "TEXT"]

        datum[1] = df[["FilmID", "Film", "Year", "Votes", "Rating"]].drop_duplicates()
        datum[1].insert(0, "FilmSerial", range(1, len(datum[1]) + 1))
        types[1] = ["SERIAL", "TEXT", "TEXT", "INT", "INT", "REAL"]

        datum[2] = df[["ActorID", "FilmID", "Genre", "Director"]].rename(
            columns={"ActorID": "start_id", "FilmID": "end_id"}
        )
        datum[2].insert(0, "ActedSerial", range(1, len(datum[2]) + 1))
        types[2] = ["SERIAL", "TEXT", "TEXT", "TEXT", "TEXT"]

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
                        cur.execute(f'CREATE INDEX ON {schema}."{table_v}"(start_id)')
                        cur.execute(f'CREATE INDEX ON {schema}."{table_v}"(end_id)')
                    elif table_k == "start":
                        cur.execute(f'CREATE INDEX ON {schema}."{table_v}"("ActorID")')
                    elif table_k == "end":
                        cur.execute(f'CREATE INDEX ON {schema}."{table_v}"("FilmID")')
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
            except Exception as e:
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
        import os
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
        COSMOS_USERNAME = "/dbs/db1/colls/actorfilms2"
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

        columns = {
            "Actor": ["Actor", "ActorID"],
            "Film": ["Film", "FilmID", "Year", "Votes", "Rating"],
        }

        total_num_of_documents = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for key, cols in columns.items():
                vertices = df[cols].drop_duplicates()
                vertices = vertices.map(
                    lambda x: x.replace("'", r"\'") if isinstance(x, str) else x
                )
                if key == "Actor":
                    tmp_query = """.addV('Actor')
                    .property('Actor', '{actor}')
                    .property('ActorID', '{actorid}')
                    .property('{pk}', '{num_of_pk}')"""
                elif key == "Film":
                    tmp_query = """.addV('Film')
                    .property('Film', '{film}')
                    .property('FilmID', '{filmid}')
                    .property('Year', {year})
                    .property('Votes', {votes})
                    .property('Rating', {rating})
                    .property('{pk}', '{num_of_pk}')"""
                chunk_size = int(MAXIMUM_OPERATOR_DEPTH / (len(cols) + 2))
                for i, chunk in enumerate(self.getChunks(vertices, chunk_size)):
                    log.info(f"Creating '{key}' {len(chunk)} vertices.")
                    if len(chunk.columns) == 2:
                        query = "g" + "".join(
                            [
                                tmp_query.format(
                                    actorid=actorid,
                                    actor=actor,
                                    pk=COSMOS_PKEY,
                                    num_of_pk=num_of_pk,
                                )
                                for idx, (actor, actorid) in chunk.iterrows()
                            ]
                        )
                    elif len(chunk.columns) == 5:
                        query = "g" + "".join(
                            [
                                tmp_query.format(
                                    filmid=filmid,
                                    film=film,
                                    year=year,
                                    votes=votes,
                                    rating=rating,
                                    pk=COSMOS_PKEY,
                                    num_of_pk=num_of_pk,
                                )
                                for idx, (
                                    film,
                                    filmid,
                                    year,
                                    votes,
                                    rating,
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
                log.info(f"Creating 'ACTED_IN' edges: {i}")
                query = f"g.V().has('ActorID', '{row.ActorID}').addE('ACTED_IN').property('Genre', '{row.Genre}').property('Director', '{row.Director}').to(g.V().has('FilmID', '{row.FilmID}'))"
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
        records = pd.read_csv(self.csv_file).to_dict(orient="records")
        schema = {
            "type": "record",
            "name": "actorfilms2",
            "fields": [
                {"name": "ActorID", "type": "string"},
                {"name": "Actor", "type": "string"},
                {"name": "FilmID", "type": "string"},
                {"name": "Film", "type": "string"},
                {"name": "Year", "type": "int"},
                {"name": "Votes", "type": "int"},
                {"name": "Rating", "type": "float"},
                {"name": "Genre", "type": "string"},
                {"name": "Director", "type": "string"},
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

        start_time = time.time()
        pd.read_csv(self.csv_file).to_parquet(self.parquet_file)
        self.showTime(start_time, sys._getframe().f_code.co_name)

    async def convertCSVtoNetworkXPickle(self) -> None:
        """
        Convert CSV to NetworkX and save as pickle
        """
        log.info("Converting CSV to NetworkX and to save as pickle")
        import networkx as nx
        import pickle

        start_time = time.time()
        df = pd.read_csv(self.csv_file)
        G = nx.DiGraph()

        for name, group in df.groupby("ActorID"):
            for idx, row in group.iterrows():
                G.add_node(row["ActorID"], label="Actor", name=row["Actor"])
                G.add_node(
                    row["FilmID"],
                    label="Film",
                    name=row["Film"],
                    year=row["Year"],
                    votes=row["Votes"],
                    rating=row["Rating"],
                )
                G.add_edge(
                    row["ActorID"],
                    row["FilmID"],
                    label="ACTED_IN",
                    genre=row["Genre"],
                    director=row["Director"],
                )
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
