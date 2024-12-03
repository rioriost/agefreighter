#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

import asyncio
from agefreighter import AgeFreighter

import networkx as nx
import pandas as pd

# for environment where PostgreSQL is not capable of loading data from local files, e.g. Azure Database for PostgreSQL


# test for loadFromSingleCSV
#
# file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
# actorfilms.csv: Actor,ActorID,Film,Year,Votes,Rating,FilmID
# # of actors: 9,623, # of films: 44,456, # of edges: 191,873
async def test_loadFromSingleCSV(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
) -> None:
    start_time = time.time()
    await af.loadFromSingleCSV(
        graph_name="actorfilms",
        csv="../data/actorfilms.csv",
        start_v_label="Actor",
        start_id="ActorID",
        start_props=["Actor"],
        edge_type="ACTED_IN",
        end_v_label="Film",
        end_id="FilmID",
        end_props=["Film", "Year", "Votes", "Rating"],
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromSingleCSV : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


# test for loadfromCSVs
#
# cities.csv: id,name,state_id,state_code,country_id,country_code,latitude,longitude
# countries.csv: id,name,iso3,iso2,numeric_code,phone_code,capital,currency,currency_symbol,tld,native,region,subregion,latitude,longitude,emoji,emojiU
# edges.csv: start_id,start_vertex_type,end_id,end_vertex_type
# # of countries: 53, # of cities: 72,485, # of edges: 72,485
async def test_loadFromCSVs(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
) -> None:
    start_time = time.time()
    await af.loadFromCSVs(
        graph_name="cities_countries",
        vertex_csvs=["../data/countries.csv", "../data/cities.csv"],
        v_labels=["Country", "City"],
        edge_csvs=["../data/edges.csv"],
        e_types=["has_city"],
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromCSVs : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


async def test_loadFrom2CSVs(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
) -> None:
    start_time = time.time()
    await af.loadFromCSVs(
        graph_name="war_btw_countries",
        vertex_csvs=["../data/countries.csv"],
        v_labels=["Country"],
        edge_csvs=["../data/fight_with.csv"],
        e_types=["FIGHT_WITH"],
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFrom2CSVs : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


# test for loadFromNetworkx
# create networkx graph from actorfilms.csv
# file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
# after creating networkx graph, load it to the database
async def test_loadFromNetworkx(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
) -> None:
    df = pd.read_csv("../data/actorfilms.csv")
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
            G.add_edge(row["ActorID"], row["FilmID"], label="ACTED_IN")

    start_time = time.time()
    await af.loadFromNetworkx(
        graph_name="actorfilms",
        networkx_graph=G,
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromNetworkx : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


# test for loadFromNeo4j
# create networkx graph from actorfilms.csv
# after creating networkx graph, load it to a graph
async def test_loadFromNeo4j(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
    init_neo4j: bool = False,
) -> None:
    try:
        n4j_uri = os.environ["NEO4J_URI"]
        n4j_user = os.environ["NEO4J_USER"]
        n4j_password = os.environ["NEO4J_PASSWORD"]
    except KeyError:
        print(
            "Please set the environment variables NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"
        )
        return

    # prepare test data for neo4j
    if init_neo4j:
        await loadTestDataToNeo4j(n4j_uri, n4j_user, n4j_password)

    start_time = time.time()
    graph_name = "actorfilms"
    await af.loadFromNeo4j(
        uri=n4j_uri,
        user=n4j_user,
        password=n4j_password,
        neo4j_database="neo4j",
        graph_name=graph_name,
        id_map={"Actor": "ActorID", "Film": "FilmID"},
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromNeo4j : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


# load test data to neo4j
# file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
async def loadTestDataToNeo4j(
    n4j_uri: str = "",
    n4j_user: str = "",
    n4j_password: str = "",
) -> None:
    from neo4j import AsyncGraphDatabase

    batch_size = 1000
    df = pd.read_csv("../data/actorfilms.csv")
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
            await session.run("CREATE INDEX film_index_id FOR (n:Film) ON (n.FilmID)")
            # create actor nodes
            for idx in range(0, len(uniq_actors), batch_size):
                actors = [
                    {"Actor": actor, "ActorID": actorid}
                    for i, (actor, actorid) in enumerate(
                        zip(
                            uniq_actors["Actor"][idx : idx + batch_size].tolist(),
                            uniq_actors["ActorID"][idx : idx + batch_size].tolist(),
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
            for idx in range(0, len(uniq_films), batch_size):
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
                            uniq_films["Film"][idx : idx + batch_size].tolist(),
                            uniq_films["FilmID"][idx : idx + batch_size].tolist(),
                            uniq_films["Year"][idx : idx + batch_size].tolist(),
                            uniq_films["Votes"][idx : idx + batch_size].tolist(),
                            uniq_films["Rating"][idx : idx + batch_size].tolist(),
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
            for idx in range(0, len(df), batch_size):
                acted_ins = [
                    {"from": actorid, "to": filmid}
                    for i, (actorid, filmid) in enumerate(
                        zip(
                            df["ActorID"][idx : idx + batch_size].tolist(),
                            df["FilmID"][idx : idx + batch_size].tolist(),
                        )
                    )
                ]
                await session.run(
                    """UNWIND $acted_ins AS row
                    MATCH (from:Actor {ActorID: row.from})
                    MATCH (to:Film {FilmID: row.to})
                    CREATE (from)-[r:ACTED_IN]->(to)
                    SET r += row""",
                    acted_ins=acted_ins,
                )


# test for loadFromPGSQL
# create tables from actorfilms.csv
# after creating table, load it to a graph
async def test_loadFromPGSQL(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
    init_pgsql: bool = False,
) -> None:
    try:
        src_connection_string = os.environ["PG_CONNECTION_STRING"]
    #    src_connection_string = os.environ["SRC_PG_CONNECTION_STRING"]
    except KeyError:
        print("Please set the environment variables SRC_PG_CONNECTION_STRING")
        return

    src_tables = {"from_nodes": "Actor", "to_nodes": "Film", "edges": "ACTED_IN"}

    if init_pgsql:
        # prepare test data for pgsql
        await loadTestDataToPGSQL(
            con_string=src_connection_string,
            src_tables=src_tables,
            src_csv="../data/actorfilms.csv",
        )

    start_time = time.time()
    graph_name = "actorfilms"
    await af.loadFromPGSQL(
        src_con_string=src_connection_string,
        src_tables=src_tables,
        graph_name=graph_name,
        # values are culumn name with small caps
        id_maps={
            "Actor": "actorid",
            "Film": "filmid",
        },
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromPGSQL : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


# load test data to PGSQL
# file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
async def loadTestDataToPGSQL(
    con_string: str = "",
    src_tables: dict = {},
    src_csv: str = "",
) -> None:
    import psycopg as pg

    df = pd.read_csv(src_csv)

    datum = [None, None, None]
    types = [None, None, None]

    datum[0] = df[["ActorID", "Actor"]].drop_duplicates()
    datum[0].insert(0, "serial", range(1, len(datum[0]) + 1))
    types[0] = ["SERIAL", "TEXT", "TEXT"]

    datum[1] = df[["FilmID", "Film", "Year", "Votes", "Rating"]].drop_duplicates()
    datum[1].insert(0, "serial", range(1, len(datum[1]) + 1))
    types[1] = ["SERIAL", "TEXT", "TEXT", "INT", "INT", "REAL"]

    datum[2] = df[["ActorID", "FilmID"]].rename(
        columns={"ActorID": "start_id", "FilmID": "end_id"}
    )
    datum[2].insert(0, "serial", range(1, len(datum[2]) + 1))
    types[2] = ["SERIAL", "TEXT", "TEXT"]

    with pg.connect(con_string) as conn:
        with conn.cursor() as cur:
            for idx, (table, data, type) in enumerate(
                zip(src_tables.values(), datum, types)
            ):
                cur.execute(f"DROP TABLE IF EXISTS {table}")
                cols = ",".join(
                    [
                        col + " " + tp
                        for _, (col, tp) in enumerate(zip(data.columns, type))
                    ]
                )
                cur.execute(f"CREATE TABLE {table} ({cols})")
                query = f"COPY {table} FROM STDIN (FORMAT TEXT, FREEZE)"
                with cur.copy(query) as copy:
                    copy.write(
                        "\n".join(
                            [
                                "\t".join(map(str, row))
                                for row in data.itertuples(index=False)
                            ]
                        )
                    )
            cur.execute("COMMIT")


# test for loadFromParquet
# create parquet from actorfilms.csv
# after creating parquet, load it to a graph
async def test_loadFromParquet(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
    init_parquet: bool = False,
) -> None:
    src_parquet = "../data/actorfilms.parquet"

    if init_parquet:
        pd.read_csv("../data/actorfilms.csv").to_parquet(src_parquet)

    start_time = time.time()
    graph_name = "actorfilms"
    await af.loadFromParquet(
        src_parquet=src_parquet,
        graph_name=graph_name,
        start_v_label="Actor",
        start_id="ActorID",
        start_props=["Actor"],
        edge_type="ACTED_IN",
        end_v_label="Film",
        end_id="FilmID",
        end_props=["Film", "Year", "Votes", "Rating"],
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromParquet : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


# test for loadFromAvro
# create avro from actorfilms.csv
# after creating avro, load it to a graph
async def test_loadFromAvro(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
    init_avro: bool = False,
) -> None:
    src_avro = "../data/actorfilms.avro"

    if init_avro:
        await convertCSVtoAvro(src_csv="../data/actorfilms.csv", tgt_avro=src_avro)

    start_time = time.time()
    graph_name = "actorfilms"
    await af.loadFromAvro(
        src_avro=src_avro,
        graph_name=graph_name,
        start_v_label="Actor",
        start_id="ActorID",
        start_props=["Actor"],
        edge_type="ACTED_IN",
        end_v_label="Film",
        end_id="FilmID",
        end_props=["Film", "Year", "Votes", "Rating"],
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromAvro : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


async def convertCSVtoAvro(src_csv: str = "", tgt_avro: str = "") -> None:
    import fastavro as fa

    records = pd.read_csv(src_csv).to_dict(orient="records")
    schema = {
        "type": "record",
        "name": "actorfilms",
        "fields": [
            {"name": "ActorID", "type": "string"},
            {"name": "Actor", "type": "string"},
            {"name": "FilmID", "type": "string"},
            {"name": "Film", "type": "string"},
            {"name": "Year", "type": "int"},
            {"name": "Votes", "type": "int"},
            {"name": "Rating", "type": "float"},
        ],
    }
    parsed_schema = fa.parse_schema(schema)
    with open(tgt_avro, "wb") as f:
        fa.writer(f, parsed_schema, records)


async def main() -> None:
    # export PG_CONNECTION_STRING="host=your_server.postgres.database.azure.com port=5432 dbname=postgres user=account password=your_password"
    try:
        connection_string = os.environ["PG_CONNECTION_STRING"]
    except KeyError:
        print("Please set the environment variable PG_CONNECTION_STRING")
        return

    try:
        af = await AgeFreighter.connect(dsn=connection_string, max_connections=64)
        # Strongly reccomended to define chunk_size with your data and server before loading large amount of data
        # Especially, the number of properties in the vertex affects the complecity of the query
        # Due to asynchronous nature of the library, the duration for loading data is not linear to the number of rows
        #
        # Addition to the chunk_size, max_wal_size and checkpoint_timeout in the postgresql.conf should be considered

        test_set = [
            [False, False],
            [True, False],
            [False, True],
        ]

        chunk_size = 128
        do = True
        if do:
            [
                await test_loadFromSingleCSV(
                    af,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                )
                for idx, (direct_loading, use_copy) in enumerate(test_set)
            ]
            print("test_loadFromSingleCSV done\n")

        do = True
        if do:
            [
                await test_loadFromCSVs(
                    af,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                )
                for idx, (direct_loading, use_copy) in enumerate(test_set)
            ]
            print("test_loadFromCSVs done\n")

        do = True
        if do:
            [
                await test_loadFromNetworkx(
                    af,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                )
                for idx, (direct_loading, use_copy) in enumerate(test_set)
            ]
            print("test_loadFromNetworkx done\n")

        do = True
        if do:
            [
                await test_loadFromNeo4j(
                    af,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                    init_neo4j=True,
                )
                for idx, (direct_loading, use_copy) in enumerate(test_set)
            ]
            print(
                "test_loadFromNeo4j done\n"
                "##### The duration for test_loadFromNeo4j depends on the performance of the neo4j server. #####\n"
            )

        do = True
        if do:
            [
                await test_loadFromPGSQL(
                    af,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                    init_pgsql=True,
                )
                for idx, (direct_loading, use_copy) in enumerate(test_set)
            ]
            print(
                "test_loadFromPGSQL done\n"
                "##### The duration for test_loadFromPGSQL depends on the performance of the source pgsql server. #####\n"
            )

        do = True
        if do:
            [
                await test_loadFromParquet(
                    af,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                    init_parquet=True,
                )
                for idx, (direct_loading, use_copy) in enumerate(test_set)
            ]
            print("test_loadFromParquet done\n")

        # not implemented yet
        do = False
        if do:
            [
                await test_loadFromAvro(
                    af,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                    init_avro=True,
                )
                for idx, (direct_loading, use_copy) in enumerate(test_set)
            ]
            print("test_loadFromAvro done\n")

    finally:
        await af.pool.close()


if __name__ == "__main__":
    asyncio.run(main())
