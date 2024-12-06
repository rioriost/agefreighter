# AGEFreighter

a Python package that helps you to create a graph database using Azure Database for PostgreSQL.

[Apache AGE™](https://age.apache.org/) is a PostgreSQL Graph database compatible with PostgreSQL's distributed assets and leverages graph data structures to analyze and use relationships and patterns in data.

[Azure Database for PostgreSQL](https://azure.microsoft.com/en-us/services/postgresql/) is a managed database service that is based on the open-source Postgres database engine.

[Introducing support for Graph data in Azure Database for PostgreSQL (Preview)](https://techcommunity.microsoft.com/blog/adforpostgresql/introducing-support-for-graph-data-in-azure-database-for-postgresql-preview/4275628).

### Features
* Asynchronous connection pool support for psycopg PostgreSQL driver
* 'direct_loading' option for loading data directly into the graph. If 'direct_loading' is True, the data is loaded into the graph using the 'INSERT' statement, not Cypher queries.
* 'COPY' protocol support for loading data into the graph. If 'use_copy' is True, the data is loaded into the graph using the 'COPY' protocol.

### Functions
* 'loadFromSingleCSV()' expects a single CSV file that contains the data for the graph as a source.
* 'loadFromCSVs()' expects multiple CSV files, two CSV files for vertices and one CSV file for edges as sources.
* 'loadFromNetworkx()' expects a NetworkX graph object as a source.
* 'loadFromNeo4j()' expects a Neo4j as a source.
* 'loadFromPGSQL()' expects a PGSQL as a source.
* 'loadFromParquet()' expects a Parquet file as a source.
* 'loadFromCosmosGremlin()' expects a Cosmos Gremlin API as a source.
* Many more coming soon...

### Release Notes
* 0.4.0 : Added 'loadFromCosmosGremlin()' function.

### Install

```bash
pip install agefreighter
```

### Prerequisites
* over Python 3.10
* This module runs on [psycopg](https://www.psycopg.org/) and [psycopg_pool](https://www.psycopg.org/)
* Enable the Apache AGE extension in your Azure Database for PostgreSQL instance. Login Azure Portal, go to 'server parameters' blade, and check 'AGE" on within 'azure.extensions' and 'shared_preload_libraries' parameters. See, above blog post for more information.
* Load the AGE extension in your PostgreSQL database.

```sql
CREATE EXTENSION IF NOT EXISTS age CASCADE;
```

### Usage
See, [tests/test_agefreighter.py](https://github.com/rioriost/agefreighter/blob/6c61f53ec2cf3daf79356690096fee2d18e37631/tests/test_agefreighter.py) for more details.

```python
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
import nest_asyncio

nest_asyncio.apply()


async def test_loadFromSingleCSV(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
) -> None:
    """
    Test for loadFromSingleCSV()
    file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
    actorfilms.csv: Actor,ActorID,Film,Year,Votes,Rating,FilmID
    # of actors: 9,623, # of films: 44,456, # of edges: 191,873
    """
    start_time = time.time()
    await af.loadFromSingleCSV(
        graph_name="ActorFilms",
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


async def test_loadFromCSVs(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
) -> None:
    """
    Test for loadFromCSVs()
    cities.csv: id,name,state_id,state_code,country_id,country_code,latitude,longitude
    countries.csv: id,name,iso3,iso2,numeric_code,phone_code,capital,currency,currency_symbol,tld,native,region,subregion,latitude,longitude,emoji,emojiU
    edges.csv: start_id,start_vertex_type,end_id,end_vertex_type
    # of countries: 53, # of cities: 72,485, # of edges: 72,485
    """
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
    """
    Test for loadFromCSVs()
    start and end vertices are in the same csv file
    """
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


async def test_loadFromNetworkx(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
) -> None:
    """
    Test for loadFromNetworkx()
    create networkx graph from actorfilms.csv
    file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
    after creating networkx graph, load it to the database
    """
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
        graph_name="ActorFilms",
        networkx_graph=G,
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromNetworkx : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


async def test_loadFromNeo4j(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
    init_neo4j: bool = False,
) -> None:
    """
    Test for loadFromNeo4j()
    create networkx graph from actorfilms.csv
    after creating networkx graph, load it to a graph
    """
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
    graph_name = "Actor_Films"
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


async def loadTestDataToNeo4j(
    n4j_uri: str = "",
    n4j_user: str = "",
    n4j_password: str = "",
) -> None:
    """
    Load test data to Neo4j
    file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
    """
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


async def test_loadFromPGSQL(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
    init_pgsql: bool = False,
) -> None:
    """
    Test for loadFromPGSQL()
    create tables from actorfilms.csv
    after creating table, load it to a graph
    """
    try:
        src_connection_string = os.environ["SRC_PG_CONNECTION_STRING"]
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
        id_map={
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


async def loadTestDataToPGSQL(
    con_string: str = "",
    src_tables: dict = {},
    src_csv: str = "",
) -> None:
    """
    Load test data to PGSQL
    file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
    """
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


async def test_loadFromParquet(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
    init_parquet: bool = False,
) -> None:
    """
    Test for loadFromParquet()
    create parquet from actorfilms.csv
    after creating parquet, load it to a graph
    """
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


async def test_loadFromAvro(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
    init_avro: bool = False,
) -> None:
    """
    NOT IMPLEMENTED YET

    Test for loadFromAvro()
    create avro from actorfilms.csv
    after creating avro, load it to a graph
    """
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
    """
    Convert CSV to Avro
    """
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


async def test_loadFromCosmosGremlin(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
    init_gremlin: bool = True,
) -> None:
    """
    NEED MORE TUNING

    Test for loadFromCosmosGremlin()
    create graph via Gremlin from actorfilms.csv
    after creating graph, load it to a graph

    export COSMOS_GREMLIN_ENDPOINT='wss://account_name.gremlin.cosmos.azure.com:443/'
    export COSMOS_GREMLIN_KEY='OwA3fVHzGzs8LsTN...........'
    """
    try:
        cosmos_gremlin_endpoint = os.environ["COSMOS_GREMLIN_ENDPOINT"]
        cosmos_gremlin_key = os.environ["COSMOS_GREMLIN_KEY"]
    except KeyError:
        print(
            "Please set the environment variables COSMOS_GREMLIN_ENDPOINT / COSMOS_GREMLIN_KEY"
        )
        return

    cosmos_db_name = "db1"
    cosmos_graph_name = "actorfilms"
    cosmos_username = f"/dbs/{cosmos_db_name}/colls/{cosmos_graph_name}"
    cosmos_pkey = "pk"

    if init_gremlin:
        await loadTestDataViaGremlin(
            cosmos_gremlin_endpoint=cosmos_gremlin_endpoint,
            cosmos_gremlin_key=cosmos_gremlin_key,
            cosmos_username=cosmos_username,
            cosmos_pkey=cosmos_pkey,
            src_csv="../data/actorfilms.csv",
        )

    start_time = time.time()
    graph_name = "actorfilms"
    await af.loadFromCosmosGremlin(
        cosmos_gremlin_endpoint=cosmos_gremlin_endpoint,
        cosmos_gremlin_key=cosmos_gremlin_key,
        cosmos_username=cosmos_username,
        cosmos_pkey=cosmos_pkey,
        graph_name=graph_name,
        id_map={"Actor": "ActorID", "Film": "FilmID"},
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromCosmosGremlin : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


async def loadTestDataViaGremlin(
    cosmos_gremlin_endpoint: str = "",
    cosmos_gremlin_key: str = "",
    cosmos_username: str = "",
    cosmos_pkey: str = "",
    src_csv: str = "",
) -> None:
    """
    Load test data to Cosmos Gremlin
    file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
    """
    from gremlin_python.driver import client, serializer

    try:
        g_client = client.Client(
            url=cosmos_gremlin_endpoint,
            traversal_source="g",
            username=cosmos_username,
            password=cosmos_gremlin_key,
            message_serializer=serializer.GraphSONSerializersV2d0(),
        )
    except Exception as e:
        print(f"Failed to connect to Gremlin server: {e}")
        return

    # g_client.submitAsync("g.V().drop()")

    df = pd.read_csv(src_csv)
    actors = df[["Actor", "ActorID"]].drop_duplicates()
    actors = actors.map(lambda x: x.replace("'", r"\'") if isinstance(x, str) else x)
    films = df[["Film", "FilmID", "Year", "Votes", "Rating"]].drop_duplicates()
    films = films.map(lambda x: x.replace("'", r"\'") if isinstance(x, str) else x)
    for idx, (actor, actorid) in actors.iterrows():
        query = "g.addV('Actor').property('Actor', '{actor}').property('ActorID', '{actorid}').property('{pk}', '{actorid}')".format(
            actorid=actorid, actor=actor, pk=cosmos_pkey
        )
        # fixed partition key for small data
        # query = "g.addV('Actor').property('Actor', '{actor}').property('ActorID', '{actorid}').property('{pk}', 'pk')".format(
        #    actorid=actorid, actor=actor, pk=cosmos_pkey
        # )
        g_client.submitAsync(query)
    for idx, (film, filmid, year, votes, rating) in films.iterrows():
        query = "g.addV('Film').property('Film', '{film}').property('FilmID', '{filmid}').property('Year', {year}).property('Votes', {votes}).property('Rating', {rating}).property('{pk}', '{filmid}')".format(
            filmid=filmid,
            film=film,
            year=year,
            votes=votes,
            rating=rating,
            pk=cosmos_pkey,
        )
        # fixed partition key for small data
        # query = "g.addV('Film').property('Film', '{film}').property('FilmID', '{filmid}').property('Year', {year}).property('Votes', {votes}).property('Rating', {rating}).property('{pk}', 'pk')".format(
        #    filmid=filmid,
        #    film=film,
        #    year=year,
        #    votes=votes,
        #    rating=rating,
        #    pk=cosmos_pkey,
        # )
        g_client.submitAsync(query)
    # can not avoid cross-partition query when the total size of documents exceeds the maximum size of logical partition, 20GB, because actor and film are in different partition
    for row in df.itertuples(index=False):
        query = "g.V().has('{pk}', '{actorid}').addE('ACTED_IN').to(g.V().has('{pk}', '{filmid}'))".format(
            actorid=row.ActorID, filmid=row.FilmID, pk=cosmos_pkey
        )
        g_client.submitAsync(query)
    g_client.close()


async def main() -> None:
    """
    Test for AgeFreighter

    export PG_CONNECTION_STRING="host=your_server.postgres.database.azure.com port=5432 dbname=postgres user=account password=your_password"

    Strongly reccomend to adjust chunk_size with your data and server before loading large amount of data
    Especially, the number of properties in the vertex affects the complecity of the query
    Due to asynchronous nature of the library, the duration for loading data is not linear to the number of rows

    Addition to the chunk_size, max_wal_size and checkpoint_timeout in the postgresql.conf should be considered
    """
    try:
        connection_string = os.environ["PG_CONNECTION_STRING"]
    except KeyError:
        print("Please set the environment variable PG_CONNECTION_STRING")
        return

    try:
        af = await AgeFreighter.connect(dsn=connection_string, max_connections=64)
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

        # NEED MORE TUNING
        do = True
        if do:
            [
                await test_loadFromCosmosGremlin(
                    af,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                    init_gremlin=False,
                )
                for idx, (direct_loading, use_copy) in enumerate(test_set)
            ]
            print(
                "test_loadFromCosmosGremlin done\n"
                "##### The duration for test_loadFromCosmosGremlin depends on the performance of the source Cosmos DB. #####\n"
            )

        # NOT IMPLEMENTED YET
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

```

### Test & Samples
```sql
export PG_CONNECTION_STRING="host=your_server.postgres.database.azure.com port=5432 dbname=postgres user=account password=your_password"
python3 tests/test_agefreighter.py
```

### For more information about [Apache AGE](https://age.apache.org/)
* Apache AGE : https://age.apache.org/
* GitHub : https://github.com/apache/age
* Document : https://age.apache.org/age-manual/master/index.html

### License
MIT License
