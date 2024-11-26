#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

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
        csv="actorfilms.csv",
        start_vertex_type="Actor",
        start_id="ActorID",
        start_properties=["Actor"],
        edge_label="ACTED_IN",
        end_vertex_type="Film",
        end_id="FilmID",
        end_properties=["Film", "Year", "Votes", "Rating"],
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
        vertex_csvs=["countries.csv", "cities.csv"],
        vertex_labels=["Country", "City"],
        edge_csvs=["edges.csv"],
        edge_labels=["has_city"],
        chunk_size=chunk_size,
        direct_loading=direct_loading,
        drop_graph=True,
        use_copy=use_copy,
    )
    print(
        f"test_loadFromCSVs : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


# test for loadFromNetworkx
# create networkx graph from actorfilms.csv
# after creating networkx graph, load it to the database
async def test_loadFromNetworkx(
    af: AgeFreighter,
    chunk_size: int = 96,
    direct_loading: bool = False,
    use_copy: bool = False,
) -> None:
    df = pd.read_csv("actorfilms.csv")
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
        f"test_copyFromNetworkx : time, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: {direct_loading}, use_copy: {use_copy}"
    )


async def main() -> None:
    # export PG_CONNECTION_STRING="host=your_server.postgres.database.azure.com port=5432 dbname=postgres user=account password=your_password"
    try:
        connection_string = os.environ["PG_CONNECTION_STRING"]
    except KeyError:
        print("Please set the environment variable PG_CONNECTION_STRING")
        return

    af = await AgeFreighter.connect(dsn=connection_string, max_connections=64)
    #    ag = await Age.connect(dsn = connection_string, log_level=logging.INFO)
    try:
        # Strongly reccomended to define chunk_size with your data and server before loading large amount of data
        # Especially, the number of properties in the vertex affects the complecity of the query
        # Due to asynchronous nature of the library, the duration for loading data is not linear to the number of rows
        #
        # Addition to the chunk_size, max_wal_size and checkpoint_timeout in the postgresql.conf should be considered

        chunk_size = 128
        # await test_loadFromSingleCSV(af, chunk_size=chunk_size, direct_loading=False)
        # await asyncio.sleep(5)
        await test_loadFromSingleCSV(af, chunk_size=chunk_size, direct_loading=True)
        await asyncio.sleep(5)
        await test_loadFromSingleCSV(af, chunk_size=chunk_size, use_copy=True)
        await asyncio.sleep(5)

        # await test_loadFromCSVs(af, chunk_size=chunk_size, direct_loading=False)
        # await asyncio.sleep(5)
        await test_loadFromCSVs(af, chunk_size=chunk_size, direct_loading=True)
        await asyncio.sleep(5)
        await test_loadFromCSVs(af, chunk_size=chunk_size, use_copy=True)
        await asyncio.sleep(5)

        await test_loadFromNetworkx(af, chunk_size=chunk_size, use_copy=True)

    finally:
        await af.pool.close()


if __name__ == "__main__":
    asyncio.run(main())
