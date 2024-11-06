#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import asyncio
from agefreighter import AgeFreighter

# for environment where PostgreSQL is not capable of loading data from local files, e.g. Azure Database for PostgreSQL

# test1 for loadFromSingleCSV
#
# file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
# actorfilms.csv: Actor,ActorID,Film,Year,Votes,Rating,FilmID
# # of actors: 9,623, # of films: 44,456, # of edges: 191,873
async def test1(af: AgeFreighter, chunk_size: int = 96) -> None:
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
        direct_loading=False
    )
    print(f"test1 : time to loadFromSingleCSV, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: False")

# test2 for directLoadFromSingleCSV
async def test2(af: AgeFreighter, chunk_size: int = 96) -> None:
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
        direct_loading = True
    )
    print(f"test2 : time to loadFromSingleCSV, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: True")

# test3 for loadfromCSVs
# loadFromCSV is a substitute for load_labels_from_file() and load_edges_from_file()
# https://age.apache.org/age-manual/master/intro/agload.html
#
# cities.csv: id,name,state_id,state_code,country_id,country_code,latitude,longitude
# continents.csv: id,name,iso3,iso2,numeric_code,phone_code,capital,currency,currency_symbol,tld,native,region,subregion,latitude,longitude,emoji,emojiU
# edges.csv: start_id,start_vertex_type,end_id,end_vertex_type
# # of countries: 53, # of cities: 72,485, # of edges: 72,485
async def test3(af: AgeFreighter, chunk_size: int = 96) -> None:
    start_time = time.time()
    await af.loadFromCSVs(
        graph_name="cities_countries",
        vertex_csvs=["countries.csv", "cities.csv"],
        vertex_labels=["Country", "City"],
        edge_csvs=["edges.csv"],
        edge_labels=["has_city"],
        chunk_size=chunk_size,
        direct_loading=False
    )
    print(f"test3 : time to loadFromCSVs, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: False")

# test4 for directLoadfromCSVs
async def test4(af: AgeFreighter, chunk_size: int = 96) -> None:
    start_time = time.time()
    await af.loadFromCSVs(
        graph_name="cities_countries",
        vertex_csvs=["countries.csv", "cities.csv"],
        vertex_labels=["Country", "City"],
        edge_csvs=["edges.csv"],
        edge_labels=["has_city"],
        chunk_size=chunk_size,
        direct_loading = True
    )
    print(f"test4 : time to loadFromCSVs, {time.time() - start_time:.2f}, chunk_size: {chunk_size}, direct_loading: True")

async def main() -> None:
    # export PG_CONNECTION_STRING="host=your_server.postgres.database.azure.com port=5432 dbname=postgres user=account password=your_password"
    try:
        connection_string = os.environ["PG_CONNECTION_STRING"]
    except KeyError:
        print("Please set the environment variable PG_CONNECTION_STRING")
        return

    af = await AgeFreighter.connect(dsn = connection_string)
#    ag = await Age.connect(dsn = connection_string, log_level=logging.INFO)
    try:
        # Strongly reccomended to define chunk_size with your data and server before loading large amount of data
        # Especially, the number of properties in the vertex affects the complecity of the query
        # Due to asynchronous nature of the library, the duration for loading data is not linear to the number of rows
        chunk_size = 128
        await test1(af, chunk_size = chunk_size)
        await asyncio.sleep(10)

        await test2(af, chunk_size = chunk_size)
        await asyncio.sleep(10)

        await test3(af, chunk_size = chunk_size)
        await asyncio.sleep(10)

        await test4(af, chunk_size = chunk_size)
    finally:
        await af.pool.close()

if __name__ == "__main__":
    asyncio.run(main())
