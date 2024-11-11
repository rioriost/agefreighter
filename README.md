# AGEFreighter

a Python package that helps you to create a graph database using Azure Database for PostgreSQL.

[Apache AGE™](https://age.apache.org/) is a PostgreSQL Graph database compatible with PostgreSQL's distributed assets and leverages graph data structures to analyze and use relationships and patterns in data.

[Azure Database for PostgreSQL](https://azure.microsoft.com/en-us/services/postgresql/) is a managed database service that is based on the open-source Postgres database engine.

[Introducing support for Graph data in Azure Database for PostgreSQL (Preview)](https://techcommunity.microsoft.com/blog/adforpostgresql/introducing-support-for-graph-data-in-azure-database-for-postgresql-preview/4275628).

### Features
* Asynchronous connection pool support for psycopg PostgreSQL driver
* 'direct_load' option for loading data directly into the graph for better performance
* 'COPY' protocol support for loading data into the graph for much better performance

### Install

```bash
pip install agefreighter
```

### Prerequisites
* over Python 3.11
* This module runs on [psycopg](https://www.psycopg.org/) and [psycopg_pool](https://www.psycopg.org/)
* Enable the Apache AGE extension in your Azure Database for PostgreSQL instance. Login Azure Portal, go to 'server parameters' blade, and check 'AGE" on within 'azure.extensions' and 'shared_preload_libraries' parameters. See, above blog post for more information.
* Load the AGE extension in your PostgreSQL database.

```sql
CREATE EXTENSION IF NOT EXISTS age CASCADE;
```

### Usage

```python
import os

import asyncio
from agefreighter import AgeFreighter

# file downloaded from https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors
# actorfilms.csv: Actor,ActorID,Film,Year,Votes,Rating,FilmID
# # of actors: 9,623, # of films: 44,456, # of edges: 191,873
async def test_loadFromSingleCSV(af: AgeFreighter, chunk_size: int = 96, direct_loading: bool = False) -> None:
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
        direct_loading = direct_loading,
        drop_graph = True
    )

# cities.csv: id,name,state_id,state_code,country_id,country_code,latitude,longitude
# continents.csv: id,name,iso3,iso2,numeric_code,phone_code,capital,currency,currency_symbol,tld,native,region,subregion,latitude,longitude,emoji,emojiU
# edges.csv: start_id,start_vertex_type,end_id,end_vertex_type
# # of countries: 53, # of cities: 72,485, # of edges: 72,485
async def test_loadFromCSVs(af: AgeFreighter, chunk_size: int = 96, direct_loading: bool = False) -> None:
    await af.loadFromCSVs(
        graph_name="cities_countries",
        vertex_csvs=["countries.csv", "cities.csv"],
        vertex_labels=["Country", "City"],
        edge_csvs=["edges.csv"],
        edge_labels=["has_city"],
        chunk_size=chunk_size,
        direct_loading = direct_loading,
        drop_graph = True
    )

async def test_copyFromSingleCSV(af: AgeFreighter, chunk_size: int = 96) -> None:
    start_time = time.time()
    await af.copyFromSingleCSV(
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
        drop_graph = True
    )

async def test_copyFromCSVs(af: AgeFreighter, chunk_size: int = 96) -> None:
    start_time = time.time()
    await af.copyFromCSVs(
        graph_name="cities_countries",
        vertex_csvs=["countries.csv", "cities.csv"],
        vertex_labels=["Country", "City"],
        edge_csvs=["edges.csv"],
        edge_labels=["has_city"],
        chunk_size=chunk_size,
        drop_graph = True
    )

async def main() -> None:
    # export PG_CONNECTION_STRING="host=your_server.postgres.database.azure.com port=5432 dbname=postgres user=account password=your_password"
    try:
        connection_string = os.environ["PG_CONNECTION_STRING"]
    except KeyError:
        print("Please set the environment variable PG_CONNECTION_STRING")
        return

    af = await AgeFreighter.connect(dsn = connection_string, max_connections = 64)
    try:
        # Strongly reccomended to define chunk_size with your data and server before loading large amount of data
        # Especially, the number of properties in the vertex affects the complecity of the query
        # Due to asynchronous nature of the library, the duration for loading data is not linear to the number of rows
        #
        # Addition to the chunk_size, max_wal_size and checkpoint_timeout in the postgresql.conf should be considered
        chunk_size = 64
        await test_loadFromSingleCSV(af, chunk_size = chunk_size, direct_loading = False)
        await asyncio.sleep(10)
        await test_loadFromSingleCSV(af, chunk_size = chunk_size, direct_loading = True)
        await asyncio.sleep(10)
        await test_copyFromSingleCSV(af, chunk_size = chunk_size)
        await asyncio.sleep(10)

        await test_loadFromCSVs(af, chunk_size = chunk_size, direct_loading = False)
        await asyncio.sleep(10)
        await test_loadFromCSVs(af, chunk_size = chunk_size, direct_loading = True)
        await asyncio.sleep(10)
        await test_copyFromCSVs(af, chunk_size = chunk_size)
        await asyncio.sleep(10)

    finally:
        await af.pool.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Test & Samples
```
export PG_CONNECTION_STRING="host=your_server.postgres.database.azure.com port=5432 dbname=postgres user=account password=your_password"
python3 tests/test_agefreighter.py
```

### For more information about [Apache AGE](https://age.apache.org/)
* Apache AGE : https://age.apache.org/
* GitHub : https://github.com/apache/age
* Document : https://age.apache.org/age-manual/master/index.html

### License
MIT License
