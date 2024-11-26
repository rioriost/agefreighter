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
* 'loadFromSingleCSV()' expects a single CSV file that contains the data for the graph.
* 'loadFromCSVs()' expects multiple CSV files. Two CSV files for vertices and one CSV file for edges.
* 'loadFromNetworkx()' expects a NetworkX graph object.

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
See, tests/test_agefreighter.py for more details.

```python
import os
import time

import asyncio
from agefreighter import AgeFreighter

import networkx as nx
import pandas as pd


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
    try:
        chunk_size = 128
        await test_loadFromSingleCSV(af, chunk_size=chunk_size, direct_loading=True)
        await asyncio.sleep(5)
        await test_loadFromSingleCSV(af, chunk_size=chunk_size, use_copy=True)
        await asyncio.sleep(5)

        await test_loadFromCSVs(af, chunk_size=chunk_size, direct_loading=True)
        await asyncio.sleep(5)
        await test_loadFromCSVs(af, chunk_size=chunk_size, use_copy=True)
        await asyncio.sleep(5)

        await test_loadFromNetworkx(af, chunk_size=chunk_size, use_copy=True)

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
