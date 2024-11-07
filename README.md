# AGEFreighter

a Python package that helps you to create a graph database using Azure Database for PostgreSQL.

[Apache AGE™](https://age.apache.org/) is a PostgreSQL Graph database compatible with PostgreSQL's distributed assets and leverages graph data structures to analyze and use relationships and patterns in data.

[Azure Database for PostgreSQL](https://azure.microsoft.com/en-us/services/postgresql/) is a managed database service that is based on the open-source Postgres database engine.

[Introducing support for Graph data in Azure Database for PostgreSQL (Preview)](https://techcommunity.microsoft.com/blog/adforpostgresql/introducing-support-for-graph-data-in-azure-database-for-postgresql-preview/4275628).

### Features
* Asynchronous connection pool support for psycopg PostgreSQL driver
* 'direct_load' option for loading data directly into the graph for better performance

### Installation for test.pypi.org
```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ agefreighter
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
from agefreighter import AgeFreighter
import asyncio
import os

async def test1(af: AgeFreighter, chunk_size: int = 96) -> None:
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

async def test2(af: AgeFreighter, chunk_size: int = 96) -> None:
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

async def main() -> None:
    connection_string = os.environ["PG_CONNECTION_STRING"]
    af = await AgeFreighter.connect(dsn = connection_string)
    try:
        await test1(af, chunk_size = 96)
        await test2(af, chunk_size = 96)
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
