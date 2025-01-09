# AGEFreighter

a Python package that helps you to create a graph database using Azure Database for PostgreSQL.

[Apache AGE™](https://age.apache.org/) is a PostgreSQL Graph database compatible with PostgreSQL's distributed assets and leverages graph data structures to analyze and use relationships and patterns in data.

[Azure Database for PostgreSQL](https://azure.microsoft.com/en-us/services/postgresql/) is a managed database service that is based on the open-source Postgres database engine.

[Introducing support for Graph data in Azure Database for PostgreSQL (Preview)](https://techcommunity.microsoft.com/blog/adforpostgresql/introducing-support-for-graph-data-in-azure-database-for-postgresql-preview/4275628).

## 0.5.0 Release
Refactored the code to make it more readable and maintainable with the separated classes for factory model.
Please note how to use the new version of the package is tottally different from the previous versions.

### 0.5.2 Release -AzureStorageFreighter-
* AzureStorageFreighter class is used to load data from Azure Storage into the graph database. It's totally different from other classes. The class works as follows:
  * If the argument, 'subscription_id' is not set, the class tries to find the Azure Subscription ID from your local environment using the 'az' command.
  * Creates an Azure Storage account and a blob container under the resource group where the PostgreSQL server runs in.
  * Enables the 'azure_storage' extension in the PostgreSQL server, if it's not enabled.
  * Uploads the CSV file to the blob container.
  * Creates a UDF (User Defined Function) named 'load_from_azure_storage' in the PostgreSQL server. The UDF loads data from the Azure Storage into the graph database.
  * Executes the UDF.
* The above process takes time to prepare for loading data, making it unsuitable for loading small files, but effective for loading large files. For instance, it takes under 3 seconds to load 'actorfilms.csv' after uploading.
* However, please note that it is still in the early stages of implementation, so there is room for optimization and potential issues due to insufficient testing.

### 0.5.3 Release -AzureStorageFreighter-
* AzureStorageFreighter class is totally refactored for better performance and scalability.
  * 0.5.2 didn't work well for large files.
  * Now, it works well for large files.
    Checked with a 5.4GB CSV file consisting of 10M of start vertices, 10K of end vertices, and 25M edges,
    it took 512 seconds to load the data into the graph database with PostgreSQL Flex,
    Standard_D32ds_v4 (32 vcpus, 128 GiB memory) and 512TB / 7500 iops of storage.
  * Tested data was generated with tests/generate_dummy_data.py.
  * UDF to load the data to graph is no longer used.
* However, please note that it is still in the early stages of implementation, so there is room for optimization and potential issues due to insufficient testing.

### 0.6.0 Release
* Added edge properties support.
  * 'edge_props' argument (list) is added to the 'load()' method.
* 'drop_graph' argument is obsoleted. 'create_graph' argument is added.
  * 'create_graph' is set to True by default. CAUTION: If the graph already exists, the graph is dropped before loading the data.
  * If 'create_graph' is False, the data is loaded into the existing graph.

### Features
* Asynchronous connection pool support for psycopg PostgreSQL driver
* 'direct_loading' option for loading data directly into the graph. If 'direct_loading' is True, the data is loaded into the graph using the 'INSERT' statement, not Cypher queries.
* 'COPY' protocol support for loading data into the graph. If 'use_copy' is True, the data is loaded into the graph using the 'COPY' protocol.

### Classes
* AzureStorageFreighter
* AvroFreighter
* CosmosGremlinFreighter
* CSVFreighter
* MultiCSVFreighter
* Neo4jFreighter
* NetworkXFreighter
* ParquetFreighter
* PGFreighter

### Method
All the classes have the same load() method. The method loads data into the graph database.

### Arguments
* Common arguments
  * graph_name (str) : the name of the graph
  * chunk_size (int) : the number of rows to be loaded at once
  * direct_loading (bool) : if True, the data is loaded into the graph using the 'INSERT' statement, not Cypher queries
  * use_copy (bool) : if True, the data is loaded into the graph using the 'COPY' protocol
  * create_graph (bool) : if True, the graph will be created after the existing graph is dropped

* Common arguments for 'Single Source' classes
  * AvroFreighter
  * AzureStorageFreighter
  * CosmosGremlinFreighter
  * Neo4jFreighter
  * NetworkXFreighter
  * ParquetFreighter
  * PGFreighter
    * start_v_label (str): Start Vertex Label
    * start_id (str): Start Vertex ID
    * start_props (list): Start Vertex Properties
    * end_v_label (str): End Vertex Label
    * end_id (str): End Vertex ID
    * end_props (list): End Vertex Properties
    * edge_type (str): Edge Type
    * edge_props (list): Edge Properties

* Class specific arguments
  * AzureStorageFreighter
    * csv (str): The path to the CSV file.

  * AvroFreighter
    * source_avro (str): The path to the Avro file.

  * CosmosGremlinFreighter
    * cosmos_gremlin_endpoint (str): The Cosmos Gremlin endpoint.
    * cosmos_gremlin_key (str): The Cosmos Gremlin key.
    * cosmos_username (str): The Cosmos username.
    * id_map (dict): ID Mapping

  * MultiCSVFreighter
    * vertex_csvs (list): The paths to the vertex CSV files.
    * vertex_labels (list): The labels of the vertices.
    * edge_csvs (list): The paths to the edge CSV files.
    * edge_types (list): The types of the edges.

  * Neo4jFreighter
    * neo4j_uri (str): The URI of the Neo4j database.
    * neo4j_user (str): The username of the Neo4j database.
    * neo4j_password (str): The password of the Neo4j database.
    * neo4j_database (str): The database of the Neo4j database.
    * id_map (dict): ID Mapping

  * NetworkXFreighter
    * networkx_graph (nx.Graph): The NetworkX graph.
    * id_map (dict): ID Mapping

  * ParquetFreighter
    * source_parquet (str): The path to the Parquet file.

  * PGFreighter
    * source_pg_con_string (str): The connection string of the source PostgreSQL database.
    * source_schema (str): The source schema.
    * source_tables (list): The source tables.
    * id_map (dict): ID Mapping


### Release Notes
* 0.4.0 : Added 'loadFromCosmosGremlin()' function.
* 0.4.1 : Changed base Python version to 3.9 to run on Azure Cloud Shell and Databricks 15.4ML.
* 0.4.2 : Tuning for 'loadFromCosmosGremlin()' function.
* 0.4.3 : Standardized the argument names. Enhanced the tests for each functions.
* 0.4.4 : Performance tuning.
* 0.4.5 : Simplified 'loadFromNeo4j'.
* 0.4.6 : Added 'loadFromAvro()' function.
* 0.5.0 : Refactored the code to make it more readable and maintainable with the separated classes for factory model. Introduced concurrent.futures for better performance.
* 0.5.1 : Improved the usage
* 0.5.2 : Added AzureStorageFreighter class, fixed a bug in ParquetFreighter class (THX! Reported from my co-worker, Srikanth-san)
* 0.5.3 : Refactored AzureStorageFreighter class for better performance and scalability.
* 0.6.0 : Added edge properties support. 'drop_graph' argument is obsoleted. 'create_graph' argument is added.
* 0.6.1 : Added 'load_multi()' method to AzureStorageFreighter class.

### Install

```bash
pip install agefreighter
```

### Prerequisites
* over Python 3.9
* This module runs on [psycopg](https://www.psycopg.org/) and [psycopg_pool](https://www.psycopg.org/)
* Enable the Apache AGE extension in your Azure Database for PostgreSQL instance. Login Azure Portal, go to 'server parameters' blade, and check 'AGE" on within 'azure.extensions' and 'shared_preload_libraries' parameters. See, above blog post for more information.
* Load the AGE extension in your PostgreSQL database.

```sql
CREATE EXTENSION IF NOT EXISTS age CASCADE;
```

### File Format for CSVFreighter
CSVFreighter class loads data from a CSV file. The CSV file should have the following format.

```csv
Actor,ActorID,Film,Year,Votes,Rating,FilmID,Role,Time_in_sec
Fred Astaire,nm0000001,Ghost Story,1981,7731,6.3,tt0082449,Hero,3643
Fred Astaire,nm0000001,The Purple Taxi,1977,533,6.6,tt0076851,Hero,270
```

### Usage of CSVFreighter
```python
import asyncio
import os
from agefreighter import Factory
import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    class_name = "CSVFreighter"
    instance = Factory.create_instance(class_name)

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
    )
    await instance.load(
        graph_name="AgeTester",
        start_v_label="Actor",
        start_id="ActorID",
        start_props=["Actor"],
        edge_type="ACTED_IN",
        edge_props=["Role", "Time_in_sec"],
        end_v_label="Film",
        end_id="FilmID",
        end_props=["Film", "Year", "Votes", "Rating"],
        csv="./actorfilms.csv",
        drop_graph=True,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

### File Format for MultiCSVFreighter
MultiCSVFreighter class loads data from multiple CSV files. The CSV files should have the following format.
MultiCSVFreighter class handles all the columns in CSV except 'id' column as properties and all the columns in CSV except 'start_id' / 'start_vertex_type' / 'end_id' / 'end_vertex_type' columns as properties for the edges.

countries.csv
```csv
id,name,iso3,iso2,numeric_code,phone_code,capital,currency,currency_symbol,tld,native,region,subregion,latitude,longitude,emoji,emojiU
2,Aland Islands,ALA,AX,248,+358-18,Mariehamn,EUR,€,.ax,Åland,Europe,Northern Europe,60.116667,19.9,🇦🇽,U+1F1E6 U+1F1FD
3,Albania,ALB,AL,8,355,Tirana,ALL,Lek,.al,Shqipëria,Europe,Southern Europe,41.0,20.0,🇦🇱,U+1F1E6 U+1F1F1
```

cities.csv
```csv
id,name,state_id,state_code,country_id,country_code,latitude,longitude
153,Banaj,629,BR,3,AL,40.82492,19.84074
154,Bashkia Berat,629,BR,3,AL,40.69997,19.94983
```

edges.csv
```csv
start_id,start_vertex_type,end_id,end_vertex_type,Year,Population
153,City,3,Country,1973,12000
154,City,3,Country,1960,35000
```

### Usage of MultiCSVFreighter
```python
import asyncio
import os
from agefreighter import Factory
import logging

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    class_name = "MultiCSVFreighter"
    instance = Factory.create_instance(class_name)

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
    )
    await instance.load(
        graph_name="AgeTester",
        vertex_csvs=["countries.csv", "cities.csv"],
        edge_csvs=["edges.csv"],
        edge_types=["has_city"],
        drop_graph=True,
    )

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

### File Format for AzureStorageFreighter
AzureStorageFreighter class loads data from Azure Storage. It has two methods to load data from Azure Storage, 'load' and 'load_multi'. The 'load' method loads data from a single CSV file, and the 'load_multi' method loads data from multiple CSV files.
  * 'load' expects the exactly same format as CSVFreighter.
  * 'load_multi' expects the exactly same format as MultiCSVFreighter.

See, [tests/agefreightertester.py](https://github.com/rioriost/agefreighter/blob/main/tests/agefreightertester.py) and [docs](https://github.com/rioriost/agefreighter/blob/main/docs/) for more details.

### Test & Samples
```sql
export PG_CONNECTION_STRING="host=your_host.postgres.database.azure.com port=5432 dbname=postgres user=account password=your_password"
cd tests/
python3.9 agefreightertester.py
```

### For more information about [Apache AGE](https://age.apache.org/)
* Apache AGE : https://age.apache.org/
* GitHub : https://github.com/apache/age
* Document : https://age.apache.org/age-manual/master/index.html

### License
MIT License
