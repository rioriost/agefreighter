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
* common arguments
  * graph_name (str) : the name of the graph
  * chunk_size (int) : the number of rows to be loaded at once
  * direct_loading (bool) : if True, the data is loaded into the graph using the 'INSERT' statement, not Cypher queries
  * use_copy (bool) : if True, the data is loaded into the graph using the 'COPY' protocol
  * drop_graph (bool) : if True, the graph is dropped before loading the data
* 'loadFromSingleCSV()' expects a single CSV file that contains the data for the graph as a source.
  *  start_v_label (str): The label of the start vertex.
  *  start_id (str): The ID of the start vertex.
  *  start_props (list): The properties of the start vertex.
  *  edge_type (str): The type of the edge.
  *  end_v_label (str): The label of the end vertex.
  *  end_id (str): The ID of the end vertex.
  *  end_props (list): The properties of the end vertex.
* 'loadFromCSVs()' expects multiple CSV files, two CSV files for vertices and one CSV file for edges as sources.
  *  vertex_csvs (list): The list of CSV files for vertices.
  *  vertex_labels (list): The list of labels for vertices.
  *  edge_csvs (list): The list of CSV files for edges.
  *  edge_types (list): The list of types for edges.
* 'loadFromNetworkx()' expects a NetworkX graph object as a source.
  * networkx_graph (DiGraph): The NetworkX graph.
  *  graph_name (str): The name of the graph to load the data into.
  *  id_map (dict): The ID map.
* 'loadFromNeo4j()' expects a Neo4j as a source.
  *  uri (str): The URI of the Neo4j server.
  *  user (str): The user name of the Neo4j server.
  *  password (str): The password of the Neo4j server.
  *  neo4j_database (str): The name of the Neo4j database.
  *  id_map (dict): The mapping of the vertex label to the vertex ID.
* 'loadFromPGSQL()' expects a PGSQL as a source.
  *  src_con_string (str): The connection string of the source PostgreSQL database.
  *  src_tables (list): The source tables.
  *  id_map (dict): The ID map.
* 'loadFromParquet()' expects a Parquet file as a source.
  *  src_parquet (str): The source Parquet file.
  *  start_v_label (str): The label of the start vertex.
  *  start_id (str): The ID of the start vertex.
  *  start_props (list): The properties of the start vertex.
  *  edge_type (str): The type of the edge.
  *  end_v_label (str): The label of the end vertex.
  *  end_id (str): The ID of the end vertex.
  *  end_props (list): The properties of the end vertex.
* 'loadFromCosmosGremlin()' expects a Cosmos Gremlin API as a source.
  *  cosmos_gremlin_endpoint (str): The endpoint of the Cosmos Gremlin API.
  *  cosmos_gremlin_key (str): The key of the Cosmos Gremlin API.
  *  cosmos_username (str): The username of the Cosmos Gremlin API.
  *  cosmos_pkey (str): The partition key of the Cosmos Gremlin API.
  *  id_map (dict): The ID map.
* Many more coming soon...

### Release Notes
* 0.4.0 : Added 'loadFromCosmosGremlin()' function.
* 0.4.1 : Changed base Python version to 3.9 to run on Azure Cloud Shell and Databricks 15.4ML.
* 0.4.2 : Tuning for 'loadFromCosmosGremlin()' function.
* 0.4.3 : Standardized the argument names. Enhanced the tests for each functions.
* 0.4.4 : Performance tuning.

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

### Usage
See, [tests/test_agefreighter.py](https://github.com/rioriost/agefreighter/blob/main/tests/test_agefreighter.py) for more details.

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
