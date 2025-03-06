# AGEFreighter

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.13%2B-blue)

a Python package that helps you to create a graph database using Azure Database for PostgreSQL.

[Apache AGE™](https://age.apache.org/) is a PostgreSQL Graph database compatible with PostgreSQL's distributed assets and leverages graph data structures to analyze and use relationships and patterns in data.

[Azure Database for PostgreSQL](https://azure.microsoft.com/en-us/services/postgresql/) is a managed database service that is based on the open-source Postgres database engine.

[Introducing support for Graph data in Azure Database for PostgreSQL (Preview)](https://techcommunity.microsoft.com/blog/adforpostgresql/introducing-support-for-graph-data-in-azure-database-for-postgresql-preview/4275628).

## Table of Contents

- [Version1.0](#version1.0)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Usage](#usage)
- [How to edit the CSV files to load them to the graph database with PGFreighter](#how-to-edit-the-csv-files-to-load-them-to-the-graph-database-with-pgfreighter)
- [Release Notes](#release-notes)
- [Known Issues](#known-issues)
- [For More Information](#for-more-information)
- [License](#license)

## Version1.0

AGEFreighter 1.0 is totally refactored as a CLI tool for loading data into Apache AGE and viewing graph data in Apache AGE.

## Features

- Asynchronous connection pool support for psycopg PostgreSQL driver
- COPY protocol support for loading data into the graph.
- On macOS / Linux, 'tab' completion is available.
- On demand Python package installation to reduce installation time.

```bash
agefreighter load
required module 'neo4j' is not installed. Install it? [Y/n]: Y
pip module is not available. Trying with uv...
```

### Subcommands

- 'load' subcommand to load CSV files, graph data in Neo4j, Cosmos DB for NoSQL into Apache AGE.
- 'view' subcommand to view graph data in Apache AGE.
- 'parse' subcommand to parse a Cypher query.
- 'generate' subcommand to generate dummy graph data.
- 'convert' subcommand to convert Gremlin queries to Cypher queries.
- 'prepare' subcommand to prepare environment for testing AGEFreighter.
- 'completion' subcommand to show completion instructions.

## Prerequisites

- Python 3.13 and above
- This module runs on [psycopg](https://www.psycopg.org/) and [psycopg_pool](https://www.psycopg.org/)
- Enable the Apache AGE extension in your Azure Database for PostgreSQL instance. Login Azure Portal, go to 'server parameters' blade, and check 'AGE" on within 'azure.extensions' and 'shared_preload_libraries' parameters. See, above blog post for more information.
- Load the AGE extension in your PostgreSQL database.

```sql
CREATE EXTENSION IF NOT EXISTS age CASCADE;
```

## Install

- with brew

```bash
brew tap rioriost/agefreighter https://github.com/rioriost/agefreighter.git
brew install agefreighter
```

- with uv

```bash
uv init your_project
cd your_project
uv venv
source .venv/bin/activate
uv add agefreighter
```

- with python venv on macOS / Linux

```bash
mkdir your_project
cd your_project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install agefreighter
```

- with python venv on Windows

```bash
mkdir your_project
cd your_project
python -m venv venv
.\venv\Scripts\activate
python -m pip install agefreighter
```

## Usage

```bash
agefreighter --help
usage: agefreighter [-h] [--graphname GRAPHNAME] [--pg-con-str PG_CON_STR] [--pg-min-connections PG_MIN_CONNECTIONS] [--pg-max-connections PG_MAX_CONNECTIONS] [--debug]
                    {completion,load,view,parse,generate,convert,prepare} ...

AGEFreighter, a tool to export data from various sources and load it into Apache AGE.

positional arguments:
  {completion,load,view,parse,generate,convert,prepare}
    completion          Show Completion Instructions
    load                Load data into Apache AGE
    view                View data in Apache AGE
    parse               Parse a cypher query
    generate            Generate dummy data
    convert             Convert Gremlin queries to Cypher queries.
    prepare             Prepare data for testing AGEFreighter.

options:
  -h, --help            show this help message and exit
  --graphname GRAPHNAME
                        Name of a new graph on Apache AGE
  --pg-con-str PG_CON_STR
                        Connection string of the Azure Database for PostgreSQL
  --pg-min-connections PG_MIN_CONNECTIONS
                        Minimum number of connections to PostgreSQL
  --pg-max-connections PG_MAX_CONNECTIONS
                        Maximum number of connections to PostgreSQL
  --debug               Enable debug logging
```

Each subcommand has its own set of options.

```bash
agefreighter load --help
usage: agefreighter load [-h] [--source-type {neo4j,cosmosdb,pgsql,csv}] [--trial] [--save-temps] [--chunk-size CHUNK_SIZE] [--progress] [--config CONFIG] [--neo4j-uri NEO4J_URI] [--neo4j-user NEO4J_USER]
                         [--neo4j-password NEO4J_PASSWORD] [--neo4j-database NEO4J_DATABASE] [--cosmos-endpoint COSMOS_ENDPOINT] [--cosmos-key COSMOS_KEY] [--cosmos-database COSMOS_DATABASE]
                         [--cosmos-container COSMOS_CONTAINER] [--src-pg-con-str SRC_PG_CON_STR]

options:
  -h, --help            show this help message and exit
  --source-type {neo4j,cosmosdb,pgsql,csv}
                        Source type of the graph data
  --trial               Extract only 100 edges per relationship type
  --save-temps          Save data from source as CSV files into a directory
  --chunk-size CHUNK_SIZE
                        Chunk size for exporting data
  --progress            Show progress
  --config CONFIG       Path to the configuration file
  --neo4j-uri NEO4J_URI
                        Neo4j URI
  --neo4j-user NEO4J_USER
                        Neo4j username
  --neo4j-password NEO4J_PASSWORD
                        Neo4j password
  --neo4j-database NEO4J_DATABASE
                        Neo4j database
  --cosmos-endpoint COSMOS_ENDPOINT
                        Cosmos endpoint
  --cosmos-key COSMOS_KEY
                        Cosmos key
  --cosmos-database COSMOS_DATABASE
                        Cosmos database
  --cosmos-container COSMOS_CONTAINER
                        Cosmos container
  --src-pg-con-str SRC_PG_CON_STR
                        Source PostgreSQL connection string
```

### Load from Neo4j

To load graph data from Neo4j, use the following command:

```bash
agefreighter --pg-con-str "host=YOUR_SERVER.postgres.database.azure.com port=5432 dbname=postgres user=YOUR_USERNAME password=YOUR_PASSWORD" load --neo4j-uri neo4j://localhost:7687 --neo4j-user neo4j --neo4j-password password
```

Or, you can use the environment variables:

- macOS / Linux

```bash
export PG_CONNECTION_STRING="host=YOUR_SERVER.postgres.database.azure.com port=5432 dbname=postgres user=YOUR_USERNAME password=YOUR_PASSWORD"
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

- Windows (cmd.exe)

```bash
set PG_CONNECTION_STRING=host=YOUR_SERVER.postgres.database.azure.com port=5432 dbname=postgres user=YOUR_USERNAME password=YOUR_PASSWORD
set NEO4J_URI=neo4j://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=password
```

- PowerShell

```bash
$env:PG_CONNECTION_STRING="host=YOUR_SERVER.postgres.database.azure.com port=5432 dbname=postgres user=YOUR_USERNAME password=YOUR_PASSWORD"
$env:NEO4J_URI="neo4j://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="password"
```

If all the required environment variables are set, you can just run:

```bash
agefreighter load
```

### Load from CSV Files

To load data from CSV files, you need to prepare a configuration JSON file that specifies how to load the CSV files into your PostgreSQL.

(supposed PG_CONNECTION_STRING is set)
```bash
agefreighter load --source-type csv --config config.json
```

We have 4 structure patterns of nodes and edges.

1. Single kind of nodes and single kind of edges:
```
(AirPort) - [AirRoute] -> (AirPort)
```

This pattern is described in JSON as follows [docs/configs/single_edge_single_node.json](https://github.com/rioriost/agefreighter/raw/main/docs/configs/single_edge_single_node.json):
```json
{
  "edge": {
    "csv_path": "data/airroute/airroute_airport_airport.csv",
    "type": "AirRoute",
    "props": ["distance"],
    "start_vertex": {
      "csv_path": "data/airroute/airroute_airport_airport.csv",
      "id": "start_id",
      "label": "AirPort",
      "props": []
    },
    "end_vertex": {
      "csv_path": "data/airroute/airroute_airport_airport.csv",
      "id": "end_id",
      "label": "AirPort",
      "props": []
    }
  }
}
```

And the contents of the CSV files [data/airroute/airroute_airport_airport.csv](https://github.com/rioriost/agefreighter/raw/main/data/airroute/airroute_airport_airport.csv):

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type","distance"
"1","1388","AirPort","794","AirPort","1373"
"2","2998","AirPort","823","AirPort","11833"
"3","2058","AirPort","2423","AirPort","2180"
"4","794","AirPort","2868","AirPort","880"
```

2. Single kind of nodes and multiple kinds of edges:
```
(Person) - [KNOWS | LIKES] -> (Person)
```

This pattern is described in JSON as follows [docs/configs/multiple_edges_single_node.json](https://github.com/rioriost/agefreighter/raw/main/docs/configs/multiple_edges_single_node.json):
```json
{
  "edge": [
    {
      "csv_path": "data/persons/knows_person_person.csv",
      "type": "KNOWS",
      "props": ["since"],
      "vertex": {
        "csv_path": "data/persons/person.csv",
        "id": "id",
        "label": "Person",
        "props": ["Name"]
      }
    },
    {
      "csv_path": "data/persons/likes_person_person.csv",
      "type": "LIKES",
      "props": ["since"],
      "vertex": {
        "csv_path": "data/persons/person.csv",
        "id": "id",
        "label": "Person",
        "props": ["Name"]
      }
    }
  ]
}
```

And the contents of the CSV files:

[data/persons/person.csv](https://github.com/rioriost/agefreighter/raw/main/data/persons/person.csv)

```csv
"id","Name"
"1","Donna Smith"
"2","Kristin Villanueva"
"3","Margaret Jackson"
"4","Emily Morrison"
```

[data/persons/knows_person_person.csv](https://github.com/rioriost/agefreighter/raw/main/data/persons/knows_person_person.csv)

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type"
"1","247","Person","818","Person"
"2","609","Person","62","Person"
"3","687","Person","573","Person"
"4","975","Person","963","Person"
```

[data/persons/likes_person_person.csv](https://github.com/rioriost/agefreighter/raw/main/data/persons/likes_person_person.csv)

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type"
"1","566","Person","892","Person"
"2","263","Person","409","Person"
"3","637","Person","788","Person"
"4","454","Person","487","Person"
```

3. Multiple kinds of nodes and single kind of edges:
```
(Country) - [has] -> (City)
```

This pattern is described in JSON as follows [docs/configs/single_edge_multiple_nodes.json](https://github.com/rioriost/agefreighter/raw/main/docs/configs/single_edge_multiple_nodes.json):
```json
{
  "edge": {
    "csv_path": "data/countries/has_country_city.csv",
    "type": "has",
    "props": ["since"],
    "start_vertex": {
      "csv_path": "data/countries/country.csv",
      "id": "id",
      "label": "Country",
      "props": ["Name", "Capital", "Population", "ISO", "TLD", "FlagURL"]
    },
    "end_vertex": {
      "csv_path": "data/countries/city.csv",
      "id": "id",
      "label": "City",
      "props": ["Name", "Latitude", "Longitude"]
    }
  }
}
```

And the contents of the CSV files:

[data/countries/country.csv](https://github.com/rioriost/agefreighter/raw/main/data/countries/country.csv)

```csv
""id"",""Name"",""Capital"",""Population"",""ISO"",""TLD"",""FlagURL""
""1"",""El	Salvador"",""Kristybury"",""355169921"",""TN"",""wxu"",""https://dummyimage.com/777x133""
""2"",""Lebanon "for" Special Character testing"",""New William"",""413929227"",""UK"",""akj"",""https://picsum.photos/772/459""
""3"",""Pakistan"",""Justinstad"",""568781337"",""PG"",""rqv"",""https://dummyimage.com/245x330""
""4"",""Bahamas"",""Williamland"",""115914464"",""TD"",""nzd"",""https://placekitten.com/425/474""
```

[data/countries/city.csv](https://github.com/rioriost/agefreighter/raw/main/data/countries/city.csv)

```csv
"id","Name","Latitude","Longitude"
"1","東京","-56.4217435","-44.924586"
"2","Bryantton","-62.714695","-162.083092"
"3","Royview","-72.721467","-33.926544"
"4","New Jonathanfurt","18.281926","-63.675749"
```

[data/countries/has_country_city.csv](https://github.com/rioriost/agefreighter/raw/main/data/countries/has_country_city.csv)

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type","since"
"1","86","Country","3633","City","1975-12-07 04:45:00.790431"
"2","22","Country","6194","City","1984-06-05 13:23:51.858147"
"3","80","Country","6479","City","1986-10-20 01:18:47.200926"
"4","185","Country","8148","City","1990-01-05 14:47:47.343686"
```

4. Multiple kinds of nodes and multiple kinds of edges:
```
(Cookie | CreditCard) - [UsedIn | PerformedBy] -> (Payment)
```

This pattern is a little complicatedlly described in JSON as follows [docs/configs/multiple_edges_multiple_nodes.json](https://github.com/rioriost/agefreighter/raw/main/docs/configs/multiple_edges_multiple_nodes.json):

```json
{
  "edge": [
    {
      "csv_path": "data/payment_small/usedin_cookie_payment.csv",
      "type": "UsedIn",
      "props": ["available_since", "inserted_at", "schema_version"],
      "start_vertex": {
        "csv_path": "data/payment_small/cookie.csv",
        "id": "id",
        "label": "Cookie",
        "props": ["available_since", "inserted_at", "uaid", "schema_version"]
      },
      "end_vertex": {
        "csv_path": "data/payment_small/payment.csv",
        "id": "id",
        "label": "Payment",
        "props": [
          "available_since",
          "inserted_at",
          "payment_id",
          "schema_version"
        ]
      }
    },
    {
      "csv_path": "data/payment_small/usedin_creditcard_payment.csv",
      "type": "UsedIn",
      "props": ["available_since", "inserted_at", "schema_version"],
      "start_vertex": {
        "csv_path": "data/payment_small/creditcard.csv",
        "id": "id",
        "label": "CreditCard",
        "props": [
          "available_since",
          "inserted_at",
          "expiry_month",
          "expiry_year",
          "masked_number",
          "creditcard_identifier",
          "schema_version"
        ]
      },
      "end_vertex": {
        "csv_path": "data/payment_small/payment.csv",
        "id": "id",
        "label": "Payment",
        "props": [
          "available_since",
          "inserted_at",
          "payment_id",
          "schema_version"
        ]
      }
    },
    {
      "csv_path": "data/payment_small/performedby_cookie_payment.csv",
      "type": "PerformedBy",
      "props": ["available_since", "inserted_at", "schema_version"],
      "start_vertex": {
        "csv_path": "data/payment_small/cookie.csv",
        "id": "id",
        "label": "Cookie",
        "props": ["available_since", "inserted_at", "uaid", "schema_version"]
      },
      "end_vertex": {
        "csv_path": "data/payment_small/payment.csv",
        "id": "id",
        "label": "Payment",
        "props": [
          "available_since",
          "inserted_at",
          "payment_id",
          "schema_version"
        ]
      }
    },
    {
      "csv_path": "data/payment_small/performedby_creditcard_payment.csv",
      "type": "PerformedBy",
      "props": ["available_since", "inserted_at", "schema_version"],
      "start_vertex": {
        "csv_path": "data/payment_small/creditcard.csv",
        "id": "id",
        "label": "CreditCard",
        "props": [
          "available_since",
          "inserted_at",
          "expiry_month",
          "expiry_year",
          "masked_number",
          "creditcard_identifier",
          "schema_version"
        ]
      },
      "end_vertex": {
        "csv_path": "data/payment_small/payment.csv",
        "id": "id",
        "label": "Payment",
        "props": [
          "available_since",
          "inserted_at",
          "payment_id",
          "schema_version"
        ]
      }
    }
  ]
}
```

And the contents of the CSV files:

[data/payment_small/cookie.csv](https://github.com/rioriost/agefreighter/raw/main/data/payment_small/cookie.csv)

```csv
"id","available_since","inserted_at","uaid","schema_version"
"1","2025-01-13 03:17:10.612472","2025-01-13 03:17:10.612472","a3ef7bff-1d7f-4e59-9963-79139940d9b4","1"
"2","2025-01-02 02:05:26.888933","2025-01-02 02:05:26.888933","23cbd6f6-b11d-4594-a2b5-87243725abe1","1"
"3","2025-01-03 14:46:58.899109","2025-01-03 14:46:58.899109","0891cfda-3c0a-4089-ba35-a5b30c0e2d26","1"
"4","2025-01-04 18:57:37.857941","2025-01-04 18:57:37.857941","2e8c8f67-b1cd-429d-b856-3599ee2cdd59","1"
```

[data/payment_small/creditcard.csv](https://github.com/rioriost/agefreighter/raw/main/data/payment_small/creditcard.csv)

```csv
"id","available_since","inserted_at","expiry_month","expiry_year","masked_number","creditcard_identifier","schema_version"
"1","2025-01-22 00:25:27.612965","2025-01-22 00:25:27.612965","12","1969","30473257263635","28d9abb3-0ce5-415e-b179-3aef91d9aaab","1"
"2","2025-01-18 16:23:55.876961","2025-01-18 16:23:55.876961","12","1969","4618391378453392","a4050d78-a224-4269-9377-5b7fa39d4c4e","1"
"3","2025-01-20 11:23:23.517449","2025-01-20 11:23:23.517449","12","1969","2720988758344312","ba65d7ff-769c-441a-bce6-3a9daae1a7a7","1"
"4","2025-01-17 03:11:17.633718","2025-01-17 03:11:17.633718","12","1969","2706558801011685","7cde9b77-ba58-4779-b435-3a22733a363a","1"
```

[data/payment_small/payment.csv](https://github.com/rioriost/agefreighter/raw/main/data/payment_small/payment.csv)

```csv
"id","available_since","inserted_at","payment_id","schema_version"
"1","2025-01-06 06:03:20.048255","2025-01-06 06:03:20.048255","8dd082fa-cb55-450c-aee4-04893a282f41","1"
"2","2025-01-13 20:41:58.478938","2025-01-13 20:41:58.478938","ec490576-65ca-414f-9914-929ba1ffe26f","1"
"3","2025-01-19 07:58:39.699038","2025-01-19 07:58:39.699038","124659b6-0d7c-4de1-bba8-6de9f1dfee3c","1"
"4","2025-01-10 00:33:53.890556","2025-01-10 00:33:53.890556","6a455a45-a156-44e4-8f01-0750a8f2a9ca","1"
```

[data/payment_small/usedin_cookie_payment.csv](https://github.com/rioriost/agefreighter/raw/main/data/payment_small/usedin_cookie_payment.csv)

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type","available_since","inserted_at","schema_version"
"1","251","Cookie","10","Payment","2025-01-01 22:27:01.284325","2025-01-01 22:27:01.284325","1"
"2","1045","Cookie","4967","Payment","2025-01-02 09:39:38.392602","2025-01-02 09:39:38.392602","1"
"3","351","Cookie","6635","Payment","2025-01-09 13:35:43.667663","2025-01-09 13:35:43.667663","1"
"4","714","Cookie","6532","Payment","2025-01-19 23:54:52.656538","2025-01-19 23:54:52.656538","1"
```

[data/payment_small/usedin_creditcard_payment.csv](https://github.com/rioriost/agefreighter/raw/main/data/payment_small/usedin_creditcard_payment.csv)

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type","available_since","inserted_at","schema_version"
"1","592","CreditCard","1252","Payment","2025-01-17 19:03:49.372280","2025-01-17 19:03:49.372280","1"
"2","353","CreditCard","6369","Payment","2025-01-02 12:14:50.556809","2025-01-02 12:14:50.556809","1"
"3","815","CreditCard","3581","Payment","2025-01-06 02:18:22.648581","2025-01-06 02:18:22.648581","1"
"4","1187","CreditCard","5420","Payment","2025-01-12 12:09:19.853148","2025-01-12 12:09:19.853148","1"
```

[data/payment_small/performedby_cookie_payment.csv](https://github.com/rioriost/agefreighter/raw/main/data/payment_small/performedby_cookie_payment.csv)

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type","available_since","inserted_at"
"1","211","Cookie","6806","Payment","2025-01-15 18:16:30.558804","2025-01-15 18:16:30.558804"
"2","900","Cookie","4174","Payment","2025-01-03 04:26:16.651747","2025-01-03 04:26:16.651747"
"3","1002","Cookie","5877","Payment","2025-01-15 00:25:23.377948","2025-01-15 00:25:23.377948"
"4","215","Cookie","2852","Payment","2025-01-05 15:35:52.251420","2025-01-05 15:35:52.251420"
```

[data/payment_small/performedby_creditcard_payment.csv](https://github.com/rioriost/agefreighter/raw/main/data/payment_small/performedby_creditcard_payment.csv)

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type","available_since","inserted_at"
"1","719","CreditCard","1532","Payment","2025-01-05 18:59:48.910719","2025-01-05 18:59:48.910719"
"2","1025","CreditCard","2153","Payment","2025-01-03 17:06:29.660728","2025-01-03 17:06:29.660728"
"3","563","CreditCard","2622","Payment","2025-01-01 05:33:03.489332","2025-01-01 05:33:03.489332"
"4","1042","CreditCard","350","Payment","2025-01-12 06:40:14.256650","2025-01-12 06:40:14.256650"
```

### Load from Cosmos DB

To load graph data from Cosmos DB, use the following command:

(supposed PG_CONNECTION_STRING is set)
```bash
agefreighter load --source-type cosmosdb --cosmos-endpoint https://YOUR_ACCOUNT.documents.azure.com:443/ --cosmos-key YOUR_KEY --cosmos-database YOUR_DB --cosmos-container YOUR_CONTAINER
```

Or, you can use the environment variables:

- macOS / Linux

```bash
export COSMOS_ENDPOINT="http://YOUR_ACCOUNT.documents.azure.com:443/"
export COSMOS_KEY="YOUR_KEY"
export COSMOS_DATABASE="YOUR_DB"
export COSMOS_CONTAINER="YOUR_CONTAINER"
```

- Windows (cmd.exe)

```bash
set COSMOS_ENDPOINT=http://YOUR_ACCOUNT.documents.azure.com:443/
set COSMOS_KEY=YOUR_KEY
set COSMOS_DATABASE=YOUR_DB
set COSMOS_CONTAINER=YOUR_CONTAINER
```

- PowerShell

```bash
$env:COSMOS_ENDPOINT="http://YOUR_ACCOUNT.documents.azure.com:443/"
$env:COSMOS_KEY="YOUR_KEY"
$env:COSMOS_DATABASE="YOUR_DB"
$env:COSMOS_CONTAINER="YOUR_CONTAINER"
```

If all the required environment variables are set, you can just run:

```bash
agefreighter load --source-type cosmos
```

## How to edit the CSV files to load them to the graph database with PGFreighter

Example: [krlawrence graph](https://github.com/krlawrence/graph/tree/master/sample-data)

1. Download air-routes-latest-edges.csv and air-routes-latest-nodes.csv

2. Edit air-routes-latest-nodes.csv

- original

```csv
~id,~label,type:string,code:string,icao:string,desc:string,region:string,runways:int,longest:int,elev:int,country:string,city:string,lat:double,lon:double,author:string,date:string
0,version,version,0.89,,Air Routes Data - Version: 0.89 Generated: 2022-08-29 14:10:18 UTC; Graph created by Kelvin R. Lawrence; Please let me know of any errors you find in the graph or routes that should be added.,,,,,,,,,Kelvin R. Lawrence,2022-08-29 14:10:18 UTC
```

- edited

```csv
id,label,type,code,icao,desc,region,runways,longest,elev,country,city,lat,lon,author,date
1,airport,airport,ATL,KATL,Hartsfield - Jackson Atlanta International Airport,US-GA,5,12390,1026,US,Atlanta,33.6366996765137,-84.4281005859375,,
```

- remove the second line
- edit the first line (CSV Header)

3. Edit air-routes-latest-edges.csv

- original

```csv
~id,~from,~to,~label,dist:int
3749,1,3,route,809
```

- edited

```csv
id,start_id,end_id,label,dist,start_vertex_type,end_vertex_type
3749,1,3,route,809,airport,airport
```

- edit the first line (CSV Header)
- add start_vertex_type and end_vertex_type columns to each lines

4. Install agefreighter

5. Make a JSON file and save it as 'config.json'

```json
{
  "edge": {
    "csv_path": "air-routes-latest-edges.csv",
    "type": "AirRoute",
    "props": ["dist"],
    "start_vertex": {
      "csv_path": "air-routes-latest-nodes.csv",
      "id": "start_id",
      "label": "AirPort",
      "props": ["code", "icao", "desc", "region", "runways", "longest", "elev", "country", "city", "lat", "lon", "author", "date"]
    },
    "end_vertex": {
      "csv_path": "air-routes-latest-nodes.csv",
      "id": "end_id",
      "label": "AirPort",
      "props": ["code", "icao", "desc", "region", "runways", "longest", "elev", "country", "city", "lat", "lon", "author", "date"]
    }
  }
}
```

6. Deploy Azure Database for PostgreSQL and enable Apache AGE extension on Azure Portal
   [Introducing support for Graph data in Azure Database for PostgreSQL (Preview)](https://techcommunity.microsoft.com/blog/adforpostgresql/introducing-support-for-graph-data-in-azure-database-for-postgresql-preview/4275628).

7. Set the PostgreSQL connection string as an environment variable

- macOS / Linux

```bash
export PG_CONNECTION_STRING="host=YOUR_SERVER.postgres.database.azure.com port=5432 dbname=postgres user=YOUR_USERNAME password=YOUR_PASSWORD"
```

- Windows (cmd.exe)

```bash
set PG_CONNECTION_STRING=host=YOUR_SERVER.postgres.database.azure.com port=5432 dbname=postgres user=YOUR_USERNAME password=YOUR_PASSWORD
```

- PowerShell

```bash
$env:PG_CONNECTION_STRING="host=YOUR_SERVER.postgres.database.azure.com port=5432 dbname=postgres user=YOUR_USERNAME password=YOUR_PASSWORD"
```

8. Run the script

```shell
agefreighter load --source-type csv --config config.json
```

9. Check the graph created in the PostgreSQL database

```sql
% psql $PG_CONNECTION_STRING
psql (16.6 (Homebrew), server 16.4)
SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, compression: off)
Type "help" for help.

postgres=> SET search_path = ag_catalog, "$user", public;
SET
postgres=> select * from air_route.airport limit 1;
       id        |                                                                                                                                                                       properties
-----------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 844424930131969 | {"id": "1", "lat": "33.6366996765137", "lon": "-84.4281005859375", "city": "Atlanta", "code": "ATL", "date": "nan", "desc": "Hartsfield - Jackson Atlanta International Airport", "elev": "1026.0", "icao": "KATL", "type": "airport", "label": "airport", "author": "nan", "region": "US-GA", "country": "US", "longest": "12390.0", "runways": "5.0"}
(1 row)

postgres=> select * from air_route.route limit 1;
        id        |    start_id     |     end_id      |                    properties
------------------+-----------------+-----------------+---------------------------------------------------
 1125899906842625 | 844424930131969 | 844424930131971 | {"id": "3749", "dist": "809.0", "label": "route"}
(1 row)
```

## Release Notes

### 1.0.0a3 Release
- Fixed documents.

### 1.0.0a2 Release
- Various bug fixes, improve the robustness of the application.

### 1.0.0a1 Release
- Totally new release, refactored codebase.

### 0.9.2 Release
- Fixed an error message from 'neo2age.py'.

### 0.9.1 Release
- Fixed asyncio on Windows.

### 0.9.0 Release
- Added 'neo2age.py' based on 'neo2mcsv.py'.
- Added 'copy()' method to AgeFreighter class.

### 0.8.13 Release
- Added handling of special characters in 'neo2mcsv.py'.

### 0.8.12 Release
- Added handling of 'labelless' nodes in 'neo2mcsv.py' and others.

### 0.8.11 Release
- Fixed a bug confusing elementId and id if the graph is exported from Neo4j with 'neo2mcsv.py'.

### 0.8.10 Release
- Added a handling of 'labelless' nodes in 'neo2mcsv.py'.

### 0.8.9 Release
- Fixed minor issue in 'neo2mcsv.py' on Windows.

### 0.8.8 Release
- Added `source_columns` parameter to PGFreighter class.

### 0.8.7 Release
- Fixed minor issue in 'neo2mcsv.py' on Windows.

### 0.8.6 Release
- Fixed minor issue in 'neo2mcsv.py' on Windows.

### 0.8.5 Release
- Fixed minor issue in 'neo2mcsv.py'.

### 0.8.4 Release
- Fixed unicode encoding issue in 'neo2mcsv.py'.

### 0.8.3 Release
- Added 'neo2mcsv.py' to export a graph from Neo4j for MultiCSVFreighter.

### 0.8.2 Release
- Updated for the dependencies

### 0.8.1 Release
- Added CosmosNoSQLFreighter class.
- CosmosGremlinFreighter class will be obsoleted in the future.

### 0.8.0 Release
- Introduced unit tests for the classes to improve the quality of the package. Currently, the tests are only for a few classes.
- Fixed code to improve the robustness of the package.

### 0.7.5 Release
- Added 'progress' argument to the load() method. It's implemented as an optional argument for all the classes. Thanks to @cjoakim for the suggestion.

### 0.7.4 Release
- Changed the required module from psycopg / psycopg_pool to psycopg[binary,pool]

### 0.7.3 Release
- Added min_connections argument to the connect() method. Added the limitation of UNIX environment to import 'resource' module.

### 0.7.2 Release
- Added Mermaid diagram for the document.

### 0.7.1 Release
- Tuned MultiAzureStorageFreighter.

### 0.7.0 Release
- Added MultiAzureStorageFreighter.

### 0.6.1 Release
- Refactored the documents. Added sample data. Fixed some bugs.

### 0.6.0 Release
- Added edge properties support.
  - 'edge_props' argument (list) is added to the 'load()' method.
- 'drop_graph' argument is obsoleted. 'create_graph' argument is added.
  - 'create_graph' is set to True by default. CAUTION: If the graph already exists, the graph is dropped before loading the data.
  - If 'create_graph' is False, the data is loaded into the existing graph.

### 0.5.3 Release -AzureStorageFreighter-
- AzureStorageFreighter class is totally refactored for better performance and scalability.
  - 0.5.2 didn't work well for large files.
  - Now, it works well for large files.
    Checked with a 5.4GB CSV file consisting of 10M of start vertices, 10K of end vertices, and 25M edges,
    it took 512 seconds to load the data into the graph database with PostgreSQL Flex,
    Standard_D32ds_v4 (32 vcpus, 128 GiB memory) and 512TB / 7500 iops of storage.
  - Tested data was generated with tests/generate_dummy_data.py.
  - UDF to load the data to graph is no longer used.
- However, please note that it is still in the early stages of implementation, so there is room for optimization and potential issues due to insufficient testing.

### 0.5.2 Release -AzureStorageFreighter-
- AzureStorageFreighter class is used to load data from Azure Storage into the graph database. It's totally different from other classes. The class works as follows:
  - If the argument, 'subscription_id' is not set, the class tries to find the Azure Subscription ID from your local environment using the 'az' command.
  - Creates an Azure Storage account and a blob container under the resource group where the PostgreSQL server runs in.
  - Enables the 'azure_storage' extension in the PostgreSQL server, if it's not enabled.
  - Uploads the CSV file to the blob container.
  - Creates a UDF (User Defined Function) named 'load_from_azure_storage' in the PostgreSQL server. The UDF loads data from the Azure Storage into the graph database.
  - Executes the UDF.
- The above process takes time to prepare for loading data, making it unsuitable for loading small files, but effective for loading large files. For instance, it takes under 3 seconds to load 'actorfilms.csv' after uploading.
- However, please note that it is still in the early stages of implementation, so there is room for optimization and potential issues due to insufficient testing.

### 0.5.0 Release
Refactored the code to make it more readable and maintainable with the separated classes for factory model.
Please note how to use the new version of the package is tottally different from the previous versions.

## Known Issues
- Apache AGE 1.5 doesn't support:
  - tab (chr(9)) in agtype. If AGEFreighter detects replacing tab with '\t', writes a log file.
  - multiple labels for nodes due to design limitations. (https://github.com/apache/age/discussions/109)

## For More Information

- Apache AGE : https://age.apache.org/
- GitHub : https://github.com/apache/age
- Document : https://age.apache.org/age-manual/master/index.html

## License

MIT License
