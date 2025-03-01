# AGEFreighter

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)

a Python package that helps you to create a graph database using Azure Database for PostgreSQL.

[Apache AGE™](https://age.apache.org/) is a PostgreSQL Graph database compatible with PostgreSQL's distributed assets and leverages graph data structures to analyze and use relationships and patterns in data.

[Azure Database for PostgreSQL](https://azure.microsoft.com/en-us/services/postgresql/) is a managed database service that is based on the open-source Postgres database engine.

[Introducing support for Graph data in Azure Database for PostgreSQL (Preview)](https://techcommunity.microsoft.com/blog/adforpostgresql/introducing-support-for-graph-data-in-azure-database-for-postgresql-preview/4275628).

## Table of Contents

- [Features](#features)
- [Benchmark](#benchmark)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Which class to use](#which-class-to-use)
- [Usage of CSVFreighter](#usage-of-csvfreighter)
- [Usage of MultiCSVFreighter (1)](#usage-of-multicsvfreighter-1)
- [Usage of MultiCSVFreighter (2)](#usage-of-multicsvfreighter-2)
- [Usage of AvroFreighter](#usage-of-avrofreighter)
- [Usage of ParquetFreighter](#usage-of-parquetfreighter)
- [Usage of AzureStorageFreighter](#usage-of-azurestoragefreighter)
- [Usage of MultiAzureStorageFreighter](#usage-of-multiazurestoragefreighter)
- [Usage of NetworkxFreighter](#usage-of-networkxfreighter)
- [Usage of CosmosGremlinFreighter](#usage-of-cosmosgremlinfreighter)
- [Usage of CosmosNoSQLFreighter](#usage-of-cosmosnosqlfreighter)
- [Usage of Neo4jFreighter](#usage-of-neo4jfreighter)
- [Usage of PGFreighter](#usage-of-pgfreighter)
- [How to edit the CSV files to load them to the graph database with PGFreighter](#how-to-edit-the-csv-files-to-load-them-to-the-graph-database-with-pgfreighter)
- [How to export the graph data from Neo4j as CSV files and load them to Apache AGE](#how-to-export-the-graph-data-from-neo4j-as-csv-files-and-load-them-to-apache-age)
- [Classes](#classes)
- [Method](#method)
- [Arguments](#arguments)
- [Release Notes](#release-notes)
- [License](#license)

## Features

- Asynchronous connection pool support for psycopg PostgreSQL driver
- 'direct_loading' option for loading data directly into the graph. If 'direct_loading' is True, the data is loaded into the graph using the 'INSERT' statement, not Cypher queries.
- 'COPY' protocol support for loading data into the graph. If 'use_copy' is True, the data is loaded into the graph using the 'COPY' protocol.
- AzureStorageFreighter and MultiAzureStorageFreighter classes to load vast amounts of graph data from Azure Storage. Typically, the number of rows in the CSV files exceeds from a million to a billion.

## Benchmark

The result with Azure Database for PostgreSQL, General Purpose, D16ds_v4, 16 vCores, 64 GiB RAM, 512 GiB storage (7,500 IOPS)
See, [tests/agefreightertester.py](https://github.com/rioriost/agefreighter/blob/main/tests/agefreightertester.py)

```bash
for d in `ls data`; do echo $d; wc -l data/$d/* | grep total; done
airroute
   23500 total
countries
   20200 total
payment_large
 96520015 total
payment_small
   96520 total
transaction
   43003 total
```

```bash
AgeFreighter version: 0.8.1
Summary of all tests are as followings:
Test for AzureStorageFreighter, chunk_size(96), direct_loading(False), use_copy(False): SUCCEEDED,  50.80 seconds
Test for MultiAzureStorageFreighter, chunk_size(96), direct_loading(False), use_copy(False): SUCCEEDED,  46.34 seconds
Test for AvroFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  1.08 seconds
Test for CosmosGremlinFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  6.86 seconds
Test for CosmosNoSQLFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  2.74 seconds
Test for CSVFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  0.82 seconds
Test for MultiCSVFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  0.67 seconds
Test for MultiCSVFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  0.74 seconds
Test for Neo4jFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  3.48 seconds
Test for NetworkXFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  0.87 seconds
Test for ParquetFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  0.95 seconds
Test for PGFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  1.03 seconds
```

## Prerequisites

- over Python 3.9
- This module runs on [psycopg](https://www.psycopg.org/) and [psycopg_pool](https://www.psycopg.org/)
- Enable the Apache AGE extension in your Azure Database for PostgreSQL instance. Login Azure Portal, go to 'server parameters' blade, and check 'AGE" on within 'azure.extensions' and 'shared_preload_libraries' parameters. See, above blog post for more information.
- Load the AGE extension in your PostgreSQL database.

```sql
CREATE EXTENSION IF NOT EXISTS age CASCADE;
```

## Install

- with python venv

```bash
mkdir your_project
cd your_project
python3 -m venv .venv
source .venv/bin/activate
pip install agefreighter
```

- with uv

```bash
uv init your_project
cd your_project
uv venv
source .venv/bin/activate
uv add agefreighter
```

## Which class to use

![Decision Tree](https://github.com/rioriost/agefreighter/raw/main/images/Decision_tree.png)

## Usage of CSVFreighter

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("CSVFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Transaction",
        start_v_label="Customer",
        start_id="CustomerID",
        start_props=["Name", "Address", "Email", "Phone"],
        edge_type="BOUGHT",
        edge_props=[],
        end_v_label="Product",
        end_id="ProductID",
        end_props=["Phrase", "SKU", "Price", "Color", "Size", "Weight"],
        csv_path="data/transaction/customer_product_bought.csv",
        use_copy=True,
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

### File Format for CSVFreighter

CSVFreighter class loads data from single CSV file. The CSV file should have the following format.

customer_product_bought.csv: The CSV file should have 'id', 'start_vertex_type', 'end_vertex_type', two id columns, 'CustomerID' and 'ProductID' in the following sample, to be used as start and end vertex IDs, and other columns as properties.

```csv
"id","CustomerID","start_vertex_type","Name","Address","Email","Phone","ProductID","end_vertex_type","Phrase","SKU","Price","Color","Size","Weight"
"1","1967","Customer","Jeffrey Joyce","26888 Brett Streets Apt. 325 South Meganberg, CA 80228","madison05@example.com","881-538-6881x35597","120","Product","Networked 3rdgeneration data-warehouse","7246676575258","834.33","DarkKhaki","S","586"
"2","8674","Customer","Craig Burton","280 Sellers Lock North Scott, AR 15307","andersonalexander@example.com","+1-677-235-8289","557","Product","Profit-focused attitude-oriented emulation","6102707440852","953.89","MediumSeaGreen","L","665"
```

See, [data/transaction/customer_product_bought.csv](https://github.com/rioriost/agefreighter/blob/main/data/transaction/customer_product_bought.csv).

## Usage of MultiCSVFreighter (1)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("MultiCSVFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Countries",
        vertex_csv_paths=[
            "data/countries/country.csv",
            "data/countries/city.csv",
        ],
        vertex_labels=["Country", "City"],
        edge_csv_paths=["data/countries/has_country_city.csv"],
        edge_types=["has"],
        use_copy=True,
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

### File Format for MultiCSVFreighter (1)

MultiCSVFreighter class loads data from multiple CSV files. The CSV files should have the following format.
MultiCSVFreighter class handles all the columns in CSV except 'id' column as properties and all the columns in CSV except 'start_id' / 'start_vertex_type' / 'end_id' / 'end_vertex_type' columns as properties for the edges.

country.csv: The node CSV file should have 'id' column and other columns as properties.

```csv
"id","Name","Capital","Population","ISO","TLD","FlagURL"
"1","El Salvador","Kristybury","355169921","TN","wxu","https://dummyimage.com/777x133"
"2","Lebanon","New William","413929227","UK","akj","https://picsum.photos/772/459"
```

city.csv: The node CSV file should have 'id' column and other columns as properties.

```csv
"id","Name","Latitude","Longitude"
"1","Michaelmouth","-56.4217435","-44.924586"
"2","Bryantton","-62.714695","-162.083092"
```

has_country_city.csv: The edge CSV file should have 'id', 'start_id', 'start_vertex_type', 'end_id', 'end_vertex_type', and other columns as properties.

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type","since"
"1","86","Country","3633","City","1975-12-07 04:45:00.790431"
"2","22","Country","6194","City","1984-06-05 13:23:51.858147"
```

See, [data/countries/](https://github.com/rioriost/agefreighter/blob/main/data/countries/).

## Usage of MultiCSVFreighter (2)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("MultiCSVFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )
    await instance.load(
        graph_name="AirRoute",
        vertex_csv_paths=[
            "data/airroute/airport.csv",
        ],
        vertex_labels=["AirPort"],
        edge_csv_paths=["data/airroute/airroute_airport_airport.csv"],
        edge_types=["ROUTE"],
        edge_props = ["distance"],
        use_copy=True,
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

### File Format for MultiCSVFreighter (2)

If the edge connects the same type of vertices, the CSV files should have the following format.

airport.csv: The node CSV file should have 'id' column and other columns as properties.

```csv
"id","Name","City","Country","IATA","ICAO","Latitude","Longitude","Altitude","Timezone","DST","Tz"
"1","East Annatown Airport","East Annatown","Eritrea","SHZ","XTIK","-2.783983","-100.199060","823","Africa/Luanda","E","Europe/Skopje"
"2","Port Laura Airport","Port Laura","Montenegro","TQY","WDLC","4.331082","-72.411319","121","Asia/Dhaka","Z","Africa/Kigali"
```

airroute_airport_airport.csv: The edge CSV file should have 'id', 'start_id', 'start_vertex_type', 'end_id', 'end_vertex_type', and other columns as properties.

```csv
"id","start_id","start_vertex_type","end_id","end_vertex_type","distance"
"1","1388","AirPort","794","AirPort","1373"
"2","2998","AirPort","823","AirPort","11833"
```

See, [data/airroute/](https://github.com/rioriost/agefreighter/blob/main/data/airroute/).

## Usage of AvroFreighter

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("AvroFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Transaction",
        start_v_label="Customer",
        start_id="CustomerID",
        start_props=["Name", "Address", "Email", "Phone"],
        edge_type="BOUGHT",
        edge_props=[],
        end_v_label="Product",
        end_id="ProductID",
        end_props=["Phrase", "SKU", "Price", "Color", "Size", "Weight"],
        avro_path="data/transaction/customer_product_bought.avro",
        use_copy=True,
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

### File Format for AvroFreighter

AvroFreighter class loads data from Avro file. The Avro file should have the following format.

```json
{
    "type": "record",
    "name": "customer_product_bought",
    "fields": [
        {
            "name": "id",
            "type": "int"
        },
        {
            "name": "CustomerID",
            "type": "int"
        },
        {
            "name": "start_vertex_type",
            "type": "string"
        },
        {
            "name": "Name",
            "type": "string"
        },
        {
            "name": "Address",
            "type": "string"
        },
        {
            "name": "Email",
            "type": "string"
        },
        {
            "name": "Phone",
            "type": "string"
        },
        {
            "name": "ProductID",
            "type": "int"
        },
        {
            "name": "end_vertex_type",
            "type": "string"
        },
        {
            "name": "Phrase",
            "type": "string"
        },
        {
            "name": "SKU",
            "type": "string"
        },
        {
            "name": "Price",
            "type": "float"
        },
        {
            "name": "Color",
            "type": "string"
        },
        {
            "name": "Size",
            "type": "string"
        },
        {
            "name": "Weight",
            "type": "int"
        }
    ]
}
{
    "id": 1,
    "CustomerID": 1967,
    "start_vertex_type": "Customer",
    "Name": "Jeffrey Joyce",
    "Address": "26888 Brett Streets Apt. 325 South Meganberg, CA 80228",
    "Email": "madison05@example.com",
    "Phone": "881-538-6881x35597",
    "ProductID": 120,
    "end_vertex_type": "Product",
    "Phrase": "Networked 3rdgeneration data-warehouse",
    "SKU": "7246676575258",
    "Price": 834.3300170898438,
    "Color": "DarkKhaki",
    "Size": "S",
    "Weight": 586
}
{
    "id": 2,
    "CustomerID": 8674,
    "start_vertex_type": "Customer",
    "Name": "Craig Burton",
    "Address": "280 Sellers Lock North Scott, AR 15307",
    "Email": "andersonalexander@example.com",
    "Phone": "+1-677-235-8289",
    "ProductID": 557,
    "end_vertex_type": "Product",
    "Phrase": "Profit-focused attitude-oriented emulation",
    "SKU": "6102707440852",
    "Price": 953.8900146484375,
    "Color": "MediumSeaGreen",
    "Size": "L",
    "Weight": 665
}
```

See, [data/transaction/customer_product_bought.avro](https://github.com/rioriost/agefreighter/blob/main/data/transaction/customer_product_bought.avro).

## Usage of ParquetFreighter

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("ParquetFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Transaction",
        start_v_label="Customer",
        start_id="CustomerID",
        start_props=["Name", "Address", "Email", "Phone"],
        edge_type="BOUGHT",
        edge_props=[],
        end_v_label="Product",
        end_id="ProductID",
        end_props=["Phrase", "SKU", "Price", "Color", "Size", "Weight"],
        parquet_path="data/transaction/customer_product_bought.parquet",
        use_copy=True,
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

### File Format for ParquetFreighter

ParquetFreighter class loads data from Parquet file. The Parquet file should have the following format.

```
required group field_id=-1 schema {
  optional int64 field_id=-1 id;
  optional int64 field_id=-1 CustomerID;
  optional binary field_id=-1 start_vertex_type (String);
  optional binary field_id=-1 Name (String);
  optional binary field_id=-1 Address (String);
  optional binary field_id=-1 Email (String);
  optional binary field_id=-1 Phone (String);
  optional int64 field_id=-1 ProductID;
  optional binary field_id=-1 end_vertex_type (String);
  optional binary field_id=-1 Phrase (String);
  optional int64 field_id=-1 SKU;
  optional double field_id=-1 Price;
  optional binary field_id=-1 Color (String);
  optional binary field_id=-1 Size (String);
  optional int64 field_id=-1 Weight;
}

   id  CustomerID start_vertex_type           Name                                            Address  ...            SKU   Price           Color Size Weight
0   1        1967          Customer  Jeffrey Joyce  26888 Brett Streets Apt. 325 South Meganberg, ...  ...  7246676575258  834.33       DarkKhaki    S    586
1   2        8674          Customer   Craig Burton             280 Sellers Lock North Scott, AR 15307  ...  6102707440852  953.89  MediumSeaGreen    L    665
```

See, [data/transaction/customer_product_bought.parquet](https://github.com/rioriost/agefreighter/blob/main/data/transaction/customer_product_bought.parquet).

## Usage of AzureStorageFreighter

### Prerequisites

Install the Azure CLI and login with your Azure account.

macOS

```bash
brew update && brew install azure-cli
```

Windows

```shell
winget install -e --id Microsoft.AzureCLI
```

Linux (RHEL)

```bash
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc

# for RHEL 9
sudo dnf install -y https://packages.microsoft.com/config/rhel/9.0/packages-microsoft-prod.rpm
# for RHEL 8
sudo dnf install -y https://packages.microsoft.com/config/rhel/8/packages-microsoft-prod.rpm

sudo dnf install azure-cli
```

Linux (Ubuntu)

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

Afrer installing the Azure CLI, login with your Azure account.

```bash
az login
```

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("AzureStorageFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Transaction",
        start_v_label="Customer",
        start_id="CustomerID",
        start_props=["Name", "Address", "Email", "Phone"],
        edge_type="BOUGHT",
        edge_props=[],
        end_v_label="Product",
        end_id="ProductID",
        end_props=["Phrase", "SKU", "Price", "Color", "Size", "Weight"],
        csv_path="data/transaction/customer_product_bought.csv",
        drop_graph=True,
        create_graph=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

### File Format for AzureStorageFreighter

AzureStorageFreighter class loads data from Azure Storage and expects the exactly same format as CSVFreighter.

## Usage of MultiAzureStorageFreighter

Banchmark: loading 965 million rows of data from Azure Storage to Azure Database for PostgreSQL with MultiAzureStorageFreighter class.

```bash
% wc -l data/payment_large/*
  900001 data/payment_large/bitcoinaddress.csv
 2700001 data/payment_large/cookie.csv
 1200001 data/payment_large/creditcard.csv
 1600001 data/payment_large/cryptoaddress.csv
  960001 data/payment_large/email.csv
 2200001 data/payment_large/ip.csv
 4000001 data/payment_large/partnerenduser.csv
 7000001 data/payment_large/payment.csv
 1000000 data/payment_large/performedby_cookie_payment.csv
 1000000 data/payment_large/performedby_creditcard_payment.csv
 1000000 data/payment_large/performedby_cryptoaddress_payment.csv
 1000000 data/payment_large/performedby_email_payment.csv
 1000000 data/payment_large/performedby_phone_payment.csv
  960001 data/payment_large/phone.csv
 8000001 data/payment_large/usedby_cookie_payment.csv
 8000001 data/payment_large/usedby_creditcard_payment.csv
 8000001 data/payment_large/usedby_cryptoaddress_payment.csv
 8000001 data/payment_large/usedby_email_payment.csv
 8000001 data/payment_large/usedby_phone_payment.csv
 6000000 data/payment_large/usedin_cookie_payment.csv
 6000001 data/payment_large/usedin_creditcard_payment.csv
 6000000 data/payment_large/usedin_cryptoaddress_payment.csv
 6000000 data/payment_large/usedin_email_payment.csv
 6000000 data/payment_large/usedin_phone_payment.csv
 96520015 total
```

The result with Azure Database for PostgreSQL, General Purpose, D16ds_v4, 16 vCores, 64 GiB RAM, 512 GiB storage (7,500 IOPS)

```bash
Finding Subscription ID...
Enabling extension...
Creating storage account...
Uploading files...
Creating temporary tables...
Loading files to temporary tables...
Creating a graph...
Creating a graph: Done!
AgeFreighter version: 0.8.0
Summary of all tests are as followings:
Test for MultiAzureStorageFreighter, chunk_size(96), direct_loading(False), use_copy(False): SUCCEEDED,  2179.69 seconds
```

### Prerequisites

Install the Azure CLI and login with your Azure account.

macOS

```bash
brew update && brew install azure-cli
```

Windows

```shell
winget install -e --id Microsoft.AzureCLI
```

Linux (RHEL)

```bash
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc

# for RHEL 9
sudo dnf install -y https://packages.microsoft.com/config/rhel/9.0/packages-microsoft-prod.rpm
# for RHEL 8
sudo dnf install -y https://packages.microsoft.com/config/rhel/8/packages-microsoft-prod.rpm

sudo dnf install azure-cli
```

Linux (Ubuntu)

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

Afrer installing the Azure CLI, login with your Azure account.

```bash
az login
```

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("MultiAzureStorageFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    data_dir = "data/payment_small/"

    await instance.load(
        graph_name="AgeTester",
        vertex_args=[
            {
                "csv_path": f"{data_dir}bitcoinaddress.csv",
                "label": "BitcoinAddress",
                "id": "id",
                "props": [
                    "available_since",
                    "inserted_at",
                    "address",
                    "schema_version",
                ],
            },
            {
                "csv_path": f"{data_dir}cookie.csv",
                "label": "Cookie",
                "id": "id",
                "props": [
                    "available_since",
                    "inserted_at",
                    "uaid",
                    "schema_version",
                ],
            },
            {
                "csv_path": f"{data_dir}ip.csv",
                "label": "IP",
                "id": "id",
                "props": [
                    "available_since",
                    "inserted_at",
                    "address",
                    "schema_version",
                ],
            },
            {
                "csv_path": f"{data_dir}phone.csv",
                "label": "Phone",
                "id": "id",
                "props": [
                    "available_since",
                    "inserted_at",
                    "address",
                    "schema_version",
                ],
            },
            {
                "csv_path": f"{data_dir}email.csv",
                "label": "Email",
                "id": "id",
                "props": [
                    "available_since",
                    "inserted_at",
                    "email",
                    "domain",
                    "handle",
                    "schema_version",
                ],
            },
            {
                "csv_path": f"{data_dir}payment.csv",
                "label": "Payment",
                "id": "id",
                "props": [
                    "available_since",
                    "inserted_at",
                    "payment_id",
                    "schema_version",
                ],
            },
            {
                "csv_path": f"{data_dir}creditcard.csv",
                "label": "CreditCard",
                "id": "id",
                "props": [
                    "available_since",
                    "inserted_at",
                    "expiry_month",
                    "expiry_year",
                    "masked_number",
                    "creditcard_identifier",
                    "schema_version",
                ],
            },
            {
                "csv_path": f"{data_dir}partnerenduser.csv",
                "label": "PartnerEndUser",
                "id": "id",
                "props": [
                    "available_since",
                    "inserted_at",
                    "partner_end_user_id",
                    "schema_version",
                ],
            },
            {
                "csv_path": f"{data_dir}cryptoaddress.csv",
                "label": "CryptoAddress",
                "id": "id",
                "props": [
                    "available_since",
                    "inserted_at",
                    "address",
                    "currency",
                    "full_address",
                    "schema_version",
                    "tag",
                ],
            },
        ],
        edge_args=[
            {
                "csv_paths": [
                    f"{data_dir}usedin_cookie_payment.csv",
                    f"{data_dir}usedin_creditcard_payment.csv",
                    f"{data_dir}usedin_cryptoaddress_payment.csv",
                    f"{data_dir}usedin_email_payment.csv",
                    f"{data_dir}usedin_phone_payment.csv",
                ],
                "type": "UsedIn",
            },
            {
                "csv_paths": [
                    f"{data_dir}usedby_cookie_payment.csv",
                    f"{data_dir}usedby_creditcard_payment.csv",
                    f"{data_dir}usedby_cryptoaddress_payment.csv",
                    f"{data_dir}usedby_email_payment.csv",
                    f"{data_dir}usedby_phone_payment.csv",
                ],
                "type": "UsedBy",
            },
            {
                "csv_paths": [
                    f"{data_dir}performedby_cookie_payment.csv",
                    f"{data_dir}performedby_creditcard_payment.csv",
                    f"{data_dir}performedby_cryptoaddress_payment.csv",
                    f"{data_dir}performedby_email_payment.csv",
                    f"{data_dir}performedby_phone_payment.csv",
                ],
                "type": "PerformedBy",
            },
        ],
        drop_graph=True,
        create_graph=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

### File Format for MultiAzureStorageFreighter

MultiAzureStorageFreighter class loads data from Azure Storage and expects the exactly same format as MultiCSVFreighter.

See, [data/payment_small/](https://github.com/rioriost/agefreighter/blob/main/data/payment_small/).

### What AzureStorageFreighter / MultiAzureStorageFreighter do

1. Find the Subscription ID of your Azure account.
2. Enable Azure Storage Extension in your Azure Database for PostgreSQL instance.
3. Create a Storage Account in the resource group where your Azure Database for PostgreSQL instance is located.
4. Create a container in the Storage Account.
5. Upload CSV files to the container.
6. Create temporary tables where the CSV files are loaded into.
7. Load the data from the temporary tables to the graph.

## Usage of NetworkxFreighter

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory

import networkx as nx


async def main():
    instance = Factory.create_instance("NetworkXFreighter")

    networkx_graph = nx.DiGraph()
    networkx_graph.add_node(
        1,
        name="Jeffrey Joyce",
        address="26888 Brett Streets Apt. 325 South Meganberg, CA 80228",
        email="madison05@example.com",
        phone="881-538-6881x35597",
        customerid=1967,
        label="Customer",
    )
    networkx_graph.add_node(
        2,
        name="Craig Burton",
        address="280 Sellers Lock North Scott, AR 15307",
        email="andersonalexander@example.com",
        phone="+1-677-235-8289",
        customerid=8674,
        label="Customer",
    )
    networkx_graph.add_node(
        3,
        phrase="Networked 3rdgeneration data-warehouse",
        sku="7246676575258",
        price=834.33,
        color="DarkKhaki",
        size="S",
        weight=586,
        productid=120,
        label="Product",
    )
    networkx_graph.add_node(
        4,
        phrase="Profit-focused attitude-oriented emulation",
        sku="6102707440852",
        price=953.89,
        color="MediumSeaGreen",
        size="L",
        weight=665,
        productid=557,
        label="Product",
    )

    networkx_graph.add_edge(1, 3, since="1975-12-07 04:45:00.790431", label="BOUGHT")
    networkx_graph.add_edge(2, 4, since="1984-06-05 13:23:51.858147", label="BOUGHT")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )
    await instance.load(
        graph_name="Transaction",
        networkx_graph=networkx_graph,
        id_map={
            "Customer": "CustomerID",
            "Product": "ProductID",
        },
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

You can also load a networkx graph from a pickle file.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory

import pickle


async def main():
    instance = Factory.create_instance("NetworkXFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )
    await instance.load(
        graph_name="Transaction",
        networkx_graph=pickle.load(
            open("data/transaction/customer_product_bought.pickle", "rb")
        ),
        id_map={
            "Customer": "CustomerID",
            "Product": "ProductID",
        },
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

See, [data/transaction/customer_product_bought.pickle](https://github.com/rioriost/agefreighter/blob/main/data/transaction/customer_product_bought.pickle).

## Usage of CosmosNoSQLFreighter

CosmosNoSQLFreighter uses NoSQL API to get data loaded via the Gremlin API for better performance.

```bash
AgeFreighter version: 0.8.1
Summary of all tests are as followings:
Test for CosmosGremlinFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  4.53 seconds
Test for CosmosNoSQLFreighter, chunk_size(96), direct_loading(False), use_copy(True): SUCCEEDED,  2.25 seconds
```

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("CosmosNoSQLFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Transaction",
        cosmos_endpoint=os.environ["COSMOS_ENDPOINT"],
        cosmos_key=os.environ["COSMOS_KEY"],
        cosmos_database="db1",
        cosmos_container="transaction",
        id_map={
            "Customer": "CustomerID",
            "Product": "ProductID",
        },
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

The above code suppose that you have a Cosmos DB account with a Gremlin API database named 'db1' with a container named 'transaction' in the Azure Portal.
You can find the '.NET SDK URI' starting with 'https://', not 'wss://' for 'cosmos_endpoint' env variable and 'PRIMARY KEY' / 'SECONDARY KEY' for 'cosmos_key' env variable in the 'Keys' blade of the Cosmos DB account.

### Document Format for CosmosNoSQLFreighter

CosmosNoSQLFreighter class loads data from Cosmos DB. The Cosmos DB should have the following format.

```json
{
  "label": "Customer",
  "id": "272e8291-d1a4-4238-ad27-92ec2101425d",
  "Name": [
    {
      "id": "ee24bc0d-1821-4854-bd00-aafba2419a1f",
      "_value": "Alicia Herrera"
    }
  ],
  "CustomerID": [
    {
      "id": "f1aafc14-1ecd-4437-b0b4-ce9926ec0924",
      "_value": "1828"
    }
  ],
  "Address": [
    {
      "id": "c611e8b5-a33a-4669-8fbb-36c0737d7ab8",
      "_value": "906 Shannon Views Apt. 370 Ryanbury, CA 73165"
    }
  ],
  "Email": [
    {
      "id": "1f951bf3-47bf-4af8-951c-7915ba810500",
      "_value": "jillian49@example.com"
    }
  ],
  "Phone": [
    {
      "id": "75ca0b65-b5a5-4fdd-8fa5-35db476fede6",
      "_value": "+1-351-871-4405x226"
    }
  ],
  "pk": "1",
  "_rid": "tfxpAMsq0qXfkwEAAAAADA==",
  "_self": "dbs/tfxpAA==/colls/tfxpAMsq0qU=/docs/tfxpAMsq0qXfkwEAAAAADA==/",
  "_etag": "\"2f0055e1-0000-2300-0000-67909dd00000\"",
  "_attachments": "attachments/",
  "_ts": 1737530832
}

{
  "label": "Product",
  "id": "b342f7ac-0170-4d10-973c-e01910e79320",
  "Phrase": [
    {
      "id": "3da120da-c934-4c31-95b2-973e5a5c88ee",
      "_value": "Reverse-engineered asymmetric leverage"
    }
  ],
  "ProductID": [
    {
      "id": "dab94ea0-dfe4-4e30-b660-cc2dd9cfd9b6",
      "_value": "113"
    }
  ],
  "SKU": [
    {
      "id": "2df5bc68-2764-4d5e-a531-eccbc9049f98",
      "_value": "280217698898"
    }
  ],
  "Price": [
    {
      "id": "4c10a970-8b00-4f6c-9ae0-244946dffcb7",
      "_value": "559.27"
    }
  ],
  "Color": [
    {
      "id": "69961024-4807-4ac9-a126-2e71f5f885b7",
      "_value": "White"
    }
  ],
  "Size": [
    {
      "id": "39fad2fd-f879-49f4-9712-473be9d578f5",
      "_value": "L"
    }
  ],
  "Weight": [
    {
      "id": "41e10b86-32c9-4599-b985-6b97ae0719be",
      "_value": "633"
    }
  ],
  "pk": "1",
  "_rid": "tfxpAMsq0qV0uAEAAAAADA==",
  "_self": "dbs/tfxpAA==/colls/tfxpAMsq0qU=/docs/tfxpAMsq0qV0uAEAAAAADA==/",
  "_etag": "\"30009b06-0000-2300-0000-67909ddd0000\"",
  "_attachments": "attachments/",
  "_ts": 1737530845
}

{
  "label": "BOUGHT",
  "id": "674441a4-457a-4d59-a9a8-5c6e2d57ea16",
  "_sink": "510a9b04-d351-4279-b42a-0ea51f7f1a8c",
  "_sinkLabel": "Product",
  "_sinkPartition": "1",
  "_vertexId": "3549c350-59ec-46c7-8e49-589bc0cc6da6",
  "_vertexLabel": "Customer",
  "_isEdge": true,
  "pk": "1",
  "_rid": "tfxpAMsq0qX3ygEAAAAADA==",
  "_self": "dbs/tfxpAA==/colls/tfxpAMsq0qU=/docs/tfxpAMsq0qX3ygEAAAAADA==/",
  "_etag": "\"3000ab19-0000-2300-0000-67909de90000\"",
  "_attachments": "attachments/",
  "_ts": 1737530857
}
```

## Usage of CosmosGremlinFreighter

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("CosmosGremlinFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Transaction",
        cosmos_gremlin_endpoint=os.environ["COSMOS_GREMLIN_ENDPOINT"],
        cosmos_gremlin_key=os.environ["COSMOS_GREMLIN_KEY"],
        cosmos_username="/dbs/db1/colls/transaction",
        id_map={
            "Customer": "CustomerID",
            "Product": "ProductID",
        },
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

The above code suppose that you have a Cosmos DB account with a Gremlin API database named 'db1' with a container named 'transaction' in the Azure Portal.
You can find the 'GREMLIN URI' for 'cosmos_gremlin_endpoint' env variable and 'PRIMARY KEY' / 'SECONDARY KEY' for 'cosmos_gremlin_key' env variable in the 'Keys' blade of the Cosmos DB account.

### Document Format for CosmosGremlinFreighter

CosmosGremlinFreighter class loads data from Cosmos DB. The Cosmos DB should have the following format.

node: g.V().limit(1)

```json
[
  {
    "id": "272e8291-d1a4-4238-ad27-92ec2101425d",
    "label": "Customer",
    "type": "vertex",
    "properties": {
      "Name": [
        {
          "id": "ee24bc0d-1821-4854-bd00-aafba2419a1f",
          "value": "Alicia Herrera"
        }
      ],
      "CustomerID": [
        {
          "id": "f1aafc14-1ecd-4437-b0b4-ce9926ec0924",
          "value": "1828"
        }
      ],
      "Address": [
        {
          "id": "c611e8b5-a33a-4669-8fbb-36c0737d7ab8",
          "value": "906 Shannon Views Apt. 370 Ryanbury, CA 73165"
        }
      ],
      "Email": [
        {
          "id": "1f951bf3-47bf-4af8-951c-7915ba810500",
          "value": "jillian49@example.com"
        }
      ],
      "Phone": [
        {
          "id": "75ca0b65-b5a5-4fdd-8fa5-35db476fede6",
          "value": "+1-351-871-4405x226"
        }
      ],
      "pk": [
        {
          "id": "272e8291-d1a4-4238-ad27-92ec2101425d|pk",
          "value": "1"
        }
      ]
    }
  }
]
```

edge: g.E().limit(1)

```json
[
  {
    "id": "efc34e39-d674-40df-9c6a-f8a24c8b8d77",
    "label": "BOUGHT",
    "type": "edge",
    "inVLabel": "Product",
    "outVLabel": "Customer",
    "inV": "1430cbf2-7d25-4c38-8ff7-f2b215bfdcad",
    "outV": "390565dc-67d7-4179-a241-8cd2f8df82b2"
  }
]
```

## Usage of Neo4jFreighter

Before starting using Neo4jFreighter, please consider to use [neo2age.py](#how-to-export-the-graph-data-from-neo4j-as-csv-files-and-load-them-to-apache-age).
Because Neo4jFreighter needs to talk with Neo4j and PostgreSQL, it is sometimes too slow to load the graph data, especially when the graph is large.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("Neo4jFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Transaction",
        neo4j_uri=os.environ["NEO4J_URI"],
        neo4j_user=os.environ["NEO4J_USER"],
        neo4j_password=os.environ["NEO4J_PASSWORD"],
        id_map={
            "Customer": "CustomerID",
            "Product": "ProductID",
        },
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

The above code suppose that you have a Neo4j or compatible graph DB.

### Data Format for Neo4jFreighter

node: MATCH (n) RETURN n LIMIT 1

```json
{
  "identity": 0,
  "labels": ["Customer"],
  "properties": {
    "Email": "madison05@example.com",
    "Address": "26888 Brett Streets Apt. 325 South Meganberg, CA 80228",
    "Customer": "Customer",
    "Phone": "881-538-6881x35597",
    "CustomerID": 1967,
    "Name": "Jeffrey Joyce"
  },
  "elementId": "4:148f2025-c6d2-4e47-8661-b2f1a28f24aa:0"
}
```

edge: MATCH ()-[r]->() RETURN r LIMIT 1

```json
{
  "identity": 20000,
  "start": 0,
  "end": 17272,
  "type": "BOUGHT",
  "properties": {
    "from": 1967,
    "to": 120
  },
  "elementId": "5:148f2025-c6d2-4e47-8661-b2f1a28f24aa:20000",
  "startNodeElementId": "4:148f2025-c6d2-4e47-8661-b2f1a28f24aa:0",
  "endNodeElementId": "4:148f2025-c6d2-4e47-8661-b2f1a28f24aa:17272"
}
```

## Usage of PGFreighter

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("PGFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Transaction",
        source_pg_con_string=os.environ["SRC_PG_CONNECTION_STRING"],
        source_tables={
            "start": "Customer",
            "end": "Product",
            "edges": "BOUGHT",
        },
        id_map={
            "Customer": "CustomerID",
            "Product": "ProductID",
        },
        drop_graph=True,
        create_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

You can specify the columns in source tables to be loaded with the `source_columns` parameter.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from agefreighter import Factory


async def main():
    instance = Factory.create_instance("PGFreighter")

    await instance.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )

    await instance.load(
        graph_name="Transaction",
        source_pg_con_string=os.environ["SRC_PG_CONNECTION_STRING"],
        source_tables={
            "start": "Customer",
            "end": "Product",
            "edges": "BOUGHT",
        },
        id_map={
            "Customer": "CustomerID",
            "Product": "ProductID",
        },
        drop_graph=True,
        create_graph=True,
        progress=True,
        source_columns={
            "Customer": ["CustomerID", "Name", "Email"],
            "Product": ["ProductID", "Phrase", "Price"],
            "BOUGHT": ["CustomerID", "ProductID"],
        },
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

### Table schemas for PGFreighter

Customer table schema

```sql
postgres=# \d+ "Customer"
                                                                  Table "public.Customer"
     Column     |  Type   | Collation | Nullable |                      Default                       | Storage  | Compression | Stats target | Description
----------------+---------+-----------+----------+----------------------------------------------------+----------+-------------+--------------+-------------
 CustomerSerial | integer |           | not null | nextval('"Customer_CustomerSerial_seq"'::regclass) | plain    |             |              |
 CustomerID     | text    |           |          |                                                    | extended |             |              |
 Name           | text    |           |          |                                                    | extended |             |              |
 Address        | text    |           |          |                                                    | extended |             |              |
 Email          | text    |           |          |                                                    | extended |             |              |
 Phone          | text    |           |          |                                                    | extended |             |              |
Indexes:
    "Customer_CustomerID_idx" btree ("CustomerID")
Access method: heap
```

Product table schema

```sql
postgres=# \d+ "Product"
                                                                 Table "public.Product"
    Column     |  Type   | Collation | Nullable |                     Default                      | Storage  | Compression | Stats target | Description
---------------+---------+-----------+----------+--------------------------------------------------+----------+-------------+--------------+-------------
 ProductSerial | integer |           | not null | nextval('"Product_ProductSerial_seq"'::regclass) | plain    |             |              |
 ProductID     | text    |           |          |                                                  | extended |             |              |
 Phrase        | text    |           |          |                                                  | extended |             |              |
 SKU           | text    |           |          |                                                  | extended |             |              |
 Price         | real    |           |          |                                                  | plain    |             |              |
 Color         | text    |           |          |                                                  | extended |             |              |
 Size          | text    |           |          |                                                  | extended |             |              |
 Weight        | integer |           |          |                                                  | plain    |             |              |
Indexes:
    "Product_ProductID_idx" btree ("ProductID")
Access method: heap
```

BOUGHT table schema

```sql
postgres=# \d+ "BOUGHT"
                                                                Table "public.BOUGHT"
    Column    |  Type   | Collation | Nullable |                    Default                     | Storage  | Compression | Stats target | Description
--------------+---------+-----------+----------+------------------------------------------------+----------+-------------+--------------+-------------
 BoughtSerial | integer |           | not null | nextval('"BOUGHT_BoughtSerial_seq"'::regclass) | plain    |             |              |
 CustomerID   | text    |           |          |                                                | extended |             |              |
 ProductID    | text    |           |          |                                                | extended |             |              |
Indexes:
    "BOUGHT_CustomerID_idx" btree ("CustomerID")
    "BOUGHT_ProductID_idx" btree ("ProductID")
Access method: heap
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

- with python venv

```bash
mkdir your_project
cd your_project
python3 -m venv .venv
source .venv/bin/activate
pip install agefreighter
```

- with uv

```bash
uv init your_project
cd your_project
uv venv
source .venv/bin/activate
uv add agefreighter
```

5. Make a Python script as below and locate the script in the same directory with the CSV files

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from agefreighter import Factory


async def main():
    loader = Factory.create_instance("MultiCSVFreighter")
    await loader.connect(
        dsn=os.environ["PG_CONNECTION_STRING"],
        max_connections=64,
        min_connections=4,
    )
    await loader.load(
        vertex_csv_paths=["air-routes-latest-nodes.csv"],
        vertex_labels=["airport"],
        edge_csv_paths=["air-routes-latest-edges.csv"],
        edge_types=["route"],
        graph_name="air_route",
        use_copy=True,
        drop_graph=True,
        progress=True,
    )


if __name__ == "__main__":
    import asyncio
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
```

6. Deploy Azure Database for PostgreSQL and enable Apache AGE extension on Azure Portal
   [Introducing support for Graph data in Azure Database for PostgreSQL (Preview)](https://techcommunity.microsoft.com/blog/adforpostgresql/introducing-support-for-graph-data-in-azure-database-for-postgresql-preview/4275628).

7. Set the PostgreSQL connection string as an environment variable

```shell
export PG_CONNECTION_STRING="host=xxxxxx.postgres.database.azure.com port=5432 dbname=postgres user=......"
```

8. Run the script

```shell
python3 <script_name>.py
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

## How to export the graph data from Neo4j as CSV files and load them to Apache AGE

'neo2age.py' under 'tests' directroy is a script that exports the graph data from neo4j as CSV files and load then to Apache AGE.

### Usage of neo2mcsv.py

```bash
git clone https://github.com/rifujita/agefreighter
cd agefreighter

# macOS, Linux
python3 -m venv venv
source venv/bin/activate
python3 -m pip install agefreighter aiofiles

# Windows
python -m venv venv
.\venv\Scripts\activate
python -m pip install agefreighter aiofiles
```

```bash
cd tests
# macOS, Linux
chmod 755 neo2age.py

./neo2age.py --help
usage: neo2age.py [-h] [--uri URI] [--user USER] [--password PASSWORD] [--database DATABASE] [--trial] [--chunk-size CHUNK_SIZE] [--progress] [--graphname GRAPHNAME] output_dir

Export data from Neo4j to CSV and load into Apache AGE.

positional arguments:
  output_dir            Output directory

optional arguments:
  -h, --help            show this help message and exit
  --uri URI             The URI of the Neo4j database
  --user USER           The username of the Neo4j database
  --password PASSWORD   The password of the Neo4j database
  --database DATABASE   The database of the Neo4j database
  --trial               Extract only 100 edges per relationship type
  --chunk-size CHUNK_SIZE
                        Chunk size
  --progress            Show progress
  --graphname GRAPHNAME
                        Name of the graph to be embedded in 'importer.py'

# Windows
python neo2age.py --help
usage: neo2age.py [-h] [--uri URI] [--user USER] [--password PASSWORD] [--database DATABASE] [--trial] [--chunk-size CHUNK_SIZE] [--progress] [--graphname GRAPHNAME] output_dir

Export data from Neo4j to CSV and load into Apache AGE.

positional arguments:
  output_dir            Output directory

optional arguments:
  -h, --help            show this help message and exit
  --uri URI             The URI of the Neo4j database
  --user USER           The username of the Neo4j database
  --password PASSWORD   The password of the Neo4j database
  --database DATABASE   The database of the Neo4j database
  --trial               Extract only 100 edges per relationship type
  --chunk-size CHUNK_SIZE
                        Chunk size
  --progress            Show progress
  --graphname GRAPHNAME
                        Name of the graph to be embedded in 'importer.py'
```

Before running the script, make sure you set an environment variable for the PostgreSQL.


```bash
#macOS/Linux
export PG_CONNECTION_STRING='host=**..postgres.database.azure.com port=5432 dbname=...'

# Windows
set PG_CONNECTION_STRING=host=**..postgres.database.azure.com port=5432 dbname=...

# PowerShell
$env:PG_CONNECTION_STRING="host=**..postgres.database.azure.com port=5432 dbname=..."
```

If you have a running neo4j instance in your local machine, you can use the following command to export the graph data.

```bash
./neo2age.py exported
INFO:root:Output directory '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported' does not exist. Creating it.
INFO:root:Exporting 8679 nodes for label 'Customer'
INFO:root:Exported 8679 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/customer.csv
INFO:root:Exporting 1000 nodes for label 'Product'
INFO:root:Exported 1000 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/product.csv
INFO:root:Exporting 2 nodes for label 'NO_LABEL'
INFO:root:Exported 2 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/no_label.csv
INFO:root:Exporting 20000 edges for relationship type 'BOUGHT'
INFO:root:Exported 20000 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/bought.csv
INFO:root:Exporting 1 edges for relationship type 'RELATES'
INFO:root:Exported 1 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/relates.csv
Copying vertex Customer{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/customer.csv', 'original_id': '_elementid'}...
Copying vertex Product{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/product.csv', 'original_id': '_elementid'}...
Copying vertex NO_LABEL{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/no_label.csv', 'original_id': '_elementid'}...
Copying edge BOUGHT{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/bought.csv', 'original_id': '_elementid'}...
Copying edge RELATES{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/relates.csv', 'original_id': '_elementid'}...
```

Or, you can specify '--trial' argument to check the functionality of 'neo2age.py'

```bash
./neo2age.py --trial exported
INFO:root:Output directory '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported' does not exist. Creating it.
INFO:root:Listing nodes for relationship type 'BOUGHT'
INFO:root:Listing nodes for relationship type 'RELATES'
INFO:root:Exporting 100 nodes for label 'Customer'
INFO:root:Exported 100 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/customer.csv
INFO:root:Exporting 100 nodes for label 'Product'
INFO:root:Exported 100 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/product.csv
INFO:root:Exporting 2 nodes for label 'NO_LABEL'
INFO:root:Exported 2 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/no_label.csv
INFO:root:Exporting 100 edges for relationship type 'BOUGHT'
INFO:root:Exported 100 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/bought.csv
INFO:root:Exporting 1 edges for relationship type 'RELATES'
INFO:root:Exported 1 records to /Users/rifujita/ownCloud/bin/agefreighter/tests/exported/relates.csv
Copying vertex Customer{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/customer.csv', 'original_id': '_elementid'}...
Copying vertex Product{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/product.csv', 'original_id': '_elementid'}...
Copying vertex NO_LABEL{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/no_label.csv', 'original_id': '_elementid'}...
Copying edge BOUGHT{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/bought.csv', 'original_id': '_elementid'}...
Copying edge RELATES{'csv_path': '/Users/rifujita/ownCloud/bin/agefreighter/tests/exported/relates.csv', 'original_id': '_elementid'}...
```

## Classes

- [AGEFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/agefreighter.txt)
- [AzureStorageFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/azurestoragefreighter.txt)
- [AvroFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/avrofreighter.txt)
- [CosmosGremlinFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/cosmosgremlinfreighter.txt)
- [CosmosNoSQLFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/cosmosnosqlfreighter.txt)
- [CSVFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/csvfreighter.txt)
- [MultiAzureStorageFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/multiazurestoragefreighter.txt)
- [MultiCSVFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/multicsvfreighter.txt)
- [Neo4jFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/neo4jfreighter.txt)
- [NetworkXFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/networkxfreighter.txt)
- [ParquetFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/parguetfreighter.txt)
- [PGFreighter](https://github.com/rioriost/agefreighter/blob/main/docs/pgfreighter.txt)

## Method

All the classes have the same load() method. The method loads data into a graph database.

## Arguments

- Common arguments

  - graph_name (str) : the name of the graph
  - chunk_size (int) : the number of rows to be loaded at once
  - direct_loading (bool) : if True, the data is loaded into the graph using the 'INSERT' statement, not Cypher queries
  - use_copy (bool) : if True, the data is loaded into the graph using the 'COPY' protocol
  - create_graph (bool) : if True, the graph will be created after the existing graph is dropped
  - progress (bool) : if True, the progress of the loading is shown

- Common arguments for 'Single Source' classes

  - AvroFreighter
  - AzureStorageFreighter
  - CosmosGremlinFreighter
  - Neo4jFreighter
  - NetworkXFreighter
  - ParquetFreighter
  - PGFreighter
    - start_v_label (str): Start Vertex Label
    - start_id (str): Start Vertex ID
    - start_props (list): Start Vertex Properties
    - end_v_label (str): End Vertex Label
    - end_id (str): End Vertex ID
    - end_props (list): End Vertex Properties
    - edge_type (str): Edge Type
    - edge_props (list): Edge Properties

- Class specific arguments

  - AzureStorageFreighter

    - csv_path (str): The path to the CSV file.

  - AvroFreighter

    - avro_path (str): The path to the Avro file.

  - CosmosGremlinFreighter

    - cosmos_gremlin_endpoint (str): The Cosmos Gremlin endpoint.
    - cosmos_gremlin_key (str): The Cosmos Gremlin key.
    - cosmos_username (str): The Cosmos username.
    - id_map (dict): ID Mapping

  - CosmosGremlinFreighter

    - cosmos_endpoint (str): The Cosmos endpoint.
    - cosmos_key (str): The Cosmos key.
    - cosmos_database (str): The Cosmos database.
    - cosmos_container (str): The Cosmos container.
    - id_map (dict): ID Mapping

  - CSVFreighter

    - csv_path (str): The path to the CSV file.

  - MultiAzureStorageFreighter

    - vertex_args (list): Vertex Arguments.
    - edge_args (list): Edge Arguments.

  - MultiCSVFreighter

    - vertex_csv_paths (list): The paths to the vertex CSV files.
    - vertex_labels (list): The labels of the vertices.
    - edge_csv_paths (list): The paths to the edge CSV files.
    - edge_types (list): The types of the edges.

  - Neo4jFreighter

    - neo4j_uri (str): The URI of the Neo4j database.
    - neo4j_user (str): The username of the Neo4j database.
    - neo4j_password (str): The password of the Neo4j database.
    - neo4j_database (str): The database of the Neo4j database.
    - id_map (dict): ID Mapping

  - NetworkXFreighter

    - networkx_graph (nx.Graph): The NetworkX graph.
    - id_map (dict): ID Mapping

  - ParquetFreighter

    - parquet_path (str): The path to the Parquet file.

  - PGFreighter
    - source_pg_con_string (str): The connection string of the source PostgreSQL database.
    - source_schema (str): The source schema.
    - source_tables (list): The source tables.
    - id_map (dict): ID Mapping
    - source_columns (dict): The source columns.

## Release Notes

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

## For more information about [Apache AGE](https://age.apache.org/)

- Apache AGE : https://age.apache.org/
- GitHub : https://github.com/apache/age
- Document : https://age.apache.org/age-manual/master/index.html

## License

MIT License
