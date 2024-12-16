#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from agefreighter import Factory

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AgeFreighterTester:
    def __init__(self, large_data: bool = False):
        """
        Initialize the AgeTester
        """
        log.debug("Creating AgeTester.")
        import os
        import pickle

        self.name = "AgeFreighterTester"
        self.author = "Rio Fujita"
        try:
            self.connection_string = os.environ["PG_CONNECTION_STRING"]
            self.dsn = (
                self.connection_string
                + " options='-c search_path=ag_catalog,\"$user\",public'"
            )
        except KeyError:
            print("Please set the environment variable PG_CONNECTION_STRING")
            raise KeyError
        self.data_dir = "../data/"

        if not large_data:
            self.test_flags = [[False, False], [True, False], [False, True]]
        else:
            self.test_flags = [[False, False]]

        self.target_classes = [
            {"name": "AzureStorageFreighter", "type": "transactions", "do": large_data},
            {"name": "AvroFreighter", "type": "actorfilms", "do": not large_data},
            {
                "name": "CosmosGremlinFreighter",
                "type": "actorfilms",
                "do": not large_data,
            },
            {"name": "CSVFreighter", "type": "actorfilms", "do": not large_data},
            {
                "name": "MultiCSVFreighter",
                "type": "citiescountries",
                "do": not large_data,
            },
            {"name": "Neo4jFreighter", "type": "actorfilms", "do": not large_data},
            {"name": "NetworkXFreighter", "type": "actorfilms", "do": not large_data},
            {"name": "ParquetFreighter", "type": "actorfilms", "do": not large_data},
            {"name": "PGFreighter", "type": "actorfilms", "do": not large_data},
        ]

        if not large_data:
            self.params = {
                "graph_name": "AgeTester",
                "start_v_label": "Actor",
                "start_id": "ActorID",
                "start_props": ["Actor"],
                "edge_type": "ACTED_IN",
                "end_v_label": "Film",
                "end_id": "FilmID",
                "end_props": ["Film", "Year", "Votes", "Rating"],
                "csv": f"{self.data_dir}actorfilms.csv",
                "id_map": {
                    "Actor": "ActorID",
                    "Film": "FilmID",
                },
                "vertex_csvs": [
                    f"{self.data_dir}countries.csv",
                    f"{self.data_dir}cities.csv",
                ],
                "vertex_labels": ["Country", "City"],
                "edge_csvs": [f"{self.data_dir}edges.csv"],
                "edge_types": ["has_city"],
                "chunk_size": 96,
                "drop_graph": True,
            }
        else:
            self.params = {
                "graph_name": "AgeTester",
                "start_v_label": "Customer",
                "start_id": "start_id",
                "start_props": ["name", "address", "email", "phone"],
                "edge_type": "TRANSACTION",
                "end_v_label": "Product",
                "end_id": "end_id",
                "end_props": ["description", "SKU", "price", "Color", "Size", "Weight"],
                "csv": f"{self.data_dir}transactions.csv",
                "id_map": {
                    "Customer": "start_id",
                    "Product": "end_id",
                },
                "chunk_size": 96,
                "drop_graph": True,
            }
        if not large_data:
            self.expected_results = {
                "actorfilms": {
                    "vertices": {"Actor": 9623, "Film": 44456},
                    "edges": {"ACTED_IN": 191873},
                },
                "citiescountries": {
                    "vertices": {"Country": 53, "City": 72485},
                    "edges": {"has_city": 72485},
                },
            }
        else:
            self.expected_results = {
                "transactions": {
                    "vertices": {"Customer": 10000000, "Product": 9999},
                    "edges": {"TRANSACTION": 25000604},
                },
            }

        for item in self.target_classes:
            if item["do"]:
                # for Python 3.9
                if item["name"] == "AvroFreighter":
                    self.params["source_avro"] = f"{self.data_dir}actorfilms.avro"
                if item["name"] == "CosmosGremlinFreighter":
                    try:
                        self.params["cosmos_gremlin_endpoint"] = os.environ[
                            "COSMOS_GREMLIN_ENDPOINT"
                        ]
                        self.params["cosmos_gremlin_key"] = os.environ[
                            "COSMOS_GREMLIN_KEY"
                        ]
                    except KeyError:
                        print(
                            "Please set the environment variables COSMOS_GREMLIN_ENDPOINT / COSMOS_GREMLIN_KEY"
                        )
                        raise KeyError
                    self.params["cosmos_username"] = "/dbs/db1/colls/actorfilms"
                    self.params["cosmos_pkey"] = "pk"
                if item["name"] == "Neo4jFreighter":
                    try:
                        self.params["neo4j_uri"] = os.environ["NEO4J_URI"]
                        self.params["neo4j_user"] = os.environ["NEO4J_USER"]
                        self.params["neo4j_password"] = os.environ["NEO4J_PASSWORD"]
                        self.params["neo4j_database"] = ""
                    except KeyError:
                        print(
                            "Please set the environment variables NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"
                        )
                        raise KeyError
                if item["name"] == "NetworkXFreighter":
                    self.params["networkx_graph"] = pickle.load(
                        open(f"{self.data_dir}actorfilms.pickle", "rb")
                    )
                if item["name"] == "ParquetFreighter":
                    self.params["source_parquet"] = f"{self.data_dir}actorfilms.parquet"
                if item["name"] == "PGFreighter":
                    try:
                        self.params["source_pg_con_string"] = os.environ[
                            "SRC_PG_CONNECTION_STRING"
                        ]
                    except KeyError:
                        print(
                            "Please set the environment variable SRC_PG_CONNECTION_STRING"
                        )
                        raise KeyError
                    self.params["source_tables"] = {
                        "start": "Actor",
                        "end": "Film",
                        "edges": "ACTED_IN",
                    }

    async def test_all_classes(self):
        """
        Test all the freighter classes
        """
        from agefreighter import AgeFreighter

        log.info(f"AgeFreighter version: {AgeFreighter.get_version()}")
        log.info("Testing all the freighter classes.")
        import time

        all_results = {}
        for target_class in self.target_classes:
            if target_class["do"]:
                all_results[target_class["name"]] = []
                # AzureStorageFreighter ignores direct_loading and use_copy
                for direct_loading, use_copy in self.test_flags:
                    attempt_cnt = len(all_results[target_class["name"]])
                    all_results[target_class["name"]].append({})
                    log.info(f"Instantiating {target_class['name']}.")
                    try:
                        instance = Factory.create_instance(target_class["name"])
                    except ValueError:
                        log.info(f"Invalid target class: {target_class['name']}")
                        all_results[target_class["name"]][attempt_cnt]["result"] = False
                        break
                    log.info("Connecting to PostgreSQL.")
                    try:
                        await instance.connect(
                            dsn=self.connection_string, max_connections=64
                        )
                    except Exception as e:
                        log.info(f"Failed to connect to the database: {e}")
                        all_results[target_class["name"]][attempt_cnt]["result"] = False
                        break
                    log.info(
                        f"Loading graph data to PostgreSQL with {target_class['name']}."
                    )
                    start_time = time.time()
                    await instance.load(
                        **self.params,
                        direct_loading=direct_loading,
                        use_copy=use_copy,
                    )
                    end_time = time.time()
                    result = await self.is_graph_created(
                        expected_result=self.expected_results[target_class["type"]],
                    )
                    all_results[target_class["name"]][attempt_cnt]["result"] = result
                    all_results[target_class["name"]][attempt_cnt]["seconds"] = (
                        end_time - start_time
                    )
                    all_results[target_class["name"]][attempt_cnt]["flags"] = {
                        "direct_loading": direct_loading,
                        "use_copy": use_copy,
                    }
                    message = f"Test: {target_class['name']} {'SUCCEEDED' if result else 'FAILED'}, direct_loading({direct_loading}), use_copy({use_copy}), {(end_time - start_time):.2f} seconds"
                    log.info(message)

        summary = [f"AgeFreighter version: {AgeFreighter.get_version()}"]
        summary.append("Summary of all tests are as followings:")
        for class_name, attempts in all_results.items():
            summary.append(f"Test Result for {class_name}: ")
            for idx, attempt in enumerate(attempts):
                if attempt["result"]:
                    summary.append(
                        f"                              case({idx}) {'SUCCEEDED' if attempt['result'] else 'FAILED'} direct_loading({attempt['flags']['direct_loading']}) and use_copy({attempt['flags']['use_copy']}), in {attempt['seconds']:.2f} seconds",
                    )
                else:
                    summary.append(f"                              case({idx}) FAILED")
        print("\n".join(summary))

    async def is_graph_created(self, expected_result: dict = None) -> None:
        """
        Check the number of vertices and edges in the graph
        """
        log.info("Checking the number of vertices and edges in the graph.")
        import psycopg as pg
        from psycopg.rows import namedtuple_row

        vertex_labels = list(expected_result["vertices"].keys())
        vertex_counts = list(expected_result["vertices"].values())
        edge_type = list(expected_result["edges"].keys())[0]
        edge_count = list(expected_result["edges"].values())[0]
        result = True
        graph_name = self.params["graph_name"]
        if graph_name.lower() != self.params["graph_name"]:
            graph_name = f'"{self.params["graph_name"]}"'
        with pg.connect(self.dsn) as conn:
            with conn.cursor(row_factory=namedtuple_row) as cur:
                for idx, (v_label, v_count) in enumerate(
                    zip(vertex_labels, vertex_counts)
                ):
                    cur.execute(f'SELECT COUNT(*) FROM {graph_name}."{v_label}";')
                    cnt_result = cur.fetchone()
                    result &= cnt_result.count == v_count
                cur.execute(f'SELECT COUNT(*) FROM {graph_name}."{edge_type}";')
                cnt_result = cur.fetchone()
                result &= cnt_result.count == edge_count
        return result

    async def cleanUp(self):
        """
        Clean up the graph
        """
        log.info("Cleaning up all the graphs for test.")
        import psycopg as pg

        with pg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                DO $$
                DECLARE
                    ns_value TEXT;
                BEGIN
                    FOR ns_value IN
                        SELECT name FROM ag_graph
                    LOOP
                        EXECUTE format('SELECT drop_graph(%L, true);', ns_value);
                    END LOOP;
                END $$;""")


async def main():
    tester = AgeFreighterTester(large_data=True)
    await tester.test_all_classes()
    await tester.cleanUp()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
