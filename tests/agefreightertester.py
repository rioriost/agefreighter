#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from agefreighter import AgeFreighter, Factory
import sys

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AgeFreighterTester:
    name = "AgeFreighterTester"
    version = "0.6.0"
    author = "Rio Fujita"

    @classmethod
    def get_version(cls) -> str:
        """
        Get the version of the AgeFreighterTester package.

        Returns:
            str: The version of the AgeFreighterTester package.
        """
        log.debug(f"Getting version, in {sys._getframe().f_code.co_name}.")
        return cls.version

    def __init__(
        self,
        cls: dict = {},
        chunk_size: int = 96,
        direct_loading: bool = False,
        use_copy: bool = False,
        **kwargs,
    ):
        """
        Initialize the AgeTester

        Args:
            cls (dict): Freight class to be tested
            chunk_size (int): Chunk size for loading data
            direct_loading (bool): Use direct loading
            use_copy (bool): Use COPY command for loading data
            kwargs (dict): Additional parameters
        """
        log.debug("Creating AgeTester.")
        import os
        import pickle

        self.name = "AgeFreighterTester"
        self.author = "Rio Fujita"
        try:
            self.dsn = (
                os.environ["PG_CONNECTION_STRING"]
                + " options='-c search_path=ag_catalog,\"$user\",public'"
            )
        except KeyError:
            print("Please set the environment variable PG_CONNECTION_STRING")
            raise KeyError
        self.data_dir = "../data/"

        self.tgt_cls = cls
        self.chunk_size = chunk_size
        self.direct_loading = direct_loading
        self.use_copy = use_copy

        if cls["type"] == "actorfilms":
            self.params = {
                "graph_name": "AgeTester",
                "start_v_label": "Actor",
                "start_id": "ActorID",
                "start_props": ["Actor"],
                "edge_type": "ACTED_IN",
                "edge_props": ["Genre", "Director"],
                "end_v_label": "Film",
                "end_id": "FilmID",
                "end_props": ["Film", "Year", "Votes", "Rating"],
                "csv": f"{self.data_dir}actorfilms2.csv",
                "id_map": {
                    "Actor": "ActorID",
                    "Film": "FilmID",
                },
            }
            self.expected_results = {
                "vertices": {"Actor": 9623, "Film": 44456},
                "edges": {"ACTED_IN": 191873},
            }
        elif cls["type"] == "citiescountries":
            self.params = {
                "vertex_csvs": [
                    f"{self.data_dir}countries.csv",
                    f"{self.data_dir}cities.csv",
                ],
                "vertex_labels": ["Country", "City"],
                "edge_csvs": [f"{self.data_dir}edges2.csv"],
                "edge_types": ["has_city"],
                "graph_name": "AgeTester",
            }
            self.expected_results = {
                "vertices": {"Country": 53, "City": 72485},
                "edges": {"has_city": 72485},
            }
        elif cls["type"] == "large_transactions":
            self.params = {
                "graph_name": "AgeTester",
                "start_v_label": "Customer",
                "start_id": "start_id",
                "start_props": ["name", "address", "email", "phone"],
                "edge_type": "TRANSACTION",
                "edge_props": [],
                "end_v_label": "Product",
                "end_id": "end_id",
                "end_props": ["description", "SKU", "price", "Color", "Size", "Weight"],
                "csv": f"{self.data_dir}transactions.csv",
                "id_map": {
                    "Customer": "start_id",
                    "Product": "end_id",
                },
            }
            self.expected_results = {
                "vertices": {"Customer": 10000000, "Product": 9999},
                "edges": {"TRANSACTION": 25000604},
            }
        elif cls["type"] == "small_transactions":
            self.params = {
                "graph_name": "AgeTester",
                "start_v_label": "Customer",
                "start_id": "start_id",
                "start_props": ["name", "address", "email", "phone"],
                "edge_type": "TRANSACTION",
                "edge_props": ["Prop1", "Prop2"],
                "end_v_label": "Product",
                "end_id": "end_id",
                "end_props": ["desc", "SKU", "price", "Color", "Size", "Weight"],
                "csv": f"{self.data_dir}transactions_small.csv",
                "id_map": {
                    "Customer": "start_id",
                    "Product": "end_id",
                },
            }
            self.expected_results = {
                "vertices": {"Customer": 10000, "Product": 9119},
                "edges": {"TRANSACTION": 25166},
            }
        else:
            raise ValueError("Invalid freighter class")

        # common parameters for all freighter classes
        self.params["graph_name"] = "AgeTester"
        self.params["create_graph"] = True

        # Additional parameters for each freighter class
        # for Python 3.9
        if cls["name"] == "AvroFreighter":
            self.params["source_avro"] = f"{self.data_dir}actorfilms2.avro"
        if cls["name"] == "CosmosGremlinFreighter":
            try:
                self.params["cosmos_gremlin_endpoint"] = os.environ[
                    "COSMOS_GREMLIN_ENDPOINT"
                ]
                self.params["cosmos_gremlin_key"] = os.environ["COSMOS_GREMLIN_KEY"]
            except KeyError:
                print(
                    "Please set the environment variables COSMOS_GREMLIN_ENDPOINT / COSMOS_GREMLIN_KEY"
                )
                raise KeyError
            self.params["cosmos_username"] = "/dbs/db1/colls/actorfilms2"
            self.params["cosmos_pkey"] = "pk"
        if cls["name"] == "Neo4jFreighter":
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
        if cls["name"] == "NetworkXFreighter":
            self.params["networkx_graph"] = pickle.load(
                open(f"{self.data_dir}actorfilms2.pickle", "rb")
            )
        if cls["name"] == "ParquetFreighter":
            self.params["source_parquet"] = f"{self.data_dir}actorfilms2.parquet"
        if cls["name"] == "PGFreighter":
            try:
                self.params["source_pg_con_string"] = os.environ[
                    "SRC_PG_CONNECTION_STRING"
                ]
            except KeyError:
                print("Please set the environment variable SRC_PG_CONNECTION_STRING")
                raise KeyError
            self.params["source_tables"] = {
                "start": "Actor",
                "end": "Film",
                "edges": "ACTED_IN",
            }

    async def do_test(self) -> dict:
        """
        Test freighter classes

        Returns:
            dict: Test results
        """
        import time

        log.info(f"Instantiating {self.tgt_cls['name']}.")
        try:
            instance = Factory.create_instance(self.tgt_cls["name"])
        except ValueError:
            log.info(f"Invalid target class: {self.tgt_cls['name']}")
            return
        log.info("Connecting to PostgreSQL.")
        try:
            await instance.connect(dsn=self.dsn, max_connections=64)
        except Exception as e:
            log.info(f"Failed to connect to the database: {e}")
            return
        log.info(f"Loading graph data to PostgreSQL with {self.tgt_cls['name']}.")
        log.info(f"Parameters: {self.params}")

        start_time = time.time()
        await instance.load(
            **self.params,
            direct_loading=self.direct_loading,
            use_copy=self.use_copy,
        )
        end_time = time.time()

        result = await self.is_graph_created(
            expected_result=self.expected_results,
        )
        message = f"Test for {self.tgt_cls['name']}, chunk_size({self.chunk_size}), direct_loading({self.direct_loading}), use_copy({self.use_copy}): {'SUCCEEDED' if result else 'FAILED'}, {(end_time - start_time):.2f} seconds"
        log.info(message)
        return {
            "result": result,
            "flags": {
                "chunk_size": self.chunk_size,
                "direct_loading": self.direct_loading,
                "use_copy": self.use_copy,
            },
            "seconds": end_time - start_time,
        }

    async def is_graph_created(self, expected_result: dict = None) -> bool:
        """
        Check the number of vertices and edges in the graph

        Args:
            expected_result (dict): Expected number of vertices and edges

        Returns:
            bool: True if the number of vertices and edges are as expected
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
                    try:
                        cur.execute(f'SELECT COUNT(*) FROM {graph_name}."{v_label}";')
                        cnt_result = cur.fetchone()
                        result &= cnt_result.count == v_count
                    except Exception as e:
                        log.info(f"Failed to count vertices: {e}")
                        result = False
                try:
                    cur.execute(f'SELECT COUNT(*) FROM {graph_name}."{edge_type}";')
                    cnt_result = cur.fetchone()
                    result &= cnt_result.count == edge_count
                except Exception as e:
                    log.info(f"Failed to count edges: {e}")
                    result = False
        return result

    async def cleanUp(self) -> None:
        """
        Clean up the graph

        Returns:
            None
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


def show_summary(all_results: list = []) -> None:
    """
    Show the summary of all tests

    Args:
        all_results (list): List of test results

    Returns:
        None

    """
    summary = [f"AgeFreighter version: {AgeFreighter.get_version()}"]
    summary.append("Summary of all tests are as followings:")
    for results in all_results:
        summary.append(
            f"Test for {results['class_name']}, chunk_size({results['flags']['chunk_size']}), direct_loading({results['flags']['direct_loading']}), use_copy({results['flags']['use_copy']}): {'SUCCEEDED' if results['result'] else 'FAILED'},  {results['seconds']:.2f} seconds"
        )
    print("\n".join(summary))


async def main():
    log.info(f"AgeFreighter version: {AgeFreighter.get_version()}")

    target_classes = [
        {"name": "AzureStorageFreighter", "type": "small_transactions", "do": True},
        {"name": "AvroFreighter", "type": "actorfilms", "do": True},
        {"name": "CosmosGremlinFreighter", "type": "actorfilms", "do": True},
        {"name": "CSVFreighter", "type": "actorfilms", "do": True},
        {"name": "MultiCSVFreighter", "type": "citiescountries", "do": True},
        {"name": "Neo4jFreighter", "type": "actorfilms", "do": True},
        {"name": "NetworkXFreighter", "type": "actorfilms", "do": True},
        {"name": "ParquetFreighter", "type": "actorfilms", "do": True},
        {"name": "PGFreighter", "type": "actorfilms", "do": True},
    ]
    chunk_size = 96
    all_results = []
    for cls in target_classes:
        if cls["do"]:
            # AzureStorageFreighter ignores direct_loading and use_copy
            if cls["name"] == "AzureStorageFreighter":
                test_flags = [[chunk_size, False, False]]
            else:
                test_flags = [
                    [chunk_size, False, False],
                    [chunk_size, True, False],
                    [chunk_size, False, True],
                ]
            for chunk_size, direct_loading, use_copy in test_flags:
                try:
                    tester = AgeFreighterTester(
                        cls=cls,
                        chunk_size=chunk_size,
                        direct_loading=direct_loading,
                        use_copy=use_copy,
                    )
                except Exception as e:
                    log.info(f"Failed to create AgeFreighterTester: {e}")
                    continue
                all_results.append(
                    {"class_name": cls["name"], **await tester.do_test()}
                )
                # await tester.cleanUp()
    show_summary(all_results)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
