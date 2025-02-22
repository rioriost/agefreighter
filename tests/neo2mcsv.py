#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import asyncio
import concurrent.futures
import logging
import os
from typing import List, Dict, Any

import aiofiles
from neo4j import GraphDatabase, Result

logging.basicConfig(level=logging.INFO)


class Neo4jExporter:
    def __init__(
        self,
        output_dir: str,
        uri: str,
        user: str,
        password: str,
        database: str,
        trial: bool,
        chunk_size: int,
        progress: bool,
    ):
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            logging.info(
                f"Output directory '{self.output_dir}' does not exist. Creating it."
            )
            os.makedirs(self.output_dir)
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.trial = trial
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")
        self.chunk_size = chunk_size
        self.progress = progress

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            logging.error(f"Error connecting to Neo4j: {e}")
            raise

    def get_labels(self) -> List[str]:
        with self.driver.session() as session:
            query = "CALL db.labels() YIELD label RETURN label;"
            result: Result = session.run(query)
            labels = [record["label"] for record in result]
        return labels

    def get_relationship_types(self) -> List[str]:
        with self.driver.session() as session:
            query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType;"
            result: Result = session.run(query)
            types = [record["relationshipType"] for record in result]
        return types

    def _count_nodes(self, label: str) -> int:
        with self.driver.session() as session:
            query = f"MATCH (n:{label}) RETURN COUNT(n) AS cnt"
            result: Result = session.run(query)
            record = result.single()
            if record is None:
                return 0
            cnt = int(record["cnt"])
        return cnt

    def _count_edges(self, rel_type: str) -> int:
        with self.driver.session() as session:
            query = f"MATCH (m)-[r:{rel_type}]->(n) RETURN COUNT(r) AS cnt"
            result: Result = session.run(query)
            record = result.single()
            if record is None:
                return 0
            cnt = int(record["cnt"])
        return cnt

    @staticmethod
    def _fetch_nodes_chunk(
        label: str, skip: int, limit: int, uri: str, user: str, password: str
    ) -> List[Dict[str, Any]]:
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                query = f"MATCH (n:{label}) SKIP {skip} LIMIT {limit} RETURN n"
                result = session.run(query)
                data = [
                    {"id": record["n"].element_id, **record["n"]._properties}
                    for record in result
                ]
            return data
        except Exception as e:
            logging.error(
                f"Error fetching nodes for label '{label}' (skip {skip}): {e}"
            )
            return []
        finally:
            try:
                driver.close()
            except Exception:
                pass

    @staticmethod
    def _fetch_edge_chunk(
        rel_type: str, skip: int, limit: int, uri: str, user: str, password: str
    ) -> List[Dict[str, Any]]:
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                query = f"MATCH (m)-[r:{rel_type}]->(n) SKIP {skip} LIMIT {limit} RETURN m, r, n"
                result = session.run(query)
                data: List[Dict[str, Any]] = []
                for record in result:
                    m_labels = list(record["m"].labels) if record["m"].labels else []
                    n_labels = list(record["n"].labels) if record["n"].labels else []
                    for m_label in m_labels:
                        for n_label in n_labels:
                            entry = {
                                "id": record["r"].element_id,
                                "start_id": record["m"].element_id,
                                "start_vertex_type": m_label,
                                "end_id": record["n"].element_id,
                                "end_vertex_type": n_label,
                                **record["r"]._properties,
                            }
                            data.append(entry)
            return data
        except Exception as e:
            logging.error(
                f"Error fetching edges for type '{rel_type}' (skip {skip}): {e}"
            )
            return []
        finally:
            try:
                driver.close()
            except Exception:
                pass

    async def write_csv(self, file_name: str, data: List[Dict[str, Any]]) -> None:
        if not data:
            logging.info(f"No data to write for '{file_name}'.")
            return

        file_path = os.path.join(self.output_dir, f"{file_name.lower()}.csv")
        headers = list(data[0].keys())
        async with aiofiles.open(file_path, "w") as f:
            await f.write(",".join(f'"{h}"' for h in headers) + "\n")
            for row in data:
                line = ",".join(f'"{row.get(h, "")}"' for h in headers)
                await f.write(line + "\n")
        logging.info(f"Exported {len(data)} records to {file_path}")

    async def export_nodes(
        self, pool: concurrent.futures.ProcessPoolExecutor
    ) -> Dict[str, List[str]]:
        loop = asyncio.get_running_loop()
        vertex_args: Dict[str, List[str]] = {
            "vertex_labels": [],
            "vertex_csv_paths": [],
        }
        labels = self.get_labels()
        for label in labels:
            count = self._count_nodes(label)
            logging.info(f"Exporting {count} nodes for label '{label}'")
            tasks = [
                loop.run_in_executor(
                    pool,
                    self._fetch_nodes_chunk,
                    label,
                    skip,
                    self.chunk_size,
                    self.uri,
                    self.user,
                    self.password,
                )
                for skip in range(0, count, self.chunk_size)
            ]
            chunks = await asyncio.gather(*tasks)
            all_data = [item for sublist in chunks for item in sublist]
            await self.write_csv(label, all_data)
            file_path = os.path.join(self.output_dir, f"{label.lower()}.csv")
            vertex_args["vertex_labels"].append(label)
            vertex_args["vertex_csv_paths"].append(file_path)
        return vertex_args

    async def export_edges(
        self, pool: concurrent.futures.ProcessPoolExecutor
    ) -> Dict[str, List[str]]:
        loop = asyncio.get_running_loop()
        edge_args: Dict[str, List[str]] = {"edge_types": [], "edge_csv_paths": []}
        types = self.get_relationship_types()
        for rel_type in types:
            count = self._count_edges(rel_type)
            if self.trial:
                logging.info(
                    "Trial mode enabled. Limiting export to 100 edges per relationship type."
                )
                count = min(count, 100)
            logging.info(f"Exporting {count} edges for relationship type '{rel_type}'")
            tasks = [
                loop.run_in_executor(
                    pool,
                    self._fetch_edge_chunk,
                    rel_type,
                    skip,
                    self.chunk_size,
                    self.uri,
                    self.user,
                    self.password,
                )
                for skip in range(0, count, self.chunk_size)
            ]
            chunks = await asyncio.gather(*tasks)
            all_data = [item for sublist in chunks for item in sublist]
            await self.write_csv(rel_type, all_data)
            file_path = os.path.join(self.output_dir, f"{rel_type.lower()}.csv")
            edge_args["edge_types"].append(rel_type)
            edge_args["edge_csv_paths"].append(file_path)
        return edge_args

    async def export(self) -> Dict[str, Dict[str, List[str]]]:
        pool = concurrent.futures.ProcessPoolExecutor()
        try:
            nodes_files, edges_files = await asyncio.gather(
                self.export_nodes(pool), self.export_edges(pool)
            )
        finally:
            pool.shutdown()
        self.driver.close()
        return {"nodes": nodes_files, "edges": edges_files}


class CodeGenerator:
    TEMPLATE = """#!/usr/bin/env python3
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
        graph_name="{graph_name}",
        vertex_csv_paths=[
{vertex_csv_paths}
        ],
        vertex_labels=[{vertex_labels}],
        edge_csv_paths=[
{edge_csv_paths}
        ],
        edge_types=[{edge_types}],
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
"""

    def __init__(self, graph_name: str, output_dir: str, arguments_dict: dict):
        self.graph_name = graph_name
        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            logging.info(
                f"Output directory '{self.output_dir}' does not exist. Creating it."
            )
            os.makedirs(self.output_dir)
        self.arguments_dict = arguments_dict

    def generate_code(self) -> None:
        code = self.TEMPLATE.format(
            graph_name=self.graph_name,
            vertex_csv_paths=",\n".join(
                [
                    " " * 12 + f'"{path}"'
                    for path in self.arguments_dict["nodes"]["vertex_csv_paths"]
                ]
            ),
            vertex_labels=", ".join(
                [
                    f'"{label}"'
                    for label in self.arguments_dict["nodes"]["vertex_labels"]
                ]
            ),
            edge_csv_paths=",\n".join(
                [
                    " " * 12 + f'"{path}"'
                    for path in self.arguments_dict["edges"]["edge_csv_paths"]
                ]
            ),
            edge_types=", ".join(
                [f'"{etype}"' for etype in self.arguments_dict["edges"]["edge_types"]]
            ),
        )
        with open(os.path.join(self.output_dir, "importer.py"), "w") as f:
            f.write(code)
            print(f"importer.py file under {self.output_dir} created successfully")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export data from Neo4j to CSV")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--uri",
        type=str,
        default="bolt://localhost:7687",
        help="The URI of the Neo4j database",
    )
    parser.add_argument(
        "--user", type=str, default="neo4j", help="The username of the Neo4j database"
    )
    parser.add_argument(
        "--password",
        type=str,
        default="neo4jpass",
        help="The password of the Neo4j database",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="neo4j",
        help="The database of the Neo4j database",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        default=False,
        help="Extract only 100 edges per relationship type",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument(
        "--progress", action="store_true", default=True, help="Show progress"
    )
    parser.add_argument(
        "--graphname",
        type=str,
        default="FROM_NEO4J",
        help="Name of the graph to be embedded in 'importer.py'",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_arguments()
    exporter = Neo4jExporter(
        output_dir=args.output_dir,
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        trial=args.trial,
        chunk_size=args.chunk_size,
        progress=args.progress,
    )
    result = await exporter.export()

    cg = CodeGenerator(
        graph_name=args.graphname,
        output_dir=args.output_dir,
        arguments_dict=result,
    )
    cg.generate_code()


if __name__ == "__main__":
    asyncio.run(main())
