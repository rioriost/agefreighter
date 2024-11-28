import logging
import time

import asyncio
import networkx as nx
from networkx import DiGraph
import numpy as np
import pandas as pd
from psycopg import sql
from psycopg.rows import namedtuple_row
from psycopg_pool import AsyncConnectionPool
import resource
from typing import Self, List, Dict
from typing_extensions import Callable


class AgeFreighter:
    def __init__(self):
        self.pool: AsyncConnectionPool = None
        self.dsn: str = ""
        self.graphName: str = ""
        self.graphNameAgType: str = ""
        self.name = "AgeLoader"
        self.version = "0.1.7"
        self.description = "AgeFreighter is a Python package that helps you to create a graph database using Azure Database for PostgreSQL."
        self.author = "Rio Fujita"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.pool.close()

    # connect to PostgreSQL, create a graph
    async def setUpGraph(self, graph_name: str = "", drop_graph: bool = False) -> None:
        self.graphName = graph_name
        self.graphNameAgType = sql.Literal(graph_name).as_string()
        if drop_graph:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=namedtuple_row) as cur:
                    await cur.execute(
                        sql.SQL("CREATE EXTENSION IF NOT EXISTS age CASCADE")
                    )
                    await cur.execute(
                        sql.SQL(
                            f"SELECT count(*) FROM ag_graph WHERE name='{self.graphName}'"
                        )
                    )
                    if (row := await cur.fetchone()) is not None:
                        if row.count == 1:
                            await cur.execute(
                                sql.SQL(
                                    f"SELECT drop_graph({self.graphNameAgType}, true)"
                                )
                            )
                        await cur.execute(
                            sql.SQL(f"SELECT create_graph({self.graphNameAgType})")
                        )

    def checkKeys(self, keys: List = [], elements: List = []):
        if not np.all(np.isin(elements, keys)):
            raise ValueError(
                f"CSV file must have {elements} columns, but {keys} columns were found."
            )

    async def createLabel(self, label_type: str = "", label: str = "") -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                if label_type == "vertex":
                    await cur.execute(
                        sql.SQL(
                            f"SELECT create_vlabel({self.graphNameAgType}, '{label}');"
                        )
                    )
                    await cur.execute(
                        sql.SQL(
                            f'CREATE INDEX ON {self.graphName}."{label}" USING GIN (properties);'
                        )
                    )
                    await cur.execute(
                        sql.SQL(
                            f'CREATE INDEX ON {self.graphName}."{label}" USING BTREE (id);'
                        )
                    )
                elif label_type == "edge":
                    await cur.execute(
                        sql.SQL(
                            f"SELECT create_elabel({self.graphNameAgType}, '{label}');"
                        )
                    )
                    await cur.execute(
                        sql.SQL(
                            f'CREATE INDEX ON {self.graphName}."{label}" (start_id);'
                        )
                    )
                    await cur.execute(
                        sql.SQL(f'CREATE INDEX ON {self.graphName}."{label}" (end_id);')
                    )

    # create vertices via Cypher
    async def createVertices(
        self, vertices: pd.DataFrame = None, label: str = "", chunk_size: int = 0
    ) -> None:
        chunk_multiplier = 1
        args = []
        for i in range(0, len(vertices), chunk_size * chunk_multiplier):
            parts = []
            for idx, cols in vertices[i : i + chunk_size * chunk_multiplier].iterrows():
                properties = [f"{k}:'{v}'" for k, v in cols.items()]
                parts.append(f"(v{idx}:{label} {{{','.join(properties)}}})")
            query = sql.SQL(
                f"SELECT * FROM cypher({self.graphNameAgType}, $$ CREATE {','.join(parts)} $$) AS (a agtype);"
            )
            args.append(query)
        await self.executeWithTasks(self, self.executeQuery, args)

    # create edges via Cypher
    async def createEdges(
        self, edges: pd.DataFrame = None, label: str = "", chunk_size: int = 0
    ) -> None:
        chunk_multiplier = 2
        args = []
        for i in range(0, len(edges), chunk_size * chunk_multiplier):
            parts = []
            for idx, cols in edges[i : i + chunk_size * chunk_multiplier].iterrows():
                parts.append(
                    f"MATCH (n:{cols['start_vertex_type']} {{id: '{cols['start_id']}'}}), (m:{cols['end_vertex_type']} {{id: '{cols['end_id']}'}}) CREATE (n)-[:{label}]->(m)"
                )
            query = sql.SQL(
                "".join(
                    [
                        f"SELECT * FROM cypher({self.graphNameAgType}, $$ {part} $$) AS (a agtype);"
                        for part in parts
                    ]
                )
            )
            args.append(query)
        await self.executeWithTasks(self, self.executeQuery, args)

    # create vertices directly, not via Cypher
    async def createVerticesDirectly(
        self, vertices: pd.DataFrame = None, label: str = "", chunk_size: int = 0
    ) -> None:
        chunk_multiplier = 1
        args = []
        for i in range(0, len(vertices), chunk_size * chunk_multiplier):
            values = []
            for idx, cols in vertices[i : i + chunk_size * chunk_multiplier].iterrows():
                properties = [f'"{k}":"{v}"' for k, v in cols.items()]
                values.append(f"('{{{','.join(properties)}}}')")
            args.append(
                sql.SQL(
                    "".join(
                        f"INSERT INTO {self.graphName}.\"{label}\" (properties) VALUES {','.join(values)};"
                    )
                )
            )
        await self.executeWithTasks(self, self.executeQuery, args)

    # create edges directly, not via Cypher
    async def createEdgesDirectly(
        self, edges: pd.DataFrame = None, label: str = "", chunk_size: int = 0
    ) -> None:
        chunk_multiplier = 2
        # create idmaps to convert entry_id to id(graphid)
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                idmaps = {}
                for e_label in [
                    edges["start_vertex_type"][0],
                    edges["end_vertex_type"][0],
                ]:
                    query = sql.SQL(
                        f'SELECT id, properties->\'"id"\' AS entry_id FROM {self.graphName}."{e_label}"'
                    )
                    await cur.execute(query)
                    rows = await cur.fetchall()
                    idmaps[e_label] = {
                        row.entry_id.replace('"', ""): row.id for row in rows
                    }

        # create queries for edges
        args = []
        for i in range(0, len(edges), chunk_size * chunk_multiplier):
            values = []
            for idx, cols in edges[i : i + chunk_size * chunk_multiplier].iterrows():
                values.append(
                    f"('{idmaps[str(cols['start_vertex_type'])][str(cols['start_id'])]}'::graphid, '{idmaps[str(cols['end_vertex_type'])][str(cols['end_id'])]}'::graphid)"
                )
            query = "".join(
                f"INSERT INTO {self.graphName}.\"{label}\" (start_id, end_id) VALUES {','.join(values)};"
            )
            args.append(
                sql.SQL(
                    "".join(
                        f"INSERT INTO {self.graphName}.\"{label}\" (start_id, end_id) VALUES {','.join(values)};"
                    )
                )
            )
        await self.executeWithTasks(self, self.executeQuery, args)

    # execute queries with tasks
    async def executeWithTasks(self, target: Callable = None, args: List = []) -> None:
        tasks = []
        for arg in args:
            task = asyncio.create_task(target(self.pool, arg))
            tasks.append(task)
        await asyncio.gather(*tasks)

    # execute query with async connection pool
    async def executeQuery(pool: AsyncConnectionPool = None, query: str = "") -> None:
        while True:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query)
                        break
            except Exception as e:
                print(e)
                time.sleep(1)
                pass

    # create vertices via COPY protocol
    async def copyVertices(
        self,
        vertices: pd.DataFrame = None,
        label: str = "",
        chunk_size: int = 0,
        drop_graph: bool = False,
    ) -> None:
        chunk_multiplier = 1000
        first_id = await self.getFirstId(self, label)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = f'COPY {self.graphName}."{label}" FROM STDIN (FORMAT TEXT)'
                if drop_graph:
                    await cur.execute(f'TRUNCATE {self.graphName}."{label}"')
                    query = f'COPY {self.graphName}."{label}" FROM STDIN (FORMAT TEXT, FREEZE)'
                async with cur.copy(query) as copy:
                    for i in range(0, len(vertices), chunk_size * chunk_multiplier):
                        args = ""
                        for idx, cols in vertices[
                            i : i + chunk_size * chunk_multiplier
                        ].iterrows():
                            properties = [f'"{k}": "{v}"' for k, v in cols.items()]
                            args += (
                                f"{first_id + i + idx}\t{{{', '.join(properties)}}}\n"
                            )
                        await copy.write(args)
                await cur.execute("COMMIT")

    # create edges via COPY protocol
    async def copyEdges(
        self,
        edges: pd.DataFrame = None,
        label: str = "",
        chunk_size: int = 0,
        drop_graph: bool = False,
    ) -> None:
        chunk_multiplier = 1000

        # create idmaps to convert entry_id to id(graphid)
        idmaps = await self.getIdMaps(self, edges=edges)

        # create queries for edges
        first_id = await self.getFirstId(self, label=label)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = f'COPY {self.graphName}."{label}" (id,start_id,end_id) FROM STDIN (FORMAT TEXT)'
                if drop_graph:
                    await cur.execute(f'TRUNCATE {self.graphName}."{label}"')
                    query = f'COPY {self.graphName}."{label}" (id,start_id,end_id) FROM STDIN (FORMAT TEXT, FREEZE)'
                async with cur.copy(query) as copy:
                    for i in range(0, len(edges), chunk_size * chunk_multiplier):
                        args = ""
                        for idx, cols in edges[
                            i : i + chunk_size * chunk_multiplier
                        ].iterrows():
                            start_id = idmaps[str(cols["start_vertex_type"])][
                                str(cols["start_id"])
                            ]
                            end_id = idmaps[str(cols["end_vertex_type"])][
                                str(cols["end_id"])
                            ]
                            args += f"{first_id + i + idx}\t{start_id}\t{end_id}\n"
                        await copy.write(args)

    # get idmaps between entry_id and id(graphid)
    async def getIdMaps(self, edges: pd.DataFrame = None) -> Dict:
        # create idmaps to convert entry_id to id(graphid)
        idmaps = {}
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                for e_label in [
                    edges["start_vertex_type"][0],
                    edges["end_vertex_type"][0],
                ]:
                    query = sql.SQL(
                        f'SELECT id, properties->\'"id"\' AS entry_id FROM {self.graphName}."{e_label}"'
                    )
                    await cur.execute(query)
                    rows = await cur.fetchall()
                    idmaps[e_label] = {
                        row.entry_id.replace('"', ""): row.id for row in rows
                    }
        return idmaps

    # get the first id for vertex / edge
    async def getFirstId(self, label: str = "") -> int:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                query = sql.SQL(f"SELECT id FROM ag_label WHERE name='{label}'")
                await cur.execute(query)
                row = await cur.fetchone()

                ENTRY_ID_BITS = 32 + 16
                ENTRY_ID_MASK = np.uint64(0x0000FFFFFFFFFFFF)
                first_id = ((np.uint64(row.id)) << ENTRY_ID_BITS) | (
                    (np.uint64(1)) & ENTRY_ID_MASK
                )

                return first_id

    # open connection pool
    @classmethod
    async def connect(
        cls, dsn: str = "", max_connections: int = 64, log_level=None, **kwargs
    ) -> Self:
        # to make large number of connections
        current_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, current_limit[1]))

        if log_level is not None:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="[%X]",
            )
        cls.dsn = dsn + " options='-c search_path=ag_catalog,\"$user\",public'"
        cls.pool = AsyncConnectionPool(
            cls.dsn,
            max_size=max_connections,
            min_size=64,
            open=False,
            timeout=600,
            **kwargs,
        )
        await cls.pool.open()
        return cls

    # load data from single CSV
    @classmethod
    async def loadFromSingleCSV(
        cls,
        graph_name: str = "",
        csv: str = "",
        start_vertex_type: str = "",
        start_id: str = "",
        start_properties: List = [],
        edge_label: str = "",
        end_vertex_type: str = "",
        end_id: str = "",
        end_properties: List = [],
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        first_chunk = True
        reader = pd.read_csv(csv, chunksize=1000000)
        for df in reader:
            if first_chunk:
                # check if the first column is 'id'. Rest of the columns are properties
                cls.checkKeys(
                    cls,
                    df.keys(),
                    [start_id] + start_properties + [end_id] + end_properties,
                )

                await cls.setUpGraph(cls, graph_name, drop_graph)

                await cls.createLabel(cls, label_type="vertex", label=start_vertex_type)
                await cls.createLabel(cls, label_type="vertex", label=end_vertex_type)
                await cls.createLabel(cls, label_type="edge", label=edge_label)
                first_chunk = False

            start_time = time.time()
            # create vertices
            for vertex_type, id, properties in zip(
                [start_vertex_type, end_vertex_type],
                [start_id, end_id],
                [start_properties, end_properties],
            ):
                vertices = (
                    df.loc[:, [id, *properties]]
                    .drop_duplicates()
                    .rename(columns={id: "id"})
                )
                if direct_loading:
                    for i in properties:
                        if isinstance(vertices[i].values[0], str):
                            vertices[i] = vertices[i].str.replace("'", "''")
                    await cls.createVerticesDirectly(
                        cls, vertices, vertex_type, chunk_size
                    )
                else:
                    for i in properties:
                        if isinstance(vertices[i].values[0], str):
                            vertices[i] = vertices[i].str.replace("'", r"\'")
                    if use_copy:
                        await cls.copyVertices(
                            cls, vertices, vertex_type, chunk_size, drop_graph
                        )
                    else:
                        await cls.createVertices(cls, vertices, vertex_type, chunk_size)
            logging.info(
                f"loadFromSingleCSV : time to create vertices, {time.time() - start_time}, chunk_size: {chunk_size}"
            )

            start_time = time.time()
            # extract unique edges (maybe already done)
            # start_id,start_vertex_type,end_id,end_vertex_type
            edges = (
                df.loc[:, [start_id, end_id]]
                .drop_duplicates()
                .rename(columns={start_id: "start_id", end_id: "end_id"})
            )
            edges["start_vertex_type"] = start_vertex_type
            edges["end_vertex_type"] = end_vertex_type

            # create edges with tasks
            if direct_loading:
                await cls.createEdgesDirectly(cls, edges, edge_label, chunk_size)
            else:
                if use_copy:
                    await cls.copyEdges(cls, edges, edge_label, chunk_size, drop_graph)
                else:
                    await cls.createEdges(cls, edges, edge_label, chunk_size)
            logging.info(
                f"loadFromSingleCSV : time to create edges, {time.time() - start_time}, chunk_size: {chunk_size}"
            )

    # this is a wrapper for load_labels_from_file() / load_edges_from_file()
    @classmethod
    async def loadFromCSVs(
        cls,
        graph_name: str = "",
        vertex_csvs: List = [],
        vertex_labels: List = [],
        edge_csvs: List = [],
        edge_labels: List = [],
        num_per_thread: int = 3,
        chunk_size: int = 10,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        await cls.setUpGraph(cls, graph_name, drop_graph)

        # create vertices
        start_time = time.time()
        for vertex_csv, vertex_label in zip(vertex_csvs, vertex_labels):
            first_chunk = True
            reader = pd.read_csv(vertex_csv, chunksize=1000000)
            for df in reader:
                if first_chunk:
                    # check if the first column is 'id'. Rest of the columns are properties
                    cls.checkKeys(cls, df.keys(), ["id"])

                    # we need to create vlabel before create vertices with tasks
                    await cls.createLabel(cls, label_type="vertex", label=vertex_label)
                    first_chunk = False

                # create vertices with tasks
                if direct_loading:
                    await cls.createVerticesDirectly(cls, df, vertex_label, chunk_size)
                else:
                    if use_copy:
                        await cls.copyVertices(cls, df, vertex_label, chunk_size)
                    else:
                        await cls.createVertices(cls, df, vertex_label, chunk_size)
        logging.info(
            f"loadFromCSVs : time to create vertices, {time.time() - start_time}, chunk_size: {chunk_size}"
        )

        # create edges
        start_time = time.time()
        for edge_csv, edge_label in zip(edge_csvs, edge_labels):
            first_chunk = True
            reader = pd.read_csv(edge_csv, chunksize=1000000)
            for df in reader:
                if first_chunk:
                    # check if the columns include 'start_id', 'start_vertex_type', 'end_id', 'end_vertex_type'
                    cls.checkKeys(
                        cls,
                        df.keys(),
                        ["start_id", "start_vertex_type", "end_id", "end_vertex_type"],
                    )

                    # we need to create vlabel before create vertices with tasks
                    await cls.createLabel(cls, label_type="edge", label=edge_label)
                    first_chunk = False

                # create edges with tasks
                if direct_loading:
                    await cls.createEdgesDirectly(cls, df, edge_label, chunk_size)
                else:
                    if use_copy:
                        await cls.copyEdges(cls, df, edge_label, chunk_size)
                    else:
                        await cls.createEdges(cls, df, edge_label, chunk_size)
        logging.info(
            f"loadFromCSVs : time to create edges, {time.time() - start_time}, chunk_size: {chunk_size}"
        )

    # load data from networkx graph
    @classmethod
    async def loadFromNetworkx(
        cls,
        graph_name: str = "",
        networkx_graph: DiGraph = None,
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        await cls.setUpGraph(cls, graph_name, drop_graph)

        # create vertices
        start_time = time.time()
        vertex_types = []
        for vertex_type in set(
            nx.get_node_attributes(networkx_graph, "label").values()
        ):
            vertex_types.append(vertex_type)
            await cls.createLabel(cls, label_type="vertex", label=vertex_type)
            nodes = [
                {"id": node, **data}
                for node, data in networkx_graph.nodes(data=True)
                if data.get("label") == vertex_type
            ]
            columns = [k for k in list(nodes[0].keys()) if k != "label"]
            vertices = pd.DataFrame(nodes, columns=columns)
            if direct_loading:
                vertices = vertices.map(
                    lambda x: x.replace("'", "''") if isinstance(x, str) else x
                )
                await cls.createVerticesDirectly(cls, vertices, vertex_type, chunk_size)
            else:
                vertices = vertices.map(
                    lambda x: x.replace("'", r"\'") if isinstance(x, str) else x
                )
                if use_copy:
                    await cls.copyVertices(
                        cls, vertices, vertex_type, chunk_size, drop_graph
                    )
                else:
                    await cls.createVertices(cls, vertices, vertex_type, chunk_size)

        logging.info(
            f"loadFromNetworkx : time to create vertices, {time.time() - start_time}, chunk_size: {chunk_size}"
        )

        # create edges
        start_time = time.time()
        for edge_type in set(nx.get_edge_attributes(networkx_graph, "label").values()):
            await cls.createLabel(cls, label_type="edge", label=edge_type)
            edges = nx.to_pandas_edgelist(networkx_graph)
            # it's a little bit tricky to get the vertex types of the start and end vertices
            # nodes keeps the last vertex information, it might be the start or end vertex. It's not guaranteed.
            first_source_id = edges["source"][0]
            first_target_id = edges["target"][0]
            try:
                start_vertex_type = [
                    item["label"] for item in nodes if item["id"] == first_source_id
                ][0]
                end_vertex_type = (
                    vertex_types[1]
                    if vertex_types[0] == start_vertex_type
                    else vertex_types[0]
                )
            except IndexError:
                end_vertex_type = [
                    item["label"] for item in nodes if item["id"] == first_target_id
                ][0]
                start_vertex_type = (
                    vertex_types[1]
                    if vertex_types[0] == end_vertex_type
                    else vertex_types[0]
                )
            edges.insert(0, "start_vertex_type", start_vertex_type)
            edges.insert(0, "end_vertex_type", end_vertex_type)
            edges.rename(
                columns={"source": "start_id", "target": "end_id"}, inplace=True
            )
            if direct_loading:
                await cls.createEdgesDirectly(cls, edges, edge_type, chunk_size)
            else:
                if use_copy:
                    await cls.copyEdges(cls, edges, edge_type, chunk_size, drop_graph)
                else:
                    await cls.createEdges(cls, edges, edge_type, chunk_size)
        logging.info(
            f"loadFromNetworkx : time to create edges, {time.time() - start_time}, chunk_size: {chunk_size}"
        )

    # load data from neo4j
    # Not completed yet
    @classmethod
    async def loadFromNeo4j(
        cls,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        neo4j_database: str = "neo4j",
        graph_name: str = "",
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        # get all nodes and edges from neo4j
        import neo4j

        await cls.setUpGraph(cls, graph_name, drop_graph)

        chunk_multiplier = 500
        async with neo4j.AsyncGraphDatabase.driver(
            uri, auth=(user, password)
        ) as driver:
            records, _, _ = await driver.execute_query(
                "MATCH (n) RETURN distinct labels(n)",
                db=neo4j_database,
            )
            for record in records:
                label = record["labels(n)"][0]
                # create label
                await cls.createLabel(cls, label_type="vertex", label=label)
                # need to know the number of nodes
                result = await driver.execute_query(
                    f"MATCH (a:{label}) RETURN count(a) AS count",
                    db=neo4j_database,
                    result_transformer_=neo4j.AsyncResult.single,
                )
                cnt = result["count"]
                for i in range(0, cnt, chunk_size * chunk_multiplier):
                    nodes = await driver.execute_query(
                        f"MATCH (a:{label}) RETURN a SKIP $skip LIMIT $limit",
                        skip=i,
                        limit=chunk_size * chunk_multiplier,
                        db=neo4j_database,
                        result_transformer_=lambda res: res.to_df(expand=True),
                    )

                    print(nodes)
                # create vertices

            records, _, _ = await driver.execute_query(
                "MATCH ()-[n]->() RETURN distinct type(n)",
                db=neo4j_database,
            )
            for record in records:
                type = record["type(n)"]
                # create edge label
                await cls.createLabel(cls, label_type="edge", label=type)
                # need to know the number of edges
                result = await driver.execute_query(
                    f"MATCH ()-[a:{type}]->() RETURN count(a) AS count",
                    result_transformer_=neo4j.AsyncResult.single,
                )
                cnt = result["count"]
                for i in range(0, cnt, chunk_size * chunk_multiplier):
                    edges = await driver.execute_query(
                        f"MATCH (a) - [r:{type}] -> (b) RETURN r SKIP $skip LIMIT $limit",
                        skip=i,
                        limit=chunk_size * chunk_multiplier,
                        db=neo4j_database,
                        result_transformer_=lambda res: res.to_df(expand=True),
                    )
                    print(edges)
                    # create edges
