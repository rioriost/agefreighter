import logging
import time

import asyncio
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from psycopg.rows import namedtuple_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from typing import Self
from typing_extensions import Callable

class AgeFreighter:
    def __init__(self):
        self.pool = None
        self.dsn = None
        self.graphName = None
        self.graphNameAgType = None
        self.name = "AgeLoader"
        self.version = "0.0.1"
        self.description = "AgeLoader is a Python package that helps you to create a graph database using Azure Database for PostgreSQL."
        self.author = "Rio Fujita"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.pool.close()

    # connect to PostgreSQL, create a graph
    async def setUpGraph(self, graph_name: str = None) -> None:
        self.graphName = graph_name
        self.graphNameAgType = sql.Literal(graph_name).as_string()
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                await cur.execute(sql.SQL(f"SELECT count(*) FROM ag_graph WHERE name='{self.graphName}'"))
                if (row := await cur.fetchone()) is not None:
                    if row.count == 1:
                        await cur.execute(sql.SQL(f"SELECT drop_graph({self.graphNameAgType}, true)"))
                    await cur.execute(sql.SQL(f"SELECT create_graph({self.graphNameAgType})"))

    def checkKeys(self, keys: tuple = None, elements: tuple = None):
        if np.all(np.isin(elements, keys)) == False:
            raise ValueError(f"CSV file must have {elements} columns, but {keys} columns were found.")

    async def createLabel(self, label_type: str = None, label: str = None) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                if label_type == 'vertex':
                    await cur.execute(sql.SQL(f"SELECT create_vlabel({self.graphNameAgType}, '{label}');"))
                    await cur.execute(sql.SQL(f'CREATE INDEX ON {self.graphName}."{label}" USING GIN (properties);'))
                    await cur.execute(sql.SQL(f'CREATE INDEX ON {self.graphName}."{label}" USING BTREE (id);'))
                elif label_type == 'edge':
                    await cur.execute(sql.SQL(f"SELECT create_elabel({self.graphNameAgType}, '{label}');"))
                    await cur.execute(sql.SQL(f'CREATE INDEX ON {self.graphName}."{label}" (start_id);'))
                    await cur.execute(sql.SQL(f'CREATE INDEX ON {self.graphName}."{label}" (end_id);'))

    # create vertices via Cypher
    async def createVertices(self, vertices: pd.DataFrame = None, label: str = None, chunk_size: int = None) -> None:
        args = []
        for i in range(0, len(vertices), chunk_size):
            parts = []
            for idx, cols in vertices[i: i + chunk_size].iterrows():
                properties = [f"{k}:'{v}'" for k, v in cols.items()]
                parts.append(f"(v{idx}:{label} {{{','.join(properties)}}})")
            query = sql.SQL(f"SELECT * FROM cypher({self.graphNameAgType}, $$ CREATE {','.join(parts)} $$) AS (a agtype);")
            args.append(query)
        await self.executeWithTasks(self, self.executeQuery, args)

    # create edges via Cypher
    async def createEdges(self, edges: pd.DataFrame = None, label: str = None, chunk_size: int = None) -> None:
        args = []
        for i in range(0, len(edges), chunk_size * 2):
            parts = []
            for idx, cols in edges[i: i + chunk_size * 2].iterrows():
                parts.append(f"MATCH (n:{cols['start_vertex_type']} {{id: '{cols['start_id']}'}}), (m:{cols['end_vertex_type']} {{id: '{cols['end_id']}'}}) CREATE (n)-[:{label}]->(m)")
            query = sql.SQL(''.join([f"SELECT * FROM cypher({self.graphNameAgType}, $$ {part} $$) AS (a agtype);" for part in parts]))
            args.append(query)
        await self.executeWithTasks(self, self.executeQuery, args)

    # create vertices directly, not via Cypher
    async def createVerticesDirectly(self, vertices: pd.DataFrame = None, label: str = None, chunk_size: int = None) -> None:
        args = []
        for i in range(0, len(vertices), chunk_size):
            values = []
            for idx, cols in vertices[i: i + chunk_size].iterrows():
                properties = [f'"{k}":"{v}"' for k, v in cols.items()]
                values.append(f"('{{{','.join(properties)}}}')")
            args.append(sql.SQL(''.join(f"INSERT INTO {self.graphName}.\"{label}\" (properties) VALUES {','.join(values)};")))
        await self.executeWithTasks(self, self.executeQuery, args)

    # create edges directly, not via Cypher
    async def createEdgesDirectly(self, edges: pd.DataFrame = None, label: str = None, chunk_size: int = None) -> None:
        # create idmaps to convert entry_id to id(graphid)
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory = namedtuple_row) as cur:
                idmaps = {}
                for e_label in [edges['start_vertex_type'][0], edges['end_vertex_type'][0]]:
                    query = sql.SQL(f"SELECT id, properties->'\"id\"' AS entry_id FROM {self.graphName}.\"{e_label}\"")
                    await cur.execute(query)
                    rows = await cur.fetchall()
                    idmaps[e_label] = {row.entry_id.replace('"', ''): row.id for row in rows}

        # create queries for edges
        args = []
        for i in range(0, len(edges), chunk_size * 2):
            values = []
            for idx, cols in edges[i: i + chunk_size * 2].iterrows():
                values.append(f"('{idmaps[str(cols['start_vertex_type'])][str(cols['start_id'])]}'::graphid, '{idmaps[str(cols['end_vertex_type'])][str(cols['end_id'])]}'::graphid)")
            query = ''.join(f"INSERT INTO {self.graphName}.\"{label}\" (start_id, end_id) VALUES {','.join(values)};")
            args.append(sql.SQL(''.join(f"INSERT INTO {self.graphName}.\"{label}\" (start_id, end_id) VALUES {','.join(values)};")))
        await self.executeWithTasks(self, self.executeQuery, args)

    # execute queries with tasks
    async def executeWithTasks(self, target: Callable = None, args: tuple = None) -> None:
        tasks = []
        for arg in args:
            task = asyncio.create_task(target(self.pool, arg))
            tasks.append(task)
        await asyncio.gather(*tasks)

    # execute query with async connection pool
    async def executeQuery(pool: AsyncConnectionPool = None, query: str = None) -> None:
        while True:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query)
                        break
            except PoolTimeout as e:
                print(e)
                time.sleep(1)
                pass

    # open connection pool
    @classmethod
    async def connect(cls, dsn: str = None, log_level = None, **kwargs) -> Self:
        if log_level is not None:
            logging.basicConfig(
                level = log_level,
                format = "%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt = "[%X]"
            )
        cls.dsn = dsn + " options='-c search_path=ag_catalog,\"$user\",public'"
        cls.pool = AsyncConnectionPool(cls.dsn, max_size = 192, min_size = 64, open = False, timeout = 600, **kwargs)
        await cls.pool.open()
        return cls

    # load data from single CSV
    @classmethod
    async def loadFromSingleCSV(
        cls,
        graph_name: str = None,
        csv: str = None,
        start_vertex_type: str = None,
        start_id: str = None,
        start_properties: tuple = None,
        edge_label: str = None,
        end_vertex_type: str = None,
        end_id: str = None,
        end_properties: tuple = None,
        chunk_size = 3,
        direct_loading = False
        ) -> None:

        df = pd.read_csv(csv)

        # check if the first column is 'id'. Rest of the columns are properties
        cls.checkKeys(cls, df.keys(), [start_id] + start_properties + [end_id] + end_properties)

        await cls.setUpGraph(cls, graph_name)

        await cls.createLabel(cls, label_type="vertex", label=start_vertex_type)
        await cls.createLabel(cls, label_type="vertex", label=end_vertex_type)
        await cls.createLabel(cls, label_type="edge", label=edge_label)

        start_time = time.time()
        # create vertices
        for vertex_type, id, properties in zip([start_vertex_type, end_vertex_type], [start_id, end_id], [start_properties, end_properties]):
            vertices = df.loc[:, [id, *properties]].drop_duplicates().rename(columns={id: 'id'})
            if direct_loading is True:
                for i in properties:
                    if type(vertices[i].values[0]) == str:
                        vertices[i] = vertices[i].str.replace("'", "''")
            else:
                for i in properties:
                    if type(vertices[i].values[0]) == str:
                        vertices[i] = vertices[i].str.replace("'", r"\'")
            if direct_loading is True:
                await cls.createVerticesDirectly(cls, vertices, vertex_type, chunk_size)
            else:
                await cls.createVertices(cls, vertices, vertex_type, chunk_size)
        logging.info(f"loadFromSingleCSV : time to create vertices, {time.time() - start_time}, chunk_size: {chunk_size}")

        start_time = time.time()
        # extract unique edges (maybe already done)
        # start_id,start_vertex_type,end_id,end_vertex_type
        edges = df.loc[:, [start_id, end_id]].drop_duplicates().rename(columns={start_id: 'start_id', end_id: 'end_id'})
        edges['start_vertex_type'] = start_vertex_type
        edges['end_vertex_type'] = end_vertex_type

        # create edges with tasks
        if direct_loading is True:
            await cls.createEdgesDirectly(cls, edges, edge_label, chunk_size)
        else:
            await cls.createEdges(cls, edges, edge_label, chunk_size)
        logging.info(f"loadFromSingleCSV : time to create edges, {time.time() - start_time}, chunk_size: {chunk_size}")

    # this is a wrapper for load_labels_from_file() / load_edges_from_file()
    @classmethod
    async def loadFromCSVs(
        cls,
        graph_name: str = None,
        vertex_csvs: tuple = None, vertex_labels: tuple = None,
        edge_csvs: tuple = None, edge_labels: tuple = None,
        num_per_thread: int = 3,
        chunk_size: int = 10,
        direct_loading = False
        ) -> None:

        await cls.setUpGraph(cls, graph_name)

        # create vertices
        start_time = time.time()
        for vertex_csv, vertex_label in zip(vertex_csvs, vertex_labels):
            df = pd.read_csv(vertex_csv)

            # check if the first column is 'id'. Rest of the columns are properties
            cls.checkKeys(cls, df.keys(), ['id'])

            # we need to create vlabel before create vertices with tasks
            await cls.createLabel(cls, label_type="vertex", label=vertex_label)

            # create vertices with tasks
            if direct_loading is True:
                await cls.createVerticesDirectly(cls, df, vertex_label, chunk_size)
            else:
                await cls.createVertices(cls, df, vertex_label, chunk_size)
        logging.info(f"loadFromCSVs : time to create vertices, {time.time() - start_time}, chunk_size: {chunk_size}")

        # create edges
        start_time = time.time()
        for edge_csv, edge_label in zip(edge_csvs, edge_labels):
            df = pd.read_csv(edge_csv)

            # check if the columns include 'start_id', 'start_vertex_type', 'end_id', 'end_vertex_type'
            cls.checkKeys(cls, df.keys(), ['start_id', 'start_vertex_type', 'end_id', 'end_vertex_type'])

            # we need to create vlabel before create vertices with tasks
            await cls.createLabel(cls, label_type="edge", label=edge_label)

            # create edges with tasks
            if direct_loading is True:
                await cls.createEdgesDirectly(cls, df, edge_label, chunk_size)
            else:
                await cls.createEdges(cls, df, edge_label, chunk_size)
        logging.info(f"loadFromCSVs : time to create edges, {time.time() - start_time}, chunk_size: {chunk_size}")
