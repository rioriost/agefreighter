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
        self.version = "0.2.0"
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

    async def checkKeys(keys: List = [], elements: List = []):
        if not np.all(np.isin(elements, keys)):
            raise ValueError(
                f"CSV file must have {elements} columns, but {keys} columns were found."
            )

    async def createLabelType(self, label_type: str = "", value: str = "") -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                if label_type == "vertex":
                    await cur.execute(
                        sql.SQL(
                            f"SELECT create_vlabel({self.graphNameAgType}, '{value}');"
                        )
                    )
                    await cur.execute(
                        sql.SQL(
                            f'CREATE INDEX ON {self.graphName}."{value}" USING GIN (properties);'
                        )
                    )
                    await cur.execute(
                        sql.SQL(
                            f'CREATE INDEX ON {self.graphName}."{value}" USING BTREE (id);'
                        )
                    )
                elif label_type == "edge":
                    await cur.execute(
                        sql.SQL(
                            f"SELECT create_elabel({self.graphNameAgType}, '{value}');"
                        )
                    )
                    await cur.execute(
                        sql.SQL(
                            f'CREATE INDEX ON {self.graphName}."{value}" (start_id);'
                        )
                    )
                    await cur.execute(
                        sql.SQL(f'CREATE INDEX ON {self.graphName}."{value}" (end_id);')
                    )

    # create vertices
    async def createVertices(
        self,
        vertices: pd.DataFrame = None,
        vertex_label: str = "",
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        if direct_loading:
            vertices = vertices.map(
                lambda x: x.replace("'", "''") if isinstance(x, str) else x
            )
            await self.createVerticesDirectly(self, vertices, vertex_label, chunk_size)
        else:
            vertices = vertices.map(
                lambda x: x.replace("'", r"\'") if isinstance(x, str) else x
            )
            if use_copy:
                await self.copyVertices(
                    self, vertices, vertex_label, chunk_size, drop_graph
                )
            else:
                await self.createVerticesCypher(
                    self, vertices, vertex_label, chunk_size
                )

    # create vertices
    async def createEdges(
        self,
        edges: pd.DataFrame = None,
        edge_type: str = "",
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        # create edges
        if direct_loading:
            await self.createEdgesDirectly(self, edges, edge_type, chunk_size)
        else:
            if use_copy:
                await self.copyEdges(self, edges, edge_type, chunk_size, drop_graph)
            else:
                await self.createEdgesCypher(self, edges, edge_type, chunk_size)

    # create vertices via Cypher
    async def createVerticesCypher(
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
    async def createEdgesCypher(
        self, edges: pd.DataFrame = None, type: str = "", chunk_size: int = 0
    ) -> None:
        chunk_multiplier = 2
        args = []
        for i in range(0, len(edges), chunk_size * chunk_multiplier):
            parts = []
            for idx, cols in edges[i : i + chunk_size * chunk_multiplier].iterrows():
                parts.append(
                    f"MATCH (n:{cols['start_v_label']} {{id: '{cols['start_id']}'}}), (m:{cols['end_v_label']} {{id: '{cols['end_id']}'}}) CREATE (n)-[:{type}]->(m)"
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
        self, edges: pd.DataFrame = None, type: str = "", chunk_size: int = 0
    ) -> None:
        chunk_multiplier = 2
        # create idmaps to convert entry_id to id(graphid)
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                idmaps = {}
                for e_label in [
                    edges["start_v_label"][0],
                    edges["end_v_label"][0],
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
                    f"('{idmaps[str(cols['start_v_label'])][str(cols['start_id'])]}'::graphid, '{idmaps[str(cols['end_v_label'])][str(cols['end_id'])]}'::graphid)"
                )
            query = "".join(
                f"INSERT INTO {self.graphName}.\"{type}\" (start_id, end_id) VALUES {','.join(values)};"
            )
            args.append(
                sql.SQL(
                    "".join(
                        f"INSERT INTO {self.graphName}.\"{type}\" (start_id, end_id) VALUES {','.join(values)};"
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
                await asyncio.sleep(1)
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
        type: str = "",
        chunk_size: int = 0,
        drop_graph: bool = False,
    ) -> None:
        chunk_multiplier = 1000

        # create idmaps to convert entry_id to id(graphid)
        idmaps = await self.getIdMaps(self, edges=edges)

        # create queries for edges
        first_id = await self.getFirstId(self, label_type=type)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = f'COPY {self.graphName}."{type}" (id,start_id,end_id) FROM STDIN (FORMAT TEXT)'
                if drop_graph:
                    await cur.execute(f'TRUNCATE {self.graphName}."{type}"')
                    query = f'COPY {self.graphName}."{type}" (id,start_id,end_id) FROM STDIN (FORMAT TEXT, FREEZE)'
                async with cur.copy(query) as copy:
                    for i in range(0, len(edges), chunk_size * chunk_multiplier):
                        args = ""
                        for idx, cols in edges[
                            i : i + chunk_size * chunk_multiplier
                        ].iterrows():
                            start_id = idmaps[str(cols["start_v_label"])][
                                str(cols["start_id"])
                            ]
                            end_id = idmaps[str(cols["end_v_label"])][
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
                    edges["start_v_label"][0],
                    edges["end_v_label"][0],
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
    async def getFirstId(self, label_type: str = "") -> int:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                query = sql.SQL(f"SELECT id FROM ag_label WHERE name='{label_type}'")
                await cur.execute(query)
                row = await cur.fetchone()

                ENTRY_ID_BITS = 32 + 16
                ENTRY_ID_MASK = np.uint64(0x0000FFFFFFFFFFFF)
                first_id = ((np.uint64(row.id)) << ENTRY_ID_BITS) | (
                    (np.uint64(1)) & ENTRY_ID_MASK
                )

                return first_id

    # rename columns name
    async def renameColumns(df: pd.DataFrame = None) -> pd.DataFrame:
        cols_to_rename = {}
        for col in df.columns.tolist():
            if col.startswith("a().prop."):
                cols_to_rename[col] = col.replace("a().prop.", "")
            elif col.startswith("a()."):
                cols_to_rename[col] = col.replace("a().", "")
            if col.startswith("a->.prop."):
                cols_to_rename[col] = col.replace("a->.prop.", "")
            elif col.startswith("a->."):
                cols_to_rename[col] = col.replace("a->.", "")
        return df.rename(columns=cols_to_rename)

    # get start/end vertex labels
    async def getVertexLabels(
        nodes: pd.DataFrame = None,
        v_labels: List = [],
        first_source_id: str = "",
        first_target_id: str = "",
    ) -> List:
        try:
            start_v_label = [
                item["label"] for item in nodes if item["id"] == first_source_id
            ][0]
            end_v_label = v_labels[1] if v_labels[0] == start_v_label else v_labels[0]
        except IndexError:
            end_v_label = [
                item["label"] for item in nodes if item["id"] == first_target_id
            ][0]
            start_v_label = v_labels[1] if v_labels[0] == end_v_label else v_labels[0]
        return start_v_label, end_v_label

    # open connection pool
    @classmethod
    async def connect(
        cls, dsn: str = "", max_connections: int = 64, log_level=None, **kwargs
    ) -> Self:
        # to make large number of connections
        current_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, current_limit[1]))
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
        start_v_label: str = "",
        start_id: str = "",
        start_props: List = [],
        edge_type: str = "",
        end_v_label: str = "",
        end_id: str = "",
        end_props: List = [],
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
                await cls.checkKeys(
                    df.keys(),
                    [start_id] + start_props + [end_id] + end_props,
                )

                await cls.setUpGraph(cls, graph_name, drop_graph)

                await cls.createLabelType(cls, label_type="vertex", value=start_v_label)
                await cls.createLabelType(cls, label_type="vertex", value=end_v_label)
                await cls.createLabelType(cls, label_type="edge", value=edge_type)
                first_chunk = False

            # create vertices
            for v_label, id, props in zip(
                [start_v_label, end_v_label],
                [start_id, end_id],
                [start_props, end_props],
            ):
                vertices = (
                    df.loc[:, [id, *props]].drop_duplicates().rename(columns={id: "id"})
                )
                await cls.createVertices(
                    cls,
                    vertices,
                    v_label,
                    chunk_size,
                    direct_loading,
                    drop_graph,
                    use_copy,
                )

            # extract unique edges (maybe already done)
            # start_id,start_v_label,end_id,end_v_label
            edges = (
                df.loc[:, [start_id, end_id]]
                .drop_duplicates()
                .rename(columns={start_id: "start_id", end_id: "end_id"})
            )
            edges["start_v_label"] = start_v_label
            edges["end_v_label"] = end_v_label

            # create edges with tasks
            await cls.createEdges(
                cls, edges, edge_type, chunk_size, direct_loading, drop_graph, use_copy
            )

    # this is a wrapper for load_labels_from_file() / load_edges_from_file()
    @classmethod
    async def loadFromCSVs(
        cls,
        graph_name: str = "",
        vertex_csvs: List = [],
        v_labels: List = [],
        edge_csvs: List = [],
        e_types: List = [],
        chunk_size: int = 10,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        # setup graph
        await cls.setUpGraph(cls, graph_name, drop_graph)

        # create vertices
        for vertex_csv, v_label in zip(vertex_csvs, v_labels):
            first_chunk = True
            reader = pd.read_csv(vertex_csv, chunksize=1000000)
            for vertices in reader:
                if first_chunk:
                    # check if the first column is 'id'. Rest of the columns are properties
                    await cls.checkKeys(vertices.keys(), ["id"])

                    # we need to create vlabel before create vertices with tasks
                    await cls.createLabelType(cls, label_type="vertex", value=v_label)
                    first_chunk = False

                    await cls.createVertices(
                        cls,
                        vertices,
                        v_label,
                        chunk_size,
                        direct_loading,
                        drop_graph,
                        use_copy,
                    )

        # create edges
        for edge_csv, edge_type in zip(edge_csvs, e_types):
            first_chunk = True
            reader = pd.read_csv(edge_csv, chunksize=1000000)
            for edges in reader:
                if first_chunk:
                    # check if the columns include 'start_id', 'start_vertex_type', 'end_id', 'end_vertex_type'
                    await cls.checkKeys(
                        edges.keys(),
                        [
                            "start_id",
                            "start_vertex_type",
                            "end_id",
                            "end_vertex_type",
                        ],
                    )

                    # we need to create vlabel before create vertices with tasks
                    await cls.createLabelType(cls, label_type="edge", value=edge_type)
                    first_chunk = False
                edges.rename(
                    columns={
                        "start_vertex_type": "start_v_label",
                        "end_vertex_type": "end_v_label",
                    },
                    inplace=True,
                )
                await cls.createEdges(
                    cls,
                    edges,
                    edge_type,
                    chunk_size,
                    direct_loading,
                    drop_graph,
                    use_copy,
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
        # setup graph
        await cls.setUpGraph(cls, graph_name, drop_graph)

        # create vertices
        v_labels = []
        for v_label in set(nx.get_node_attributes(networkx_graph, "label").values()):
            v_labels.append(v_label)
            await cls.createLabelType(cls, label_type="vertex", value=v_label)
            nodes = [
                {"id": node, **data}
                for node, data in networkx_graph.nodes(data=True)
                if data.get("label") == v_label
            ]
            columns = [k for k in list(nodes[0].keys()) if k != "label"]
            vertices = pd.DataFrame(nodes, columns=columns)
            await cls.createVertices(
                cls,
                vertices,
                v_label,
                chunk_size,
                direct_loading,
                drop_graph,
                use_copy,
            )

        # create edges
        for edge_type in set(nx.get_edge_attributes(networkx_graph, "label").values()):
            await cls.createLabelType(cls, label_type="edge", value=edge_type)
            edges = nx.to_pandas_edgelist(networkx_graph)
            # it's a little bit tricky to get the vertex types of the start and end vertices
            # nodes keeps the last vertex information, it might be the start or end vertex. It's not guaranteed.
            first_source_id = edges["source"][0]
            first_target_id = edges["target"][0]
            start_v_label, end_v_label = await cls.getVertexLabels(
                nodes, v_labels, first_source_id, first_target_id
            )
            edges.insert(0, "start_v_label", start_v_label)
            edges.insert(0, "end_v_label", end_v_label)
            edges.rename(
                columns={"source": "start_id", "target": "end_id"}, inplace=True
            )
            await cls.createEdges(
                cls,
                edges,
                edge_type,
                chunk_size,
                direct_loading,
                drop_graph,
                use_copy,
            )

    # load data from neo4j
    # Not completed yet
    @classmethod
    async def loadFromNeo4j(
        cls,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4jpass",
        neo4j_database: str = "neo4j",
        graph_name: str = "",
        id_map: Dict = {},
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        import neo4j

        # setup graph
        await cls.setUpGraph(cls, graph_name, drop_graph)

        # get all nodes and edges from neo4j
        chunk_multiplier = 500
        async with neo4j.AsyncGraphDatabase.driver(
            uri, auth=(user, password)
        ) as driver:
            # create vertices
            records, _, _ = await driver.execute_query(
                "MATCH (n) RETURN distinct labels(n)",
                db=neo4j_database,
            )
            v_labels = []
            for record in records:
                v_label = record["labels(n)"][0]
                v_labels.append(v_label)
                # create label
                await cls.createLabelType(cls, label_type="vertex", value=v_label)
                # need to know the number of nodes
                result = await driver.execute_query(
                    f"MATCH (a:{v_label}) RETURN count(a) AS count",
                    db=neo4j_database,
                    result_transformer_=neo4j.AsyncResult.single,
                )
                cnt = result["count"]
                for i in range(0, cnt, chunk_size * chunk_multiplier):
                    vertices = await driver.execute_query(
                        f"MATCH (a:{v_label}) RETURN a SKIP $skip LIMIT $limit",
                        skip=i,
                        limit=chunk_size * chunk_multiplier,
                        db=neo4j_database,
                        result_transformer_=lambda res: res.to_df(expand=True),
                    )
                    vertices.drop(columns=["a().labels"], inplace=True)
                    vertices = await cls.renameColumns(vertices)
                    vertices.rename(columns={id_map[v_label]: "id"}, inplace=True)
                    await cls.createVertices(
                        cls,
                        vertices,
                        v_label,
                        chunk_size,
                        direct_loading,
                        drop_graph,
                        use_copy,
                    )

            # create edges
            records, _, _ = await driver.execute_query(
                "MATCH ()-[n]->() RETURN distinct type(n)",
                db=neo4j_database,
            )
            for record in records:
                edge_type = record["type(n)"]
                # create edge label
                await cls.createLabelType(cls, label_type="edge", value=edge_type)
                # need to know the number of edges
                result = await driver.execute_query(
                    f"MATCH ()-[a:{edge_type}]->() RETURN count(a) AS count",
                    result_transformer_=neo4j.AsyncResult.single,
                )
                cnt = result["count"]
                result = await driver.execute_query(
                    f"MATCH (m) - [a:{edge_type}] -> (n) RETURN m, n LIMIT 1",
                    result_transformer_=lambda res: res.to_df(expand=True),
                )
                # fix me if the nodes have multiple labels
                start_v_label = "".join(map(str, result["m().labels"][0]))
                end_v_label = "".join(map(str, result["n().labels"][0]))
                for i in range(0, cnt, chunk_size * chunk_multiplier):
                    edges = await driver.execute_query(
                        f"MATCH () - [a:{edge_type}] -> () RETURN a SKIP $skip LIMIT $limit",
                        skip=i,
                        limit=chunk_size * chunk_multiplier,
                        db=neo4j_database,
                        result_transformer_=lambda res: res.to_df(expand=True),
                    )
                    edges.drop(columns=["a->.type"], inplace=True)
                    edges = await cls.renameColumns(edges)
                    edges.rename(
                        columns={"from": "start_id", "to": "end_id"}, inplace=True
                    )
                    edges.insert(0, "start_v_label", start_v_label)
                    edges.insert(0, "end_v_label", end_v_label)
                    await cls.createEdges(
                        cls,
                        edges,
                        edge_type,
                        chunk_size,
                        direct_loading,
                        drop_graph,
                        use_copy,
                    )
