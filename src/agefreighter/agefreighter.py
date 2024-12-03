import asyncio
import networkx as nx
from networkx import DiGraph
import numpy as np
import pandas as pd
from psycopg import sql
from psycopg.rows import namedtuple_row
from psycopg_pool import AsyncConnectionPool
import resource
from typing_extensions import Callable


class AgeFreighter:
    """
    AgeFreighter is a Python package that helps you to create a graph database using Azure Database for PostgreSQL.
    """

    def __init__(self):
        self.pool: AsyncConnectionPool = None
        self.dsn: str = ""
        self.graphName: str = ""
        self.name = "AgeLoader"
        self.version = "0.3.0"
        self.author = "Rio Fujita"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.pool.close()

    async def setUpGraph(self, graph_name: str = "", drop_graph: bool = False) -> None:
        """
        Set up the graph in the PostgreSQL database.

        Args:
            graph_name (str): The name of the graph to set up.
            drop_graph (bool): Whether to drop the existing graph if it exists.

        Returns:
            None
        """
        self.graphName = sql.Identifier(graph_name).as_string().strip('"')  # escaped
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
                                sql.SQL(f"SELECT drop_graph('{self.graphName}', true)")
                            )
                        await cur.execute(
                            sql.SQL(f"SELECT create_graph('{self.graphName}')")
                        )

    async def checkKeys(keys: list = [], elements: list = []):
        """
        Check if the keys of the CSV file match the elements.

        Args:
            keys (list): The keys of the CSV file.
            elements (list): The elements to check.
        """
        if not np.all(np.isin(elements, keys)):
            raise ValueError(
                f"CSV file must have {elements} columns, but {keys} columns were found."
            )

    async def createLabelType(self, label_type: str = "", value: str = "") -> None:
        """
        Create a label type in the PostgreSQL database.

        Args:
            label_type (str): The type of the label to create. It can be either "vertex" or "edge".
            value (str): The value of the label to create.
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                if label_type == "vertex":
                    await cur.execute(
                        sql.SQL(f"SELECT create_vlabel('{self.graphName}', '{value}');")
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
                        sql.SQL(f"SELECT create_elabel('{self.graphName}', '{value}');")
                    )
                    await cur.execute(
                        sql.SQL(
                            f'CREATE INDEX ON {self.graphName}."{value}" (start_id);'
                        )
                    )
                    await cur.execute(
                        sql.SQL(f'CREATE INDEX ON {self.graphName}."{value}" (end_id);')
                    )

    async def createVertices(
        self,
        vertices: pd.DataFrame = None,
        vertex_label: str = "",
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Create vertices in the PostgreSQL database.

        Args:
            vertices (pd.DataFrame): The vertices to create.
            vertex_label (str): The label of the vertices.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the vertices directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the vertices.
        """
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

    async def createEdges(
        self,
        edges: pd.DataFrame = None,
        edge_type: str = "",
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Create edges in the PostgreSQL database.

        Args:
            edges (pd.DataFrame): The edges to create.
            edge_type (str): The type of the edges.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the edges directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the edges.
        """
        # create edges
        if direct_loading:
            await self.createEdgesDirectly(self, edges, edge_type, chunk_size)
        else:
            if use_copy:
                await self.copyEdges(self, edges, edge_type, chunk_size)
            else:
                await self.createEdgesCypher(self, edges, edge_type, chunk_size)

    async def createVerticesCypher(
        self, vertices: pd.DataFrame = None, label: str = "", chunk_size: int = 0
    ) -> None:
        """
        Create vertices in the PostgreSQL database via Cypher.

        Args:
            vertices (pd.DataFrame): The vertices to create.
            label (str): The label of the vertices.
            chunk_size (int): The size of the chunks to create.
        """
        chunk_multiplier = 1
        args = []
        for i in range(0, len(vertices), chunk_size * chunk_multiplier):
            parts = []
            for idx, cols in vertices[i : i + chunk_size * chunk_multiplier].iterrows():
                properties = [f"{k}:'{v}'" for k, v in cols.items()]
                parts.append(f"(v{idx}:{label} {{{','.join(properties)}}})")
            args.append(
                f"SELECT * FROM cypher('{self.graphName}', $$ CREATE {','.join(parts)} $$) AS (a agtype);"
            )
        await self.executeWithTasks(self, self.executeQuery, args)

    async def createEdgesCypher(
        self, edges: pd.DataFrame = None, type: str = "", chunk_size: int = 0
    ) -> None:
        """
        Create edges in the PostgreSQL database via Cypher.

        Args:
            edges (pd.DataFrame): The edges to create.
            type (str): The type of the edges.
            chunk_size (int): The size of the chunks to create.
        """
        chunk_multiplier = 2
        args = []
        for i in range(0, len(edges), chunk_size * chunk_multiplier):
            parts = []
            for idx, cols in edges[i : i + chunk_size * chunk_multiplier].iterrows():
                parts.append(
                    f"MATCH (n:{cols['start_v_label']} {{id: '{cols['start_id']}'}}), (m:{cols['end_v_label']} {{id: '{cols['end_id']}'}}) CREATE (n)-[:{type}]->(m)"
                )
            query = "".join(
                [
                    f"SELECT * FROM cypher('{self.graphName}', $$ {part} $$) AS (a agtype);"
                    for part in parts
                ]
            )

            args.append(query)
        await self.executeWithTasks(self, self.executeQuery, args)

    async def createVerticesDirectly(
        self, vertices: pd.DataFrame = None, label: str = "", chunk_size: int = 0
    ) -> None:
        """
        Create vertices in the PostgreSQL database directly.

        Args:
            vertices (pd.DataFrame): The vertices to create.
            label (str): The label of the vertices.
            chunk_size (int): The size of the chunks to create.
        """
        chunk_multiplier = 1
        args = []
        for i in range(0, len(vertices), chunk_size * chunk_multiplier):
            values = []
            for idx, cols in vertices[i : i + chunk_size * chunk_multiplier].iterrows():
                properties = [f'"{k}":"{v}"' for k, v in cols.items()]
                values.append(f"('{{{','.join(properties)}}}')")
            args.append(
                "".join(
                    f"INSERT INTO {self.graphName}.\"{label}\" (properties) VALUES {','.join(values)};"
                )
            )
        await self.executeWithTasks(self, self.executeQuery, args)

    async def createEdgesDirectly(
        self, edges: pd.DataFrame = None, type: str = "", chunk_size: int = 0
    ) -> None:
        """
        Create edges in the PostgreSQL database directly.

        Args:
            edges (pd.DataFrame): The edges to create.
            type (str): The type of the edges.
            chunk_size (int): The size of the chunks to create.
        """
        chunk_multiplier = 2
        # create idmaps to convert entry_id to id(graphid)
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                idmaps = {}
                for e_label in [
                    edges.iloc[0]["start_v_label"],
                    edges.iloc[0]["end_v_label"],
                ]:
                    await cur.execute(
                        sql.SQL(
                            f'SELECT id, properties->\'"id"\' AS entry_id FROM {self.graphName}."{e_label}"'
                        )
                    )
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
            args.append(
                "".join(
                    f"INSERT INTO {self.graphName}.\"{type}\" (start_id, end_id) VALUES {','.join(values)};"
                )
            )
        await self.executeWithTasks(self, self.executeQuery, args)

    async def executeWithTasks(self, target: Callable = None, args: list = []) -> None:
        """
        Execute queries with tasks.

        Args:
            target (Callable): The target function to execute.
            args (list): The arguments to pass to the target function.
        """
        tasks = []
        for arg in args:
            task = asyncio.create_task(target(self.pool, arg))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def executeQuery(pool: AsyncConnectionPool = None, query: str = "") -> None:
        """
        Execute a query with an async connection pool.

        Args:
            pool (AsyncConnectionPool): The async connection pool to use.
            query (str): The query to execute.
        """
        while True:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(sql.SQL(query))
                        break
            except Exception as e:
                print(e)
                await asyncio.sleep(1)
                pass

    async def copyVertices(
        self,
        vertices: pd.DataFrame = None,
        label: str = "",
        chunk_size: int = 0,
        drop_graph: bool = False,
    ) -> None:
        """
        Create vertices in the PostgreSQL database via the COPY protocol.

        Args:
            vertices (pd.DataFrame): The vertices to create.
            label (str): The label of the vertices.
            chunk_size (int): The size of the chunks to create.
            drop_graph (bool): Whether to drop the existing graph if it exists.
        """
        chunk_multiplier = 1000
        first_id = await self.getFirstId(self, label)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                async with cur.copy(
                    f'COPY {self.graphName}."{label}" FROM STDIN (FORMAT TEXT)'
                ) as copy:
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
                await cur.execute(sql.SQL("COMMIT"))

    async def copyEdges(
        self,
        edges: pd.DataFrame = None,
        type: str = "",
        chunk_size: int = 0,
    ) -> None:
        """
        Create edges in the PostgreSQL database via the COPY protocol.

        Args:
            edges (pd.DataFrame): The edges to create.
            type (str): The type of the edges.
            chunk_size (int): The size of the chunks to create.
            drop_graph (bool): Whether to drop the existing graph if it exists.
        """
        chunk_multiplier = 1000

        # create idmaps to convert entry_id to id(graphid)
        idmaps = await self.getIdMaps(self, edges=edges)

        # create queries for edges
        first_id = await self.getFirstId(self, label_type=type)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                async with cur.copy(
                    f'COPY {self.graphName}."{type}" (id,start_id,end_id) FROM STDIN (FORMAT TEXT)'
                ) as copy:
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

    async def getIdMaps(self, edges: pd.DataFrame = None) -> dict:
        """
        Get the idmaps between entry_id and id(graphid).

        Args:
            edges (pd.DataFrame): The edges to create.
        """
        # create idmaps to convert entry_id to id(graphid)
        idmaps = {}
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                for e_label in [
                    edges.iloc[0]["start_v_label"],
                    edges.iloc[0]["end_v_label"],
                ]:
                    await cur.execute(
                        sql.SQL(
                            f'SELECT id, properties->\'"id"\' AS entry_id FROM {self.graphName}."{e_label}"'
                        )
                    )
                    rows = await cur.fetchall()
                    idmaps[e_label] = {
                        row.entry_id.replace('"', ""): row.id for row in rows
                    }
        return idmaps

    async def getFirstId(self, label_type: str = "") -> int:
        """
        Get the first id for a vertex or edge.

        Args:
            label_type (str): The type of the label to get the first id for.
        """
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                await cur.execute(
                    sql.SQL(f"SELECT id FROM ag_label WHERE name='{label_type}'")
                )
                row = await cur.fetchone()

                ENTRY_ID_BITS = 32 + 16
                ENTRY_ID_MASK = np.uint64(0x0000FFFFFFFFFFFF)
                first_id = ((np.uint64(row.id)) << ENTRY_ID_BITS) | (
                    (np.uint64(1)) & ENTRY_ID_MASK
                )

                return first_id

    async def renameColumns(df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Rename the columns of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to rename the columns of.
        """
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

    async def getVertexLabels(
        nodes: pd.DataFrame = None,
        v_labels: list = [],
        first_source_id: str = "",
        first_target_id: str = "",
    ) -> list:
        """
        Get the start and end vertex labels.

        Args:
            nodes (pd.DataFrame): The nodes to get the start and end vertex labels from.
            v_labels (list): The vertex labels.
            first_source_id (str): The first source id.
            first_target_id (str): The first target id.
        """
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

    @classmethod
    async def connect(
        cls, dsn: str = "", max_connections: int = 64, log_level=None, **kwargs
    ) -> "AgeFreighter":
        """
        Open a connection pool.

        Args:
            dsn (str): The data source name.
            max_connections (int): The maximum number of connections.
            log_level: The log level.
            **kwargs: Additional keyword arguments.
        """
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

    async def createGraphFromRelations(
        cls,
        graph_name: str = "",
        src: pd.DataFrame = None,
        existing_node_ids: list = [],
        first_chunk: bool = True,
        start_v_label: str = "",
        start_id: str = "",
        start_props: list = [],
        edge_type: str = "",
        end_v_label: str = "",
        end_id: str = "",
        end_props: list = [],
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Create a graph from DataFrame

        Args:
            graph_name (str): The name of the graph to load the data into.
            src (pd.DataFrame): The DataFrame to load the data from.
            existing_node_ids (list): The existing node IDs.
            first_chunk (bool): Whether it is the first chunk.
            start_v_label (str): The label of the start vertex.
            start_id (str): The ID of the start vertex.
            start_props (list): The properties of the start vertex.
            edge_type (str): The type of the edge.
            end_v_label (str): The label of the end vertex.
            end_id (str): The ID of the end vertex.
            end_props (list): The properties of the end vertex.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.
        """
        if first_chunk:
            await cls.checkKeys(
                src.keys(),
                [start_id] + start_props + [end_id] + end_props,
            )

            await cls.setUpGraph(cls, graph_name, drop_graph)

            await cls.createLabelType(cls, label_type="vertex", value=start_v_label)
            await cls.createLabelType(cls, label_type="vertex", value=end_v_label)
            await cls.createLabelType(cls, label_type="edge", value=edge_type)
            first_chunk = False

        for v_label, id, props in zip(
            [start_v_label, end_v_label],
            [start_id, end_id],
            [start_props, end_props],
        ):
            vertices = (
                src.loc[:, [id, *props]].drop_duplicates().rename(columns={id: "id"})
            )
            if not first_chunk:
                vertices = vertices[~vertices["id"].isin(existing_node_ids)]
            existing_node_ids.extend(vertices["id"].tolist())
            await cls.createVertices(
                cls,
                vertices,
                v_label,
                chunk_size,
                direct_loading,
                drop_graph,
                use_copy,
            )

        edges = (
            src.loc[:, [start_id, end_id]]
            .drop_duplicates()
            .rename(columns={start_id: "start_id", end_id: "end_id"})
        )
        edges["start_v_label"] = start_v_label
        edges["end_v_label"] = end_v_label
        await cls.createEdges(
            cls, edges, edge_type, chunk_size, direct_loading, drop_graph, use_copy
        )

    @classmethod
    async def loadFromSingleCSV(
        cls,
        graph_name: str = "",
        csv: str = "",
        start_v_label: str = "",
        start_id: str = "",
        start_props: list = [],
        edge_type: str = "",
        end_v_label: str = "",
        end_id: str = "",
        end_props: list = [],
        chunk_size: int = 3,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Load data from a single CSV file.

        Args:
            graph_name (str): The name of the graph to load the data into.
            csv (str): The path to the CSV file.
            start_v_label (str): The label of the start vertex.
            start_id (str): The ID of the start vertex.
            start_props (list): The properties of the start vertex.
            edge_type (str): The type of the edge.
            end_v_label (str): The label of the end vertex.
            end_id (str): The ID of the end vertex.
            end_props (list): The properties of the end vertex.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.
        """
        chunk_multiplier = 10000
        existing_node_ids = []
        first_chunk = True
        reader = pd.read_csv(csv, chunksize=chunk_size * chunk_multiplier)
        for df in reader:
            await cls.createGraphFromRelations(
                cls,
                graph_name=graph_name,
                src=df,
                existing_node_ids=existing_node_ids,
                first_chunk=first_chunk,
                start_v_label=start_v_label,
                start_id=start_id,
                start_props=start_props,
                edge_type=edge_type,
                end_v_label=end_v_label,
                end_id=end_id,
                end_props=end_props,
                chunk_size=chunk_size,
                direct_loading=direct_loading,
                drop_graph=drop_graph,
                use_copy=use_copy,
            )

    @classmethod
    async def loadFromCSVs(
        cls,
        graph_name: str = "",
        vertex_csvs: list = [],
        v_labels: list = [],
        edge_csvs: list = [],
        e_types: list = [],
        chunk_size: int = 128,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Load data from multiple CSV files.

        Args:
            graph_name (str): The name of the graph to load the data into.
            vertex_csvs (list): The paths to the vertex CSV files.
            v_labels (list): The labels of the vertices.
            edge_csvs (list): The paths to the edge CSV files.
            e_types (list): The types of the edges.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.
        """
        chunk_multiplier = 10000
        await cls.setUpGraph(cls, graph_name, drop_graph)
        for vertex_csv, v_label in zip(vertex_csvs, v_labels):
            first_chunk = True
            reader = pd.read_csv(vertex_csv, chunksize=chunk_size * chunk_multiplier)
            for vertices in reader:
                if first_chunk:
                    await cls.checkKeys(vertices.keys(), ["id"])

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

        for edge_csv, edge_type in zip(edge_csvs, e_types):
            first_chunk = True
            reader = pd.read_csv(edge_csv, chunksize=chunk_size * chunk_multiplier)
            for edges in reader:
                if first_chunk:
                    await cls.checkKeys(
                        edges.keys(),
                        [
                            "start_id",
                            "start_vertex_type",
                            "end_id",
                            "end_vertex_type",
                        ],
                    )

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

    @classmethod
    async def loadFromNetworkx(
        cls,
        graph_name: str = "",
        networkx_graph: DiGraph = None,
        chunk_size: int = 128,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Load data from a NetworkX graph.

        Args:
            graph_name (str): The name of the graph to load the data into.
            networkx_graph (DiGraph): The NetworkX graph.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.
        """
        await cls.setUpGraph(cls, graph_name, drop_graph)

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

    @classmethod
    async def loadFromNeo4j(
        cls,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4jpass",
        neo4j_database: str = "neo4j",
        graph_name: str = "",
        id_map: dict = {},
        chunk_size: int = 128,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Load data from a Neo4j graph.

        Args:
            uri (str): The URI of the Neo4j database.
            user (str): The user of the Neo4j database.
            password (str): The password of the Neo4j database.
            neo4j_database (str): The name of the Neo4j database.
            graph_name (str): The name of the graph to load the data into.
            id_map (dict): The ID map.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.
        """
        import neo4j

        await cls.setUpGraph(cls, graph_name, drop_graph)

        chunk_multiplier = 100
        async with neo4j.AsyncGraphDatabase.driver(
            uri, auth=(user, password)
        ) as driver:
            records, _, _ = await driver.execute_query(
                "MATCH (n) RETURN distinct labels(n)",
                db=neo4j_database,
            )
            v_labels = []
            for record in records:
                v_label = record["labels(n)"][0]
                v_labels.append(v_label)
                await cls.createLabelType(cls, label_type="vertex", value=v_label)
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

            records, _, _ = await driver.execute_query(
                "MATCH ()-[n]->() RETURN distinct type(n)",
                db=neo4j_database,
            )
            for record in records:
                edge_type = record["type(n)"]
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

    @classmethod
    async def loadFromPGSQL(
        cls,
        src_con_string: str = "",
        src_tables: list = [],
        graph_name: str = "",
        id_maps: dict = {},
        chunk_size: int = 128,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Load data from a PostgreSQL database.

        Args:
            src_con_string (str): The connection string of the source PostgreSQL database.
            src_tables (list): The source tables.
            graph_name (str): The name of the graph to load the data into.
            id_maps (dict): The ID maps.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.
        """
        import psycopg as pg

        chunk_multiplier = 1000

        try:
            with pg.connect(src_con_string) as conn:
                with conn.cursor(row_factory=namedtuple_row) as cur:
                    await cls.setUpGraph(cls, graph_name, drop_graph)
                    for src_table in src_tables.values():
                        if id_maps.get(src_table) is not None:  # nodes
                            id_map = id_maps[src_table]
                            await cls.createLabelType(
                                cls, label_type="vertex", value=src_table
                            )
                            cur.execute(sql.SQL(f"SELECT COUNT(*) FROM {src_table}"))
                            cnt = cur.fetchone()[0]
                            for i in range(0, cnt, chunk_size * chunk_multiplier):
                                cur.execute(
                                    sql.SQL(
                                        f"SELECT * FROM {src_table} LIMIT {chunk_size * chunk_multiplier} OFFSET {i}"
                                    )
                                )
                                rows = cur.fetchall()
                                vertices = pd.DataFrame(rows)
                                vertices.rename(columns={id_map: "id"}, inplace=True)
                                await cls.createVertices(
                                    cls,
                                    vertices,
                                    src_table,
                                    chunk_size,
                                    direct_loading,
                                    drop_graph,
                                    use_copy,
                                )
                        else:  # edges
                            await cls.createLabelType(
                                cls, label_type="edge", value=src_table
                            )
                            cur.execute(sql.SQL(f"SELECT COUNT(*) FROM {src_table}"))
                            cnt = cur.fetchone()[0]
                            for i in range(0, cnt, chunk_size * chunk_multiplier):
                                cur.execute(
                                    sql.SQL(
                                        f"SELECT * FROM {src_table} LIMIT {chunk_size * chunk_multiplier} OFFSET {i}"
                                    )
                                )
                                rows = cur.fetchall()
                                edges = pd.DataFrame(rows)
                                edges.insert(
                                    0, "start_v_label", list(id_maps.keys())[0]
                                )
                                edges.insert(0, "end_v_label", list(id_maps.keys())[1])
                                await cls.createEdges(
                                    cls,
                                    edges,
                                    src_table,
                                    chunk_size,
                                    direct_loading,
                                    drop_graph,
                                    use_copy,
                                )
        except Exception as e:
            raise e

    @classmethod
    async def loadFromParquet(
        cls,
        src_parquet: str = "",
        graph_name: str = "",
        start_v_label: str = "",
        start_id: str = "",
        start_props: list = [],
        edge_type: str = "",
        end_v_label: str = "",
        end_id: str = "",
        end_props: list = [],
        chunk_size: int = 128,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Load data from a Parquet file.

        Args:
            src_parquet (str): The path to the Parquet file.
            graph_name (str): The name of the graph to load the data into.
            start_v_label (str): The label of the start vertex.
            start_id (str): The ID of the start vertex.
            start_props (list): The properties of the start vertex.
            edge_type (str): The type of the edge.
            end_v_label (str): The label of the end vertex.
            end_id (str): The ID of the end vertex.
            end_props (list): The properties of the end vertex.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.
        """
        import pyarrow as pa
        from pyarrow.parquet import ParquetFile

        chunk_multiplier = 10000

        pf = ParquetFile(src_parquet)
        first_chunk = True
        existing_node_ids = []

        for batch in pf.iter_batches(chunk_size * chunk_multiplier):
            df = batch.to_pandas()
            await cls.createGraphFromRelations(
                cls,
                graph_name=graph_name,
                src=df,
                existing_node_ids=existing_node_ids,
                first_chunk=first_chunk,
                start_v_label=start_v_label,
                start_id=start_id,
                start_props=start_props,
                edge_type=edge_type,
                end_v_label=end_v_label,
                end_id=end_id,
                end_props=end_props,
                chunk_size=chunk_size,
                direct_loading=direct_loading,
                drop_graph=drop_graph,
                use_copy=use_copy,
            )
            first_chunk = False

    @classmethod
    async def loadFromAvro(
        cls,
        src_avro: str = "",
        graph_name: str = "",
        start_v_label: str = "",
        start_id: str = "",
        start_props: list = [],
        edge_type: str = "",
        end_v_label: str = "",
        end_id: str = "",
        end_props: list = [],
        chunk_size: int = 128,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Load data from an Avro file.

        Args:
            src_avro (str): The path to the Avro file.
            graph_name (str): The name of the graph to load the data into.
            start_v_label (str): The label of the start vertex.
            start_id (str): The ID of the start vertex.
            start_props (list): The properties of the start vertex.
            edge_type (str): The type of the edge.
            end_v_label (str): The label of the end vertex.
            end_id (str): The ID of the end vertex.
            end_props (list): The properties of the end vertex.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.
        """
        import fastavro as fa

        chunk_multiplier = 10000

        with open(src_avro, "rb") as f:
            reader = fa.reader(f)
            first_chunk = True
            existing_node_ids = []

            for records in reader:
                df = pd.DataFrame.from_records(records, index=[0])
                await cls.createGraphFromRelations(
                    cls,
                    graph_name=graph_name,
                    src=df,
                    existing_node_ids=existing_node_ids,
                    first_chunk=first_chunk,
                    start_v_label=start_v_label,
                    start_id=start_id,
                    start_props=start_props,
                    edge_type=edge_type,
                    end_v_label=end_v_label,
                    end_id=end_id,
                    end_props=end_props,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    drop_graph=drop_graph,
                    use_copy=use_copy,
                )
                first_chunk = False
