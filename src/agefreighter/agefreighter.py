import asyncio
import sys
import logging
import pandas as pd
from psycopg_pool import AsyncConnectionPool
from typing_extensions import Callable

log = logging.getLogger(__name__)


class Factory:
    """
    Factory class to create instances of AgeFreighter.
    """

    @staticmethod
    def create_instance(type: str = ""):
        if type == "AzureStorageFreighter":
            import agefreighter.azurestoragefreighter as azurestoragefreighter

            return azurestoragefreighter.AzureStorageFreighter()
        elif type == "MultiAzureStorageFreighter":
            import agefreighter.multiazurestoragefreighter as multiazurestoragefreighter

            return multiazurestoragefreighter.MultiAzureStorageFreighter()
        elif type == "AvroFreighter":
            import agefreighter.avrofreighter as avrofreighter

            return avrofreighter.AvroFreighter()
        elif type == "CosmosGremlinFreighter":
            import agefreighter.cosmosgremlinfreighter as cosmosgremlinfreighter

            return cosmosgremlinfreighter.CosmosGremlinFreighter()
        elif type == "CSVFreighter":
            import agefreighter.csvfreighter as csvfreighter

            return csvfreighter.CSVFreighter()
        elif type == "MultiCSVFreighter":
            import agefreighter.multicsvfreighter as multicsvfreighter

            return multicsvfreighter.MultiCSVFreighter()
        elif type == "Neo4jFreighter":
            import agefreighter.neo4jfreighter as neo4jfreighter

            return neo4jfreighter.Neo4jFreighter()
        elif type == "NetworkXFreighter":
            import agefreighter.networkxfreighter as networkxfreighter

            return networkxfreighter.NetworkXFreighter()
        elif type == "ParquetFreighter":
            import agefreighter.parquetfreighter as parquetfreighter

            return parquetfreighter.ParquetFreighter()
        elif type == "PGFreighter":
            import agefreighter.pgfreighter as pgfreighter

            return pgfreighter.PGFreighter()
        else:
            raise ValueError(f"Unknown type: {type}")


class AgeFreighter:
    """
    AgeFreighter is a Python package that helps you to create a graph database using Azure Database for PostgreSQL.
    """

    name = "AgeFreighter"
    version = "0.7.3"
    author = "Rio Fujita"

    def __init__(self):
        """
        Initialize the AgeFreighter object.
        """
        log.debug(f"Creating AgeFreighter, in {sys._getframe().f_code.co_name}.")
        self.pool: AsyncConnectionPool = None
        self.dsn: str = ""
        self.graph_name: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type:
            print(f"Exception: {exc_type}, {exc}")
        await self.pool.close()

    @classmethod
    def get_version(cls) -> str:
        """
        Get the version of the AgeFreighter package.

        Returns:
            str: The version of the AgeFreighter package.
        """
        log.debug(f"Getting version, in {sys._getframe().f_code.co_name}.")
        return cls.version

    async def connect(
        self,
        dsn: str = "",
        max_connections: int = 64,
        min_connections: int = 4,
        log_level=None,
        **kwargs,
    ) -> "AgeFreighter":
        """
        Open a connection pool.

        Args:
            dsn (str): The data source name.
            max_connections (int): The maximum number of connections.
            min_connections (int): The minimum number of connections.
            log_level: The log level.
            **kwargs: Additional keyword arguments.

        Returns:
            AgeFreighter: The AgeFreighter object.
        """
        log.debug("Opening connection pool.")
        log.debug(f"Opening connection pool, in {sys._getframe().f_code.co_name}.")

        import platform

        # to make large number of connections
        if platform.system() in ["Darwin", "Linux"]:
            import resource

            current_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (8192, current_limit[1]))

        self.dsn_wo_options = dsn
        self.dsn = dsn + " options='-c search_path=ag_catalog,\"$user\",public'"
        self.pool = AsyncConnectionPool(
            self.dsn,
            max_size=max_connections,
            min_size=min_connections,
            open=False,
            timeout=600,
            **kwargs,
        )
        try:
            await self.pool.open()
            await self.pool.wait()
        except Exception as e:
            log.error(f"Error: {e}, in {sys._getframe().f_code.co_name}.")
            raise e

    async def close(self) -> None:
        """
        Close the connection pool.
        """
        log.debug("Closing connection pool.")
        try:
            await self.pool.close()
        except Exception as e:
            log.error(f"Error: {e}, in {sys._getframe().f_code.co_name}.")

    async def load(self) -> None:
        """
        Dummy method for loading data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def load_multi(self) -> None:
        """
        Dummy method for loading data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def checkKeys(keys: list = [], elements: list = [], error_msg: str = "") -> None:
        """
        Check if the keys of the CSV file match the elements.

        Args:
            keys (list): The keys of the CSV file.
            elements (list): The elements to check.
        """
        log.debug(
            f"Checking keys {keys} against elements {elements}, in {sys._getframe().f_code.co_name}."
        )
        import numpy as np

        if not np.all(np.isin(elements, keys)):
            raise ValueError(error_msg.format(elements=elements, keys=keys))

    @staticmethod
    def quotedGraphName(graph_name: str = "") -> str:
        """
        Quote the graph name.

        Args:
            graph_name (str): The name of the graph.
        """
        log.debug(
            f"Quoting graph name {graph_name}, in {sys._getframe().f_code.co_name}."
        )
        if graph_name.lower() != graph_name:
            return f'"{graph_name}"'
        return graph_name

    async def setUpGraph(
        self,
        graph_name: str = "",
        create_graph: bool = False,
    ) -> None:
        """
        Set up the graph in the PostgreSQL database.

        Args:
            graph_name (str): The name of the graph to set up.
            create_graph (bool): Whether to create the graph.

        Returns:
            None
        """
        log.debug("Setting up graph '%s'.", graph_name)
        from psycopg.sql import SQL, Identifier
        from psycopg.rows import namedtuple_row

        log.debug(
            f"Setting up graph '{graph_name}', in {sys._getframe().f_code.co_name}."
        )
        # for more precise graph name,
        # see https://github.com/apache/age/blob/master/src/include/utils/name_validation.h
        self.graph_name = Identifier(graph_name).as_string().strip('"')
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                await cur.execute(SQL("CREATE EXTENSION IF NOT EXISTS age CASCADE"))
                await cur.execute(
                    SQL(f"SELECT count(*) FROM ag_graph WHERE name='{self.graph_name}'")
                )
                row = await cur.fetchone()
                exists = row.count == 1
                if exists:
                    log.debug(
                        f"Graph '{self.graph_name}' already exists, in {sys._getframe().f_code.co_name}."
                    )
                    if create_graph:
                        log.debug(
                            f"Dropping graph '{self.graph_name}', in {sys._getframe().f_code.co_name}."
                        )
                        await cur.execute(
                            SQL(f"SELECT drop_graph('{self.graph_name}', true)")
                        )
                        log.debug(
                            f"Creating graph '{self.graph_name}', in {sys._getframe().f_code.co_name}."
                        )
                        await cur.execute(
                            SQL(f"SELECT create_graph('{self.graph_name}')")
                        )
                    else:
                        log.debug(
                            f"Try to add data to '{self.graph_name}', in {sys._getframe().f_code.co_name}."
                        )
                        pass
                else:
                    log.debug(
                        f"Graph '{self.graph_name}' doesn't exist, in {sys._getframe().f_code.co_name}."
                    )
                    if create_graph:
                        log.debug(
                            f"Creating graph '{self.graph_name}', in {sys._getframe().f_code.co_name}."
                        )
                        await cur.execute(
                            SQL(f"SELECT create_graph('{self.graph_name}')")
                        )
                    else:
                        raise ValueError(
                            f"Graph '{self.graph_name}' doesn't exist. Please set create_graph to True."
                        )

    async def createLabelType(self, label_type: str = "", value: str = "") -> None:
        """
        Create a label type in the PostgreSQL database.

        Args:
            label_type (str): The type of the label to create. It can be either "vertex" or "edge".
            value (str): The value of the label to create.

        Returns:
            None
        """
        log.debug("Creating a label or a type '%s', value '%s'", label_type, value)

        from psycopg.sql import SQL, Identifier, Literal

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                if label_type == "vertex":
                    try:
                        log.debug(
                            f"Creating a vlabel '{value}', in {sys._getframe().f_code.co_name}."
                        )
                        await cur.execute(
                            SQL("SELECT create_vlabel({schema}, {label});").format(
                                schema=self.graph_name,
                                label=Literal(value),
                            )
                        )
                        log.debug(
                            f"Creating indices for vertices, in {sys._getframe().f_code.co_name}."
                        )
                        await cur.execute(
                            SQL(
                                "CREATE INDEX ON {schema}.{label} USING GIN (properties);"
                            ).format(
                                schema=Identifier(self.graph_name),
                                label=Identifier(value),
                            )
                        )
                        await cur.execute(
                            SQL(
                                "CREATE INDEX ON {schema}.{label} USING BTREE (id);"
                            ).format(
                                schema=Identifier(self.graph_name),
                                label=Identifier(value),
                            )
                        )
                    except Exception as e:
                        pass
                elif label_type == "edge":
                    try:
                        log.debug(
                            f"Creating an elabel '{value}', in {sys._getframe().f_code.co_name}."
                        )
                        await cur.execute(
                            SQL("SELECT create_elabel({schema}, {label});").format(
                                schema=self.graph_name,
                                label=Literal(value),
                            )
                        )
                        log.debug(
                            f"Creating indices for edges, in {sys._getframe().f_code.co_name}."
                        )
                        await cur.execute(
                            SQL("CREATE INDEX ON {schema}.{label} (start_id);").format(
                                schema=Identifier(self.graph_name),
                                label=Identifier(value),
                            )
                        )
                        await cur.execute(
                            SQL("CREATE INDEX ON {schema}.{label} (end_id);").format(
                                schema=Identifier(self.graph_name),
                                label=Identifier(value),
                            )
                        )
                    except Exception as e:
                        pass

    async def createVertices(
        self,
        vertices: pd.DataFrame = None,
        vertex_label: str = "",
        chunk_size: int = 3,
        direct_loading: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Create vertices in the PostgreSQL database.

        Args:
            vertices (pd.DataFrame): The vertices to create.
            vertex_label (str): The label of the vertices.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the vertices directly.
            use_copy (bool): Whether to use the COPY protocol to load the vertices.

        Returns:
            None
        """
        log.debug(
            f"Creating vertices {len(vertices)}, in {sys._getframe().f_code.co_name}."
        )
        if direct_loading:
            try:
                vertices = vertices.map(
                    lambda x: x.replace("'", "''") if isinstance(x, str) else x
                )
                await self.createVerticesDirectly(vertices, vertex_label, chunk_size)
            except AttributeError:
                print(
                    "You maybe specify a column twice in the arguements. e.g. 'v_label' and 'props'"
                )
        else:
            try:
                vertices = vertices.map(
                    lambda x: x.replace("'", r"\'") if isinstance(x, str) else x
                )
                if use_copy:
                    await self.copyVertices(
                        vertices=vertices,
                        vertex_label=vertex_label,
                        chunk_size=chunk_size,
                    )
                else:
                    await self.createVerticesCypher(vertices, vertex_label, chunk_size)
            except AttributeError:
                print(
                    "You maybe specify a column twice in the arguements. e.g. 'v_label' and 'props'"
                )

    async def createEdges(
        self,
        edges: pd.DataFrame = None,
        edge_type: str = "",
        edge_props: list = [],
        chunk_size: int = 3,
        direct_loading: bool = False,
        use_copy: bool = False,
    ) -> None:
        """
        Create edges in the PostgreSQL database.

        Args:
            edges (pd.DataFrame): The edges to create.
            edge_type (str): The type of the edges.
            edge_props (list): The properties of the edges.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the edges directly.
            use_copy (bool): Whether to use the COPY protocol to load the edges.

        Returns:
            None
        """
        log.debug(f"Creating edges {len(edges)}, in {sys._getframe().f_code.co_name}.")
        # create edges
        if direct_loading:
            await self.createEdgesDirectly(
                edges=edges,
                edge_type=edge_type,
                edge_props=edge_props,
                chunk_size=chunk_size,
            )
        else:
            if use_copy:
                await self.copyEdges(
                    edges=edges,
                    edge_type=edge_type,
                    edge_props=edge_props,
                    chunk_size=chunk_size,
                )

            else:
                await self.createEdgesCypher(
                    edges=edges,
                    edge_type=edge_type,
                    edge_props=edge_props,
                    chunk_size=chunk_size,
                )

    async def createVerticesCypher(
        self, vertices: pd.DataFrame = None, label: str = "", chunk_size: int = 0
    ) -> None:
        """
        Create vertices in the PostgreSQL database via Cypher.

        Args:
            vertices (pd.DataFrame): The vertices to create.
            label (str): The label of the vertices.
            chunk_size (int): The size of the chunks to create.

        Returns:
            None

        Note:
            psycopg can not handle Cypher's UNWIND as of 9th Dec 2024.
        """
        log.debug(
            f"Creating vertices {len(vertices)} via Cypher, in {sys._getframe().f_code.co_name}."
        )
        CHUNK_MULTIPLIER = 2
        args = []
        for i in range(0, len(vertices), chunk_size * CHUNK_MULTIPLIER):
            chunk = vertices.iloc[i : i + chunk_size * CHUNK_MULTIPLIER]
            parts = chunk.apply(
                lambda row: "(v{0}:{1} {{{2}}})".format(
                    row.name,
                    label,
                    ",".join([f"{k}:'{v}'" for k, v in row.items()]),
                ),
                axis=1,
            ).tolist()
            args.append(
                f"SELECT * FROM cypher('{self.graph_name}', $$ CREATE {','.join(parts)} $$) AS (a agtype);"
            )
        await self.executeWithTasks(self.executeQuery, args)

    async def createEdgesCypher(
        self,
        edges: pd.DataFrame = None,
        edge_type: str = "",
        edge_props: list = [],
        chunk_size: int = 0,
    ) -> None:
        """
        Create edges in the PostgreSQL database via Cypher.

        Args:
            edges (pd.DataFrame): The edges to create.
            edge_type (str): The type of the edges.
            edge_props (list): The properties of the edges.
            chunk_size (int): The size of the chunks to create.

        Returns:
            None
        """
        log.debug(
            f"Creating edges {len(edges)} via Cypher, in {sys._getframe().f_code.co_name}."
        )

        CHUNK_MULTIPLIER = 2
        args = []
        for i in range(0, len(edges), chunk_size * CHUNK_MULTIPLIER):
            chunk = edges.iloc[i : i + chunk_size * CHUNK_MULTIPLIER]
            parts = chunk.apply(
                lambda row: self.createEdgeCypher(
                    row=row, edge_type=edge_type, edge_props=edge_props
                ),
                axis=1,
            ).tolist()
            query = "".join(
                [
                    f"SELECT * FROM cypher('{self.graph_name}', $$ {part} $$) AS (a agtype);"
                    for part in parts
                ]
            )
            args.append(query)
        await self.executeWithTasks(self.executeQuery, args)

    def createEdgeCypher(
        self,
        row: pd.core.series.Series = None,
        edge_type: str = "",
        edge_props: list = [],
    ) -> str:
        """
        Create a Cypher query to create an edge.

        Args:
            row (pd.core.series.Series): The row to create the edge from.
            edge_type (str): The type of the edge.
            edge_props (list): The properties of the edge.

        Returns:
            str: The Cypher query to create the edge.
        """
        props = [f"{edge_prop}:'{row[edge_prop]}'" for edge_prop in edge_props]
        if len(props) > 0:
            return """MATCH (n:{0} {{id: '{1}'}}),
                        (m:{2} {{id: '{3}'}})
                        CREATE (n)-[:{4} {{{5}}}]->(m)""".format(
                row["start_v_label"],
                row["start_id"],
                row["end_v_label"],
                row["end_id"],
                edge_type,
                ",".join(props),
            )
        else:
            return """MATCH (n:{0} {{id: '{1}'}}),
                        (m:{2} {{id: '{3}'}})
                        CREATE (n)-[:{4}]->(m)""".format(
                row["start_v_label"],
                row["start_id"],
                row["end_v_label"],
                row["end_id"],
                edge_type,
            )

    async def createVerticesDirectly(
        self, vertices: pd.DataFrame = None, label: str = "", chunk_size: int = 0
    ) -> None:
        """
        Create vertices in the PostgreSQL database directly.

        Args:
            vertices (pd.DataFrame): The vertices to create.
            label (str): The label of the vertices.
            chunk_size (int): The size of the chunks to create.

        Returns:
            None
        """
        log.debug(
            f"Creating vertices {len(vertices)} with SQL, in {sys._getframe().f_code.co_name}."
        )
        CHUNK_MULTIPLIER = 2
        args = []
        graph_name = self.quotedGraphName(self.graph_name)
        for i in range(0, len(vertices), chunk_size * CHUNK_MULTIPLIER):
            chunk = vertices.iloc[i : i + chunk_size * CHUNK_MULTIPLIER]
            values = chunk.apply(
                lambda row: "('{"
                + ",".join([f'"{k}":"{v}"' for k, v in row.items()])
                + "}')",
                axis=1,
            ).tolist()
            args.append(
                f'INSERT INTO {graph_name}."{label}" (properties) VALUES {",".join(values)};'
            )
        await self.executeWithTasks(self.executeQuery, args)

    async def createEdgesDirectly(
        self,
        edges: pd.DataFrame = None,
        edge_type: str = "",
        edge_props: list = [],
        chunk_size: int = 0,
    ) -> None:
        """
        Create edges in the PostgreSQL database directly.

        Args:
            edges (pd.DataFrame): The edges to create.
            edge_type (str): The type of the edges.
            edge_props (list): The properties of the edges.
            chunk_size (int): The size of the chunks to create.

        Returns:
            None
        """
        log.debug(
            f"Creating edges {len(edges)} with SQL, in {sys._getframe().f_code.co_name}."
        )
        CHUNK_MULTIPLIER = 2
        # create id_maps to convert entry_id to id(graphid)
        id_maps = await self.getIdMaps(edges=edges)

        # create queries for edges
        graph_name = self.quotedGraphName(self.graph_name)
        args = []
        if edge_props:
            query = f'INSERT INTO {graph_name}."{edge_type}" (start_id, end_id, properties) VALUES {{values}};'
        else:
            query = f'INSERT INTO {graph_name}."{edge_type}" (start_id, end_id) VALUES {{values}};'

        for i in range(0, len(edges), chunk_size * CHUNK_MULTIPLIER):
            chunk = edges.iloc[i : i + chunk_size * CHUNK_MULTIPLIER]
            parts = self.processChunkDirectly(chunk, query, edge_props, id_maps)
            args.append("".join(parts))
        await self.executeWithTasks(self.executeQuery, args)

    def processChunkDirectly(
        self,
        chunk: pd.DataFrame,
        query: str = "",
        edge_props: list = [],
        id_maps: list = [],
    ) -> list:
        """
        Process a chunk of data directly.

        Args:
            chunk (pd.DataFrame): The chunk of data to process.
            query (str): The query to execute.
            edge_props (list): The properties of the edges.
            id_maps (list): The ID maps.

        Returns:
            list: The processed chunk of data.
        """
        return chunk.apply(
            lambda row: query.format(
                values=self.createValuesDirectly(row, edge_props, id_maps)
            ),
            axis=1,
        ).tolist()

    def createValuesDirectly(
        self,
        row: pd.core.series.Series = None,
        edge_props: list = [],
        id_maps: list = [],
    ) -> str:
        """
        Create values directly.

        Args:
            row (pd.core.series.Series): The row to create the values from.
            edge_props (list): The properties of the edges.
            id_maps (list): The ID maps.

        Returns:
            str: The values.
        """
        start_id = id_maps[str(row["start_v_label"])][str(row["start_id"])]
        end_id = id_maps[str(row["end_v_label"])][str(row["end_id"])]
        if edge_props:
            props = ",".join(
                [f'"{edge_prop}":"{row[edge_prop]}"' for edge_prop in edge_props]
            )
            return (
                f"('{start_id}'::graphid, '{end_id}'::graphid, '{{{props}}}'::agtype)"
            )
        else:
            return f"('{start_id}'::graphid, '{end_id}'::graphid)"

    async def executeWithTasks(self, target: Callable = None, args: list = []) -> None:
        """
        Execute queries with tasks.

        Args:
            target (Callable): The target function to execute.
            args (list): The arguments to pass to the target function.
        """
        import concurrent.futures

        loop = asyncio.get_running_loop()
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                tasks = [
                    loop.run_in_executor(
                        executor, lambda arg=arg: asyncio.run(target(arg))
                    )
                    for arg in args
                ]
                await asyncio.gather(*tasks)
        except Exception as e:
            log.debug(f"Error: {e}, in {sys._getframe().f_code.co_name}.")

    async def executeQuery(self, query: str = "") -> None:
        """
        Execute a query with an async connection pool.

        Args:
            pool (AsyncConnectionPool): The async connection pool to use.
            query (str): The query to execute.

        Returns:
            None
        """
        try:
            log.debug(f"Executing query: {query}, in {sys._getframe().f_code.co_name}.")
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
        except Exception as e:
            log.debug(f"Error: {e}, in {sys._getframe().f_code.co_name}.")

    async def copyChunk(
        self,
        chunk: pd.DataFrame,
        first_id: int = 0,
        graph_name: str = "",
        label: str = "",
        id_maps: dict = [],
        is_edge: bool = False,
    ):
        """
        Copy a chunk of data to the PostgreSQL database.

        Args:
            chunk (pd.DataFrame): The chunk of data to copy.
            first_id (int): The first ID.
            graph_name (str): The name of the graph.
            label (str): The label of the data.
            id_maps (dict): The ID maps.
            is_edge (bool): Whether the data is an edge.
        """
        from psycopg.sql import SQL

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                if is_edge:
                    query = f'COPY {graph_name}."{label}" (id,start_id,end_id) FROM STDIN (FORMAT TEXT)'
                else:
                    query = f'COPY {graph_name}."{label}" FROM STDIN (FORMAT TEXT)'

                async with cur.copy(query) as copy:
                    args_list = []
                    if is_edge:
                        chunk_args = chunk.apply(
                            lambda row: "{0}\t{1}\t{2}\n".format(
                                first_id + row.name,
                                id_maps[str(row["start_v_label"])][
                                    str(row["start_id"])
                                ],
                                id_maps[str(row["end_v_label"])][str(row["end_id"])],
                            ),
                            axis=1,
                        ).tolist()
                    else:
                        chunk_args = chunk.apply(
                            lambda row: "{0}\t{{{1}}}\n".format(
                                first_id + row.name,
                                ", ".join([f'"{k}": "{v}"' for k, v in row.items()]),
                            ),
                            axis=1,
                        ).tolist()

                    args_list.extend(chunk_args)
                    args = "".join(args_list)
                    await copy.write(args)
                await cur.execute(SQL("COMMIT"))

    async def copyVertices(
        self,
        vertices: pd.DataFrame = None,
        vertex_label: str = "",
        chunk_size: int = 0,
    ) -> None:
        """
        Create vertices in the PostgreSQL database via the COPY protocol.

        Args:
            vertices (pd.DataFrame): The vertices to create.
            label (str): The label of the vertices.
            chunk_size (int): The size of the chunks to create.

        Returns:
            None
        """
        log.debug("Copying vertices via COPY protocol")
        from psycopg.sql import SQL

        log.debug(
            f"Copying vertices {len(vertices)}, in {sys._getframe().f_code.co_name}."
        )
        CHUNK_MULTIPLIER = 1000
        first_id = await self.getFirstId(
            graph_name=self.graph_name, label_type=vertex_label
        )
        graph_name = self.quotedGraphName(self.graph_name)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = f'COPY {graph_name}."{vertex_label}" FROM STDIN (FORMAT TEXT)'
                async with cur.copy(query) as copy:
                    for i in range(0, len(vertices), chunk_size * CHUNK_MULTIPLIER):
                        args_list = []
                        chunk = vertices.iloc[i : i + chunk_size * CHUNK_MULTIPLIER]
                        chunk_args = chunk.apply(
                            lambda row: "{0}\t{{{1}}}\n".format(
                                first_id + row.name,
                                ", ".join([f'"{k}": "{v}"' for k, v in row.items()]),
                            ),
                            axis=1,
                        ).tolist()
                        args_list.extend(chunk_args)
                        args = "".join(args_list)
                        await copy.write(args)
                await cur.execute(SQL("COMMIT"))

    async def copyEdges(
        self,
        edges: pd.DataFrame = None,
        edge_type: str = "",
        edge_props: list = [],
        chunk_size: int = 0,
    ) -> None:
        """
        Create edges in the PostgreSQL database via the COPY protocol.

        Args:
            edges (pd.DataFrame): The edges to create.
            edge_type (str): The type of the edges.
            edge_props (list): The properties of the edges.
            chunk_size (int): The size of the chunks to create.

        Returns:
            None
        """
        log.debug(f"Copying edges {len(edges)}, in {sys._getframe().f_code.co_name}.")
        CHUNK_MULTIPLIER = 1000

        # create id_maps to convert entry_id to id(graphid)
        id_maps = await self.getIdMaps(edges=edges)

        # create queries for edges
        first_id = await self.getFirstId(
            graph_name=self.graph_name, label_type=edge_type
        )
        graph_name = self.quotedGraphName(self.graph_name)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                if edge_props:
                    query = f'COPY {graph_name}."{edge_type}" (id,start_id,end_id, properties) FROM STDIN (FORMAT TEXT)'
                else:
                    query = f'COPY {graph_name}."{edge_type}" (id,start_id,end_id) FROM STDIN (FORMAT TEXT)'
                async with cur.copy(query) as copy:
                    for i in range(0, len(edges), chunk_size * CHUNK_MULTIPLIER):
                        chunk = edges.iloc[i : i + chunk_size * CHUNK_MULTIPLIER]
                        if edge_props:
                            chunk_args = chunk.apply(
                                lambda row: "{0}\t{1}\t{2}\t{{{3}}}\n".format(
                                    first_id + row.name,
                                    id_maps[str(row["start_v_label"])][
                                        str(row["start_id"])
                                    ],
                                    id_maps[str(row["end_v_label"])][
                                        str(row["end_id"])
                                    ],
                                    ",".join(
                                        [
                                            f'"{edge_prop}":"{row[edge_prop]}"'
                                            for edge_prop in edge_props
                                        ]
                                    ),
                                ),
                                axis=1,
                            ).tolist()
                        else:
                            chunk_args = chunk.apply(
                                lambda row: "{0}\t{1}\t{2}\n".format(
                                    first_id + row.name,
                                    id_maps[str(row["start_v_label"])][
                                        str(row["start_id"])
                                    ],
                                    id_maps[str(row["end_v_label"])][
                                        str(row["end_id"])
                                    ],
                                ),
                                axis=1,
                            ).tolist()
                        args = "".join(chunk_args)
                        await copy.write(args)

    async def getIdMaps(self, edges: pd.DataFrame = None) -> dict:
        """
        Get the idmaps between entry_id and id(graphid).

        Args:
            edges (pd.DataFrame): The edges to create.

        Returns:
            dict: The ID maps.
        """
        from psycopg.rows import namedtuple_row
        from psycopg.sql import SQL

        # create id_maps to convert entry_id to id(graphid)
        id_maps = {}
        graph_name = self.quotedGraphName(self.graph_name)
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                for e_label in [
                    edges.iloc[0]["start_v_label"],
                    edges.iloc[0]["end_v_label"],
                ]:
                    await cur.execute(
                        SQL(
                            f'SELECT id, properties->\'"id"\' AS entry_id FROM {graph_name}."{e_label}"'
                        )
                    )
                    rows = await cur.fetchall()
                    id_maps[e_label] = {
                        row.entry_id.replace('"', ""): row.id for row in rows
                    }

        return id_maps

    async def getFirstId(self, graph_name: str = "", label_type: str = "") -> int:
        """
        Get the first id for a vertex or edge.

        Args:
            label_type (str): The type of the label to get the first id for.

        Returns:
            int: The first id.
        """
        import numpy as np
        from psycopg.rows import namedtuple_row

        graph_name = self.quotedGraphName(graph_name)
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                relation = f'{graph_name}."{label_type}"'
                await cur.execute(
                    f"SELECT id FROM ag_label WHERE relation='{relation}'::regclass;"
                )
                row = await cur.fetchone()

                ENTRY_ID_BITS = 32 + 16
                ENTRY_ID_MASK = np.uint64(0x0000FFFFFFFFFFFF)
                first_id = ((np.uint64(row.id)) << ENTRY_ID_BITS) | (
                    (np.uint64(1)) & ENTRY_ID_MASK
                )

                return first_id

    async def createGraphFromDataFrame(
        self,
        graph_name: str = "",
        src: pd.DataFrame = None,
        existing_node_ids: list = [],
        first_chunk: bool = True,
        start_v_label: str = "",
        start_id: str = "",
        start_props: list = [],
        edge_type: str = "",
        edge_props: list = [],
        end_v_label: str = "",
        end_id: str = "",
        end_props: list = [],
        chunk_size: int = 3,
        direct_loading: bool = False,
        create_graph: bool = False,
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
            edge_props (list): The properties of the edge.
            end_v_label (str): The label of the end vertex.
            end_id (str): The ID of the end vertex.
            end_props (list): The properties of the end vertex.

        Returns:
            None
        """
        log.debug(
            f"Creating a graph from DataFrame, in {sys._getframe().f_code.co_name}."
        )
        if first_chunk:
            self.checkKeys(
                src.keys(),
                [start_id] + start_props + [end_id] + end_props,
                "CSV file must have {elements} columns, but {keys} columns were found.",
            )

            await self.setUpGraph(graph_name=graph_name, create_graph=create_graph)

            await self.createLabelType(label_type="vertex", value=start_v_label)
            await self.createLabelType(label_type="vertex", value=end_v_label)
            await self.createLabelType(label_type="edge", value=edge_type)

        for vertex_label, id, props in zip(
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
            await self.createVertices(
                vertices=vertices,
                vertex_label=vertex_label,
                chunk_size=chunk_size,
                direct_loading=direct_loading,
                use_copy=use_copy,
            )

        edges = (
            src.loc[:, [start_id, end_id, *edge_props]]
            .drop_duplicates()
            .rename(columns={start_id: "start_id", end_id: "end_id"})
        )
        edges["start_v_label"] = start_v_label
        edges["end_v_label"] = end_v_label
        await self.createEdges(
            edges=edges,
            edge_type=edge_type,
            edge_props=edge_props,
            chunk_size=chunk_size,
            direct_loading=direct_loading,
            use_copy=use_copy,
        )
