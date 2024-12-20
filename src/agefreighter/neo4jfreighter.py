from agefreighter import AgeFreighter
from neo4j import GraphDatabase

import logging

log = logging.getLogger(__name__)


class Neo4jFreighter(AgeFreighter):
    def __init__(self):
        super().__init__()

    async def __aenter__(self):
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await super().__aexit__(exc_type, exc, tb)
        if exc_type:
            print(f"Exception: {exc_type}, {exc}")

    async def load(
        self,
        neo4j_uri: str = "",
        neo4j_user: str = "",
        neo4j_password: str = "",
        neo4j_database: str = "",
        id_map: dict = {},
        graph_name: str = "",
        chunk_size: int = 128,
        direct_loading: bool = False,
        create_graph: bool = True,
        use_copy: bool = True,
        **kwargs,
    ) -> None:
        """
        Load data from a Neo4j graph.

        Args:
            neo4j_uri (str): The URI of the Neo4j database.
            neo4j_user (str): The username of the Neo4j database.
            neo4j_password (str): The password of the Neo4j database.
            neo4j_database (str): The database of the Neo4j database.
            id_map (dict): The ID map.
            graph_name (str): The name of the graph to load the data into.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            create_graph (bool): Whether to create the graph.
            use_copy (bool): Whether to use the COPY protocol to load the data.

        Returns:
            None
        """
        log.debug("Loading data from a Neo4j graph")
        import pandas as pd
        import nest_asyncio

        nest_asyncio.apply()

        CHUNK_MULTIPLIER = 1000
        CHUNK_SIZE = chunk_size * CHUNK_MULTIPLIER

        existing_node_ids = []
        first_chunk = True
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        async for chunk in self.fetch_edges_in_chunks(
            driver=driver, chunk_size=CHUNK_SIZE
        ):
            first_record = chunk[0]
            start_v_label = list(first_record["m"].labels)[0]
            start_id = id_map[start_v_label]
            start_props = [
                prop for prop in first_record["m"].keys() if prop != start_id
            ]
            end_v_label = list(first_record["n"].labels)[0]
            end_id = id_map[end_v_label]
            end_props = [prop for prop in first_record["n"].keys() if prop != end_id]
            edge_type = first_record["r"].type
            edge_props = [
                prop
                for prop in first_record["r"].keys()
                if prop != "from" and prop != "to"
            ]

            df = pd.DataFrame(
                [{**record["m"], **record["n"], **record["r"]} for record in chunk]
            )
            await self.createGraphFromDataFrame(
                graph_name=graph_name,
                src=df,
                existing_node_ids=existing_node_ids,
                first_chunk=first_chunk,
                start_v_label=start_v_label,
                start_id=start_id,
                start_props=start_props,
                edge_type=edge_type,
                edge_props=edge_props,
                end_v_label=end_v_label,
                end_id=end_id,
                end_props=end_props,
                chunk_size=chunk_size,
                direct_loading=direct_loading,
                create_graph=create_graph,
                use_copy=use_copy,
            )
            first_chunk = False

        await self.close()

    async def fetch_edges_in_chunks(
        self, driver: GraphDatabase.driver = None, chunk_size: int = 0
    ) -> list:
        """
        Fetch edges in chunks.

        Args:
            driver: The driver.
            chunk_size: The size of the chunks to fetch the edges in.

        Returns:
            None

        Notes:
            context manager closes after with block.
        """
        with driver.session() as session:
            skip = 0
            while True:
                result = session.run(
                    "MATCH (m)-[r]->(n) RETURN m,r,n SKIP $skip LIMIT $limit",
                    skip=skip,
                    limit=chunk_size,
                )
                records = list(result)
                if not records:
                    break
                yield records
                skip += chunk_size
