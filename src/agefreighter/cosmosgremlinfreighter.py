from agefreighter import AgeFreighter

import logging

log = logging.getLogger(__name__)


class CosmosGremlinFreighter(AgeFreighter):
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
        cosmos_gremlin_endpoint: str = "",
        cosmos_gremlin_key: str = "",
        cosmos_username: str = "",
        id_map: dict = {},
        graph_name: str = "",
        chunk_size: int = 128,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = True,
        **kwargs,
    ) -> None:
        """
        Load data from a Gremlin graph.

        Args:
            cosmos_gremlin_endpoint (str): The Cosmos Gremlin endpoint.
            cosmos_gremlin_key (str): The Cosmos Gremlin key.
            cosmos_username (str): The Cosmos username.
            id_map (dict): The ID map.

        Common Args:
            graph_name (str): The name of the graph to load the data into.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.

        Returns:
            None
        """
        log.debug("Loading data from a Gremlin graph")
        from gremlin_python.driver import client, serializer
        import pandas as pd
        import concurrent.futures
        import asyncio
        import nest_asyncio

        nest_asyncio.apply()

        chunk_multiplier = 1000

        try:
            g = client.Client(
                url=cosmos_gremlin_endpoint,
                traversal_source="g",
                username=cosmos_username,
                password=cosmos_gremlin_key,
                message_serializer=serializer.GraphSONSerializersV2d0(),
            )
        except Exception as e:
            print(f"Failed to connect to Gremlin server: {e}")
            return

        await self.setUpGraph(graph_name, drop_graph)
        for vertex_label in id_map.keys():
            query = f"g.V().hasLabel('{vertex_label}').count()"
            cnt = g.submit_async(query).result().all().result()[0]
            first_chunk = True
            for i in range(0, cnt, chunk_size * chunk_multiplier):
                if first_chunk:
                    await self.createLabelType(label_type="vertex", value=vertex_label)
                    first_chunk = False
                query = f"g.V().hasLabel('{vertex_label}').range({i}, {i + chunk_size * chunk_multiplier})"
                records = g.submit_async(query).result().all().result()
                dicts = [
                    {
                        k: v[0]["value"]
                        for k, v in record["properties"].items()
                        if k != "pk"
                    }
                    for record in records
                ]
                vertices = pd.DataFrame.from_dict(dicts)
                vertices.rename(columns={id_map[vertex_label]: "id"}, inplace=True)
                await self.createVertices(
                    vertices=vertices,
                    vertex_label=vertex_label,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    drop_graph=drop_graph,
                    use_copy=use_copy,
                )

        chunk_multiplier = 10

        start_v_label, end_v_label = list(id_map.keys())
        query_cnt = f"g.V().hasLabel('{start_v_label}').count()"
        cnt = g.submit_async(query_cnt).result().all().result()[0]
        query_edge_types = f"g.V().hasLabel('{start_v_label}').limit(1).outE().limit(1)"
        edge_types = [
            rec["label"]
            for rec in g.submit_async(query_edge_types).result().all().result()
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            for edge_type in edge_types:
                first_chunk = True
                futures = []
                for i in range(0, cnt, chunk_size * chunk_multiplier):
                    if first_chunk:
                        await self.createLabelType(label_type="edge", value=edge_type)
                        first_chunk = False
                    query = f"g.V().hasLabel('{start_v_label}').as('start').range({i}, {i + chunk_size * chunk_multiplier}).outE().hasLabel('{edge_type}').inV().as('end').select('start','end')"
                    futures.append(
                        loop.run_in_executor(
                            executor,
                            self.processRecords,
                            query,
                            start_v_label,
                            end_v_label,
                            id_map,
                            g,
                        )
                    )
                results = await asyncio.gather(*futures)
                for edges in results:
                    await self.createEdges(
                        edges,
                        edge_type,
                        chunk_size,
                        direct_loading,
                        drop_graph,
                        use_copy,
                    )

        await self.close()

    @staticmethod
    def processRecords(query, start_v_label, end_v_label, id_map, g):
        import pandas as pd

        records = g.submit_async(query).result().all().result()
        edges = pd.DataFrame(
            [
                {
                    "start_id": record["start"]["properties"][id_map[start_v_label]][0][
                        "value"
                    ],
                    "end_id": record["end"]["properties"][id_map[end_v_label]][0][
                        "value"
                    ],
                }
                for record in records
            ]
        )
        edges["start_v_label"] = start_v_label
        edges["end_v_label"] = end_v_label
        return edges
