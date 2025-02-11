from agefreighter import AgeFreighter

import logging
import pandas as pd
import networkx as nx
from typing import Generator

log = logging.getLogger(__name__)


class NetworkXFreighter(AgeFreighter):
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
        networkx_graph: nx.Graph,
        id_map: dict = {},
        graph_name: str = "",
        chunk_size: int = 128,
        direct_loading: bool = False,
        create_graph: bool = True,
        use_copy: bool = True,
        **kwargs,
    ) -> None:
        """
        Load data from a NetworkX graph.

        Args:
            networkx_graph (nx.Graph): The NetworkX graph.
            id_map (dict): The ID map.
            graph_name (str): The name of the graph to load the data into.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            create_graph (bool): Whether to create the graph.
            use_copy (bool): Whether to use the COPY protocol to load the data.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        log.debug("Loading data from a NetworkX graph")

        if "progress" in kwargs.keys():
            self.progress = kwargs["progress"]

        CHUNK_MULTIPLIER = 10000
        first_chunk = True
        existing_node_ids: list = []

        edges = nx.to_pandas_edgelist(networkx_graph)
        edge_props = [
            prop for prop in edges.columns if prop not in ["source", "target", "label"]
        ]

        start_v_label = networkx_graph.nodes[edges.iloc[0]["source"]]["label"]
        start_id = id_map[start_v_label]
        end_v_label = networkx_graph.nodes[edges.iloc[0]["target"]]["label"]
        end_id = id_map[end_v_label]
        edge_type = edges.iloc[0]["label"]

        start_props = [
            prop
            for prop in networkx_graph.nodes[edges.iloc[0]["source"]]
            if prop not in ["label", "name"]
        ]
        end_props = [
            prop
            for prop in networkx_graph.nodes[edges.iloc[0]["target"]]
            if prop not in ["label", "name"]
        ]

        for chunk in self.getChunks(edges, chunk_size * CHUNK_MULTIPLIER):
            source_ids = chunk["source"].tolist()
            target_ids = chunk["target"].tolist()

            start_attributes = self.get_node_attributes(
                networkx_graph, source_ids, ["name"] + start_props
            )
            end_attributes = self.get_node_attributes(
                networkx_graph, target_ids, ["name"] + end_props
            )

            chunk[start_v_label] = start_attributes["name"]
            for start_prop in start_props:
                chunk[start_prop] = start_attributes[start_prop]

            chunk[end_v_label] = end_attributes["name"]
            for end_prop in end_props:
                chunk[end_prop] = end_attributes[end_prop]

            chunk.rename(columns={"source": start_id, "target": end_id}, inplace=True)
            chunk.drop(columns=["label"], inplace=True)
            await self.createGraphFromDataFrame(
                graph_name=graph_name,
                src=chunk,
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

    @staticmethod
    def getChunks(df: pd.DataFrame, chunk_size: int = 0) -> Generator:
        """
        Get the DataFrame in chunks.

        Args:
            df (DataFrame): The DataFrame to get the edges from.
            chunk_size (int): The size of the chunks to get the edges in.

        Returns:
            Generator: The DataFrame in chunks.
        """
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i : i + chunk_size].copy()

    @staticmethod
    def get_node_attributes(graph, node_ids, attributes) -> dict:
        """
        Get node attributes.

        Args:
            graph: The graph.
            node_ids: The node IDs.
            attributes: The attributes.

        Returns:
            dict: The node attributes.
        """
        return {
            attr: [graph.nodes[node_id].get(attr, None) for node_id in node_ids]
            for attr in attributes
        }
