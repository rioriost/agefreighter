from agefreighter import AgeFreighter

import logging

log = logging.getLogger(__name__)


class CSVFreighter(AgeFreighter):
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
        csv: str = "",
        start_v_label: str = "",
        start_id: str = "",
        start_props: list = [],
        edge_type: str = "",
        end_v_label: str = "",
        end_id: str = "",
        end_props: list = [],
        graph_name: str = "",
        chunk_size: int = 128,
        direct_loading: bool = False,
        drop_graph: bool = False,
        use_copy: bool = True,
        **kwargs,
    ) -> None:
        """
        Load data from a single CSV file.

        Args:
            csv (str): The path to the CSV file.
            start_v_label (str): The label of the start vertex.
            start_id (str): The ID of the start vertex.
            start_props (list): The properties of the start vertex.
            edge_type (str): The type of the edge.
            end_v_label (str): The label of the end vertex.
            end_id (str): The ID of the end vertex.
            end_props (list): The properties of the end vertex.

        Common Args:
            graph_name (str): The name of the graph to load the data into.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            drop_graph (bool): Whether to drop the existing graph if it exists.
            use_copy (bool): Whether to use the COPY protocol to load the data.

        Returns:
            None
        """
        log.debug("Loading data from a single CSV file")
        import pandas as pd

        chunk_multiplier = 10000

        existing_node_ids = []
        first_chunk = True
        reader = pd.read_csv(csv, chunksize=chunk_size * chunk_multiplier)
        for df in reader:
            await self.createGraphFromDataFrame(
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
        await self.close()
