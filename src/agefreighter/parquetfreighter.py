from agefreighter import AgeFreighter

import warnings
import logging

log = logging.getLogger(__name__)


class ParquetFreighter(AgeFreighter):
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
        parquet_path: str = "",
        start_v_label: str = "",
        start_id: str = "",
        start_props: list = [],
        edge_type: str = "",
        edge_props: list = [],
        end_v_label: str = "",
        end_id: str = "",
        end_props: list = [],
        graph_name: str = "",
        chunk_size: int = 128,
        direct_loading: bool = False,
        create_graph: bool = False,
        use_copy: bool = True,
        **kwargs,
    ) -> None:
        """
        Load data from a Parquet file.

        Args:
            parquet_path (str): The path to the Parquet file.
            start_v_label (str): The label of the start vertex.
            start_id (str): The ID of the start vertex.
            start_props (list): The properties of the start vertex.
            edge_type (str): The type of the edge.
            edge_props (list): The properties of the edge.
            end_v_label (str): The label of the end vertex.
            end_id (str): The ID of the end vertex.
            end_props (list): The properties of the end vertex.
            graph_name (str): The name of the graph to load the data into.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            create_graph (bool): Whether to create the graph.
            use_copy (bool): Whether to use the COPY protocol to load the data.

        Returns:
            None
        """
        log.debug("Loading data from a Parquet file")
        from pyarrow.parquet import ParquetFile

        if "source_parquet" in kwargs.keys():
            warnings.warn(
                "The 'source_parquet' parameter is deprecated. Please use 'parquet_path' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            parquet_path = kwargs["source_parquet"]

        CHUNK_MULTIPLIER = 10000

        pf = ParquetFile(parquet_path)
        first_chunk = True
        existing_node_ids = []

        for batch in pf.iter_batches(chunk_size * CHUNK_MULTIPLIER):
            df = batch.to_pandas()
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
