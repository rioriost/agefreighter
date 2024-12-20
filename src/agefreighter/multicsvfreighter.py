from agefreighter import AgeFreighter

import logging

log = logging.getLogger(__name__)


class MultiCSVFreighter(AgeFreighter):
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
        vertex_csvs: list = [],
        vertex_labels: list = [],
        edge_csvs: list = [],
        edge_types: list = [],
        graph_name: str = "",
        chunk_size: int = 128,
        direct_loading: bool = False,
        create_graph: bool = True,
        use_copy: bool = True,
        **kwargs,
    ) -> None:
        """
        Load data from multiple CSV files.

        Args:
            vertex_csvs (list): The paths to the vertex CSV files.
            vertex_labels (list): The labels of the vertices.
            edge_csvs (list): The paths to the edge CSV files.
            edge_types (list): The types of the edges.
            graph_name (str): The name of the graph to load the data into.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            create_graph (bool): Whether to create the graph.
            use_copy (bool): Whether to use the COPY protocol to load the data.

        Returns:
            None
        """
        log.debug("Loading data from multiple CSV files")
        import pandas as pd

        CHUNK_MULTIPLIER = 10000

        await self.setUpGraph(graph_name=graph_name, create_graph=create_graph)
        for vertex_csv, vertex_label in zip(vertex_csvs, vertex_labels):
            first_chunk = True
            reader = pd.read_csv(vertex_csv, chunksize=chunk_size * CHUNK_MULTIPLIER)
            for vertices in reader:
                if first_chunk:
                    self.checkKeys(
                        vertices.keys(),
                        ["id"],
                        "CSV file must have {elements} columns, but {keys} columns were found.",
                    )

                    await self.createLabelType(label_type="vertex", value=vertex_label)
                    first_chunk = False

                await self.createVertices(
                    vertices=vertices,
                    vertex_label=vertex_label,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                )

        for edge_csv, edge_type in zip(edge_csvs, edge_types):
            first_chunk = True
            reader = pd.read_csv(edge_csv, chunksize=chunk_size * CHUNK_MULTIPLIER)
            for edges in reader:
                if first_chunk:
                    self.checkKeys(
                        edges.keys(),
                        [
                            "start_id",
                            "start_vertex_type",
                            "end_id",
                            "end_vertex_type",
                        ],
                        "CSV file must have {elements} columns, but {keys} columns were found.",
                    )

                    await self.createLabelType(label_type="edge", value=edge_type)
                    edge_props = [
                        col
                        for col in edges.columns
                        if col
                        not in [
                            "start_id",
                            "end_id",
                            "start_vertex_type",
                            "end_vertex_type",
                        ]
                    ]
                    first_chunk = False
                edges.rename(
                    columns={
                        "start_vertex_type": "start_v_label",
                        "end_vertex_type": "end_v_label",
                    },
                    inplace=True,
                )
                await self.createEdges(
                    edges=edges,
                    edge_type=edge_type,
                    edge_props=edge_props,
                    chunk_size=chunk_size,
                    direct_loading=direct_loading,
                    use_copy=use_copy,
                )
        await self.close()
