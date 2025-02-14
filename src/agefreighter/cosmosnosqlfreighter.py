from agefreighter import AgeFreighter
import json
import pandas as pd
import os
import logging
import tempfile
import asyncio
from typing import Any, Dict, List

log = logging.getLogger(__name__)


class CosmosNoSQLFreighter(AgeFreighter):
    """
    A freighter class for loading data from Cosmos NoSQL databases.
    """

    def __init__(self) -> None:
        super().__init__()

    async def __aenter__(self) -> "CosmosNoSQLFreighter":
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await super().__aexit__(exc_type, exc, tb)
        if exc_type:
            print(f"Exception: {exc_type}, {exc}")

    async def load(
        self,
        cosmos_endpoint: str = "",
        cosmos_key: str = "",
        cosmos_database: str = "",
        cosmos_container: str = "",
        id_map: Dict[Any, Any] = {},
        graph_name: str = "",
        chunk_size: int = 128,
        direct_loading: bool = False,
        create_graph: bool = False,
        use_copy: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Load data from a Cosmos NoSQL database.

        Args:
            cosmos_endpoint (str): The Cosmos endpoint.
            cosmos_key (str): The Cosmos key.
            cosmos_database (str): The Cosmos database.
            cosmos_container (str): The Cosmos container.
            id_map (dict): The ID mapping dictionary.
            graph_name (str): The graph name to load the data into.
            chunk_size (int): Size of the chunks for processing.
            direct_loading (bool): Whether to load the data directly.
            create_graph (bool): Whether to create a new graph.
            use_copy (bool): Whether to use COPY protocol for loading.
            **kwargs: Additional keyword arguments, e.g. progress monitor.

        Returns:
            None
        """
        log.debug("Loading data from a Cosmos NoSQL database")

        # If a progress callback is passed, attach it to self.progress.
        if "progress" in kwargs:
            self.progress = kwargs["progress"]

        # Define a multiplier to increase the chunk sizes for bulk reads.
        CHUNK_MULTIPLIER = 100

        # Fetch documents from Cosmos DB and store each group in temporary JSON files.
        temp_files: Dict[str, Dict[str, str]] = self.fetch_documents(
            cosmos_endpoint=cosmos_endpoint,
            cosmos_key=cosmos_key,
            cosmos_database=cosmos_database,
            cosmos_container=cosmos_container,
            page_size=chunk_size * CHUNK_MULTIPLIER,
        )

        # Set up the graph for the provided name.
        await self.setUpGraph(graph_name=graph_name, create_graph=create_graph)

        tasks_for_v: List[asyncio.Task] = []
        # Process vertex files concurrently.
        vertex_files: Dict[str, str] = temp_files.get("vertex", {})
        for label, filepath in vertex_files.items():
            self._validate_file(filepath)
            vertices: pd.DataFrame = self._read_json_file(filepath)
            self._process_vertex_columns(vertices)
            # Create a task for processing all chunks for this vertex label.
            tasks_for_v.append(
                asyncio.create_task(
                    self._process_vertices_chunks(
                        vertices=vertices,
                        label=label,
                        chunk_size=chunk_size,
                        chunk_multiplier=CHUNK_MULTIPLIER,
                        direct_loading=direct_loading,
                        use_copy=use_copy,
                    )
                )
            )

        # Await concurrently the processing of all files.
        if tasks_for_v:
            await asyncio.gather(*tasks_for_v)

        tasks_for_e: List[asyncio.Task] = []
        # Process edge files concurrently.
        edge_files: Dict[str, str] = temp_files.get("edge", {})
        for edge_type, filepath in edge_files.items():
            self._validate_file(filepath)
            edges: pd.DataFrame = self._read_json_file(filepath)
            edges = self._process_edges_file(edges)
            tasks_for_e.append(
                asyncio.create_task(
                    self._process_edges_chunks(
                        edges=edges,
                        edge_type=edge_type,
                        chunk_size=chunk_size,
                        chunk_multiplier=CHUNK_MULTIPLIER,
                        direct_loading=direct_loading,
                        use_copy=use_copy,
                    )
                )
            )

        # Await concurrently the processing of all files.
        if tasks_for_e:
            await asyncio.gather(*tasks_for_e)

    @staticmethod
    def _validate_file(filepath: str) -> None:
        """
        Validate that the given file path exists.

        Args:
            filepath (str): The path to the file.

        Raises:
            ValueError: If the file does not exist.
        """
        if not (isinstance(filepath, str) and os.path.exists(filepath)):
            raise ValueError("Invalid file path: " + str(filepath))

    @staticmethod
    def _read_json_file(filepath: str) -> pd.DataFrame:
        """
        Read a JSON file and return its content as a pandas DataFrame.

        Args:
            filepath (str): The file path to read.

        Returns:
            pd.DataFrame: The contents of the JSON file.
        """
        return pd.read_json(filepath)

    @staticmethod
    def fetch_documents(
        cosmos_endpoint: str = "",
        cosmos_key: str = "",
        cosmos_database: str = "",
        cosmos_container: str = "",
        page_size: int = 12800,
    ) -> Dict[str, Dict[str, str]]:
        """
        Fetch documents from a Cosmos NoSQL database and store each group in temporary files.

        Args:
            cosmos_endpoint (str): The Cosmos endpoint.
            cosmos_key (str): The Cosmos key.
            cosmos_database (str): The Cosmos database.
            cosmos_container (str): The Cosmos container.
            page_size (int): The maximum number of items per page read.

        Returns:
            dict: A dictionary with keys "vertex" and "edge". Each key maps labels (or types)
                  to file paths containing the corresponding documents.
        """
        from azure.cosmos import CosmosClient, exceptions

        grouped_docs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        try:
            client = CosmosClient(cosmos_endpoint, credential=cosmos_key)
            database = client.get_database_client(cosmos_database)
            container = database.get_container_client(cosmos_container)
            items_iterator = container.read_all_items(max_item_count=page_size)

            for page in items_iterator.by_page():
                for document in page:
                    label = document.get("label")
                    if not label:
                        raise ValueError("Document does not have a label.")
                    # Determine document type based on "_isEdge" flag.
                    doc_type = "edge" if document.get("_isEdge", False) else "vertex"
                    grouped_docs.setdefault(doc_type, {}).setdefault(label, []).append(
                        document
                    )

            tempfiles: Dict[str, Dict[str, str]] = {}
            # Write each document group to a temporary JSON file.
            for doc_type, label_docs in grouped_docs.items():
                tempfiles[doc_type] = {}
                for label, docs in label_docs.items():
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", suffix=".json", delete=False
                    ) as temp_file:
                        json.dump(docs, temp_file, ensure_ascii=False, indent=2)
                        tempfiles[doc_type][label] = temp_file.name

            # Log temporary file info.
            for doc_type, label_dict in tempfiles.items():
                for label, filepath in label_dict.items():
                    log.debug(f"Type: {doc_type}, Label: {label} -> File: {filepath}")

            return tempfiles

        except exceptions.CosmosHttpResponseError as e:
            print(f"Error accessing Cosmos DB: {e}")
        except Exception as ex:
            print(f"Unexpected error: {ex}")

        return {}

    def _process_vertex_columns(self, vertices: pd.DataFrame) -> None:
        """
        Process the DataFrame columns for vertex data:
        - Drop unwanted metadata columns.
        - Transform each column (except 'label' and 'id') using a mapping logic.

        Args:
            vertices (pd.DataFrame): The DataFrame with vertex data.
        """
        # Drop unnecessary metadata columns.
        cols_to_drop = ["pk", "_rid", "_self", "_etag", "_attachments", "_ts"]
        vertices.drop(columns=cols_to_drop, errors="ignore", inplace=True)

        # Transform columns that are not label or id.
        for col in vertices.columns:
            if col not in ["label", "id"]:
                vertices[col] = vertices[col].apply(
                    lambda x: x[0]["_value"].strip('"')
                    if isinstance(x, list) and x and "_value" in x[0]
                    else x
                )

    async def _process_vertices_chunks(
        self,
        vertices: pd.DataFrame,
        label: str,
        chunk_size: int,
        chunk_multiplier: int,
        direct_loading: bool,
        use_copy: bool,
    ) -> None:
        """
        Process vertex data in chunks concurrently and load them into the graph.

        Args:
            vertices (pd.DataFrame): DataFrame containing vertex data.
            label (str): The vertex label.
            chunk_size (int): The base chunk size.
            chunk_multiplier (int): A multiplier to determine the amount of data per chunk.
            direct_loading (bool): Whether to load data directly.
            use_copy (bool): Whether to use the COPY protocol for loading.
        """
        total_rows: int = len(vertices)
        # Create the vertex label type once.
        await self.createLabelType(label_type="vertex", value=label)

        tasks: List[asyncio.Task] = []
        for start in range(0, total_rows, chunk_size * chunk_multiplier):
            chunk_df = vertices.iloc[start : start + chunk_size * chunk_multiplier]
            tasks.append(
                asyncio.create_task(
                    self.createVertices(
                        vertices=chunk_df,
                        vertex_label=label,
                        chunk_size=chunk_size,
                        direct_loading=direct_loading,
                        use_copy=use_copy,
                    )
                )
            )
        if tasks:
            await asyncio.gather(*tasks)

    def _process_edges_file(self, edges: pd.DataFrame) -> pd.DataFrame:
        """
        Process an edge DataFrame by dropping unwanted columns and renaming others.

        Args:
            edges (pd.DataFrame): DataFrame of edge data.

        Returns:
            pd.DataFrame: Processed DataFrame with cleaned and renamed columns.
        """
        cols_to_drop = [
            "_sinkPartition",
            "_isEdge",
            "pk",
            "_rid",
            "_self",
            "_etag",
            "_attachments",
            "_ts",
        ]
        edges.drop(columns=cols_to_drop, errors="ignore", inplace=True)
        edges.rename(
            columns={
                "_sink": "end_id",
                "_sinkLabel": "end_v_label",
                "_vertexId": "start_id",
                "_vertexLabel": "start_v_label",
            },
            inplace=True,
        )
        return edges

    async def _process_edges_chunks(
        self,
        edges: pd.DataFrame,
        edge_type: str,
        chunk_size: int,
        chunk_multiplier: int,
        direct_loading: bool,
        use_copy: bool,
    ) -> None:
        """
        Process edge data in chunks concurrently and load them into the graph.

        Args:
            edges (pd.DataFrame): DataFrame containing edge data.
            edge_type (str): The type of the edge.
            chunk_size (int): The base chunk size.
            chunk_multiplier (int): A multiplier to determine the amount of data per chunk.
            direct_loading (bool): Whether to load data directly.
            use_copy (bool): Whether to use the COPY protocol for loading.
        """
        total_rows: int = len(edges)
        # Determine edge properties by excluding known identifier columns.
        edge_props: List[str] = [
            col
            for col in edges.columns
            if col not in ["start_id", "end_id", "start_v_label", "end_v_label"]
        ]
        # Create the edge label type once.
        await self.createLabelType(label_type="edge", value=edge_type)

        tasks: List[asyncio.Task] = []
        for start in range(0, total_rows, chunk_size * chunk_multiplier):
            chunk_df = edges.iloc[start : start + chunk_size * chunk_multiplier]
            tasks.append(
                asyncio.create_task(
                    self.createEdges(
                        edges=chunk_df,
                        edge_type=edge_type,
                        edge_props=edge_props,
                        chunk_size=chunk_size,
                        direct_loading=direct_loading,
                        use_copy=use_copy,
                    )
                )
            )
        if tasks:
            await asyncio.gather(*tasks)
