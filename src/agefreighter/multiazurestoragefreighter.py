from agefreighter import AgeFreighter
from psycopg_pool import AsyncConnectionPool
import sys
import logging

log = logging.getLogger(__name__)


class MultiAzureStorageFreighter(AgeFreighter):
    def __init__(self):
        super().__init__()
        self.subscription_id: str = ""
        self.location: str = ""
        self.storage_account_name: str = ""
        self.blob_container_name: str = ""
        self.access_key: str = ""
        self.pg_fqdn: str = ""
        self.pg_server_name: str = ""

    async def __aenter__(self):
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await super().__aexit__(exc_type, exc, tb)
        if exc_type:
            print(f"Exception: {exc_type}, {exc}")

    async def load(
        self,
        vertex_args: list = [],
        edge_args: list = [],
        graph_name: str = "",
        chunk_size: int = 128,
        create_graph: bool = True,
        **kwargs,
    ):
        """
        Load a graph data to the PostgreSQL Flex with Azure Storage.

        Args:
            vertex_args (list): Vertex Arguments
            edge_args (list): Edge Arguments
            graph_name (str): The name of the graph to load the data into.
            chunk_size (int): The size of the chunks to create.
            create_graph (bool): Whether to create the graph.

        Keyword Args:
            subscription_id (str): Azure Subscription ID
        """
        log.debug("Loading data from Azure Storage")

        CHUNK_MULTIPLIER = 10000

        # optional
        if "subscription_id" in kwargs.keys():
            if self.isValidAzureSubscriptionID(kwargs["subscription_id"]):
                self.subscription_id = kwargs["subscription_id"]
            else:
                log.error("Invalid Azure Subscription ID.")
                return

        if self.subscription_id == "":
            print("Finding Subscription ID...")
            if not self.findAzureSubscriptionID():
                log.error("Azure Subscription ID is not set.")
                return

        if not await self.setParameters():
            log.error("Parameters are not set.")
            return

        # Enable Azure Storage extension
        print("Enabling extension...")
        ae = AzureExtensions(
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group_name,
            pg_server_name=self.pg_server_name,
            extensions=["azure_storage"],
            pool=self.pool,
        )
        ae.enable()
        await ae.create()

        # Create a Storage Account and a Blob Container, and attach it to the PostgreSQL Flex
        print("Creating storage account...")
        sa = StorageAccount(
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group_name,
            location=self.location,
            pool=self.pool,
        )
        sa.create()
        self.storage_account_name = sa.storage_account_name
        self.blob_container_name = sa.blob_container_name
        self.access_key = sa.access_key
        await sa.attach()

        file_paths_dict = {
            "nodes": [
                path
                for vertex_arg in vertex_args
                for k, path in vertex_arg.items()
                if k == "csv_path"
            ],
            "edges": [
                path
                for edge_arg in edge_args
                for k, v in edge_arg.items()
                if k == "csv_paths"
                for path in v
            ],
        }

        # Upload a CSV file to the blob container
        print("Uploading files...")
        bl = BlobUploader(
            storage_account_name=self.storage_account_name,
            blob_container_name=self.blob_container_name,
            access_key=self.access_key,
            file_paths_dict=file_paths_dict,
            lines_per_chunk=chunk_size * CHUNK_MULTIPLIER,
        )
        await bl.upload()

        # Check if the columns contain arguments
        self.checkKeys(
            columns_in_csvs=bl.columns_in_csvs,
            vertex_args=vertex_args,
            edge_args=edge_args,
        )

        # Create temporary tables
        # They're named 'temp', but not the persistent tables.
        print("Creating temporary tables...")
        tt = TempTables(
            columns_in_csvs=bl.columns_in_csvs,
            file_paths_dict=file_paths_dict,
            pool=self.pool,
        )
        await tt.create()

        # Load the data from the Azure Storage to the temporary table
        print("Loading files to temporary tables...")
        sl = StorageLoader(
            tmp_file_lists=bl.tmp_file_lists,
            total_lines=bl.total_lines,
            storage_account_name=self.storage_account_name,
            blob_container_name=self.blob_container_name,
            tbls_from_storage=tt.tbls_from_storage,
            columns_in_tbls_from_storage=tt.columns_in_tbls_from_storage,
            columns_in_csvs=bl.columns_in_csvs,
            pool=self.pool,
        )
        await sl.load()

        # start to create a graph
        print("Creating a graph...")
        await self.setUpGraph(graph_name=graph_name, create_graph=create_graph)
        for vertex_arg in vertex_args:
            for k, label in vertex_arg.items():
                if k == "label":
                    await self.createLabelType(label_type="vertex", value=label)
        for edge_arg in edge_args:
            for k, type in edge_arg.items():
                if k == "type":
                    await self.createLabelType(label_type="edge", value=type)

        gl = GraphLoader(
            tbls_from_storage=tt.tbls_from_storage,
            total_lines=bl.total_lines,
            vertex_args=vertex_args,
            edge_args=edge_args,
            columns_in_csvs=bl.columns_in_csvs,
            id_map_tbls=tt.id_map_tbls,
            graph_name=graph_name,
            records_per_thread=chunk_size * CHUNK_MULTIPLIER,
            pool=self.pool,
        )
        await gl.load()

        await tt.delete()  # it should be implemented in __del__
        await self.close()
        print("Creating a graph: Done!")

    def findAzureSubscriptionID(self) -> bool:
        """
        Get the Azure Subscription ID from the Azure CLI.

        Returns:
            bool: True if the Azure Subscription ID is set, False otherwise
        """
        log.debug("Fetching Azure Subscription ID")
        import subprocess
        import json

        try:
            result = subprocess.run(
                ["az", "account", "show"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return False
            account_info = json.loads(result.stdout)
            subscription_id = account_info.get("id")
            self.subscription_id = subscription_id
            return True

        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    @staticmethod
    def isValidAzureSubscriptionID(subscriptionID: str = "") -> bool:
        """
        Check if the Azure Subscription ID is valid.

        Args:
            subscriptionID (str): Azure Subscription ID

        Returns:
            bool: True if the Azure Subscription ID is valid, False otherwise
        """
        import re

        if not subscriptionID:
            return False
        pattern = re.compile(r"^[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$")
        return True if pattern.match(subscriptionID) else False

    async def setParameters(self) -> bool:
        """
        Set the parameters for the Azure Storage Freighter.

        Returns:
            bool: True if the parameters are set, False otherwise
        """
        log.debug("Setting parameters")
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.postgresqlflexibleservers import PostgreSQLManagementClient
        import re

        client = PostgreSQLManagementClient(
            credential=DefaultAzureCredential(), subscription_id=self.subscription_id
        )
        pattern = re.compile(r"host=(?P<host>[^ ]+)")
        self.pg_fqdn = pattern.match(self.dsn).group("host")
        self.pg_server_name = self.pg_fqdn.split(".")[0]
        servers = client.servers.list()
        pattern = re.compile(
            r"/subscriptions/[^/]+/resourceGroups/(?P<resourceGroup>[^/]+)/providers"
        )
        for server in servers:
            if server.fully_qualified_domain_name == self.pg_fqdn:
                if resource_group_name := pattern.match(server.id).group(
                    "resourceGroup"
                ):
                    self.resource_group_name = resource_group_name
                    self.location = server.location
                    return True
        return False

    def checkKeys(
        self, columns_in_csvs: list = [], vertex_args: list = [], edge_args: list = []
    ) -> None:
        """
        Check if the columns contain arguments.

        Args:
            columns_in_csv (list): Columns in the CSV file
            vertex_args (list): Vertex Arguments
            edge_args (list): Edge Arguments

        Raises:
            ValueError: If the column is not in the CSV

        Returns:
            None
        """
        log.debug("Checking keys")
        for vertex_arg in vertex_args:
            file_path = vertex_arg["csv_path"]
            if set(columns_in_csvs[file_path]) != set(
                [vertex_arg["id"]] + vertex_arg["props"]
            ):
                log.error(
                    f"Columns in CSV file '{file_path}' didn't match to vertex_args."
                )
                raise ValueError
        for edge_arg in edge_args:
            for file_path in edge_arg["csv_paths"]:
                if not set(
                    [
                        "id",
                        "start_id",
                        "start_vertex_type",
                        "end_id",
                        "end_vertex_type",
                    ]
                ).issubset(set(columns_in_csvs[file_path])):
                    log.error(
                        f"Columns in CSV file '{file_path}' didn't match to edge_args."
                    )
                    raise ValueError


class AzureExtensions:
    def __init__(
        self,
        subscription_id: str = "",
        resource_group_name: str = "",
        pg_server_name: str = "",
        pool: AsyncConnectionPool = None,
        extensions: list = [],
    ):
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        self.pg_server_name = pg_server_name
        self.pool = pool
        self.extensions = extensions

    def enable(self) -> None:
        """
        Enable the Azure Storage Extension for PostgreSQL Flex Server.

        Args:
            extension_names (list): Extension Names
        """
        log.debug("Enabling Extensions")
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.postgresqlflexibleservers import PostgreSQLManagementClient

        client = PostgreSQLManagementClient(
            credential=DefaultAzureCredential(),
            subscription_id=self.subscription_id,
        )
        configuration_names = ["azure.extensions", "shared_preload_libraries"]
        enabled = True
        for extension_name in self.extensions:
            for configuration_name in configuration_names:
                try:
                    configuration = client.configurations.get(
                        resource_group_name=self.resource_group_name,
                        server_name=self.pg_server_name,
                        configuration_name=configuration_name,
                    )
                    if extension_name not in configuration.value:
                        enabled = False
                        log.info(f"Updating configuration '{configuration.value}'...")
                        new_value = f"{configuration.value},{extension_name}"
                        client.configurations.begin_update(
                            resource_group_name=self.resource_group_name,
                            server_name=self.pg_server_name,
                            configuration_name=configuration_name,
                            parameters={
                                "value": new_value,
                                "source": "user-override",
                            },
                        ).result()

                except Exception as e:
                    print(f"An error occurred: {e}")

        if not enabled:
            log.info(f"Restarting server '{self.pg_server_name}'...")
            client.servers.begin_restart(
                resource_group_name=self.resource_group_name,
                server_name=self.pg_server_name,
            ).result()

    async def create(self) -> None:
        """
        Create the Azure Extensions for PostgreSQL Flex Server.

        Args:
            extension_names (list): Extension Names
        """
        async with self.pool.connection() as conn:
            for extension_name in self.extensions:
                await conn.execute(f"CREATE EXTENSION IF NOT EXISTS {extension_name}")


class StorageAccount:
    def __init__(
        self,
        subscription_id: str = "",
        resource_group_name: str = "",
        location: str = "",
        pool: AsyncConnectionPool = None,
    ):
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        self.location = location
        self.storage_account_name = ""
        self.blob_container_name = ""
        self.access_key = ""
        self.pool = pool

    def __del__(self):
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.storage import StorageManagementClient

        client = StorageManagementClient(
            credential=DefaultAzureCredential(),
            subscription_id=self.subscription_id,
        )

        client.storage_accounts.delete(
            self.resource_group_name,
            self.storage_account_name,
        )

    def create(self) -> None:
        """
        Create a Storage Account and a Blob Container.
        """
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.storage import StorageManagementClient
        import uuid

        # Storage account name must be between 3 and 24 characters in length and use numbers and lower-case letters only.
        prefix = "agefreighter".lower()
        uid = str(uuid.uuid4())[:8].replace("-", "")
        self.storage_account_name = f"sa{prefix}{uid}"
        self.blob_container_name = f"bc{prefix}{uid}"

        client = StorageManagementClient(
            credential=DefaultAzureCredential(),
            subscription_id=self.subscription_id,
        )
        log.info(f"Creating storage account '{self.storage_account_name}'...")
        client.storage_accounts.begin_create(
            self.resource_group_name,
            self.storage_account_name,
            {
                "location": self.location,
                "kind": "StorageV2",
                "sku": {"name": "Standard_LRS"},
            },
        ).result()
        log.info(f"Creating blob container '{self.blob_container_name}'...")
        client.blob_containers.create(
            self.resource_group_name,
            self.storage_account_name,
            self.blob_container_name,
            {},
        )
        keys = client.storage_accounts.list_keys(
            self.resource_group_name, self.storage_account_name
        )
        self.access_key = [v.value for v in keys.keys][0]

    async def attach(self) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                f"SELECT azure_storage.account_add('{self.storage_account_name}', '{self.access_key}');"
            )


class BlobUploader:
    def __init__(
        self,
        storage_account_name: str = "",
        access_key: str = "",
        blob_container_name: str = "",
        file_paths_dict: dict = {},
        lines_per_chunk: int = 10000,
    ):
        self.storage_account_name = storage_account_name
        self.access_key = access_key
        self.blob_container_name = blob_container_name
        self.file_paths_dict = file_paths_dict
        self.lines_per_chunk = lines_per_chunk

    async def upload(self) -> None:
        """
        Upload a CSV file to the Blob Container.

        Args:
            None
        """
        self.splitFile()
        log.info("Uploading files...")
        import asyncio
        from azure.storage.blob.aio import BlobServiceClient

        try:
            async with BlobServiceClient(
                account_url=f"https://{self.storage_account_name}.blob.core.windows.net",
                credential=self.access_key,
            ) as blob_service_client:
                container_client = blob_service_client.get_container_client(
                    self.blob_container_name
                )
                tasks = []
                for type, file_paths in self.file_paths_dict.items():
                    for file_path, tmp_file_list in self.tmp_file_lists.items():
                        for tmp_file_path in tmp_file_list:
                            tasks.append(
                                self.uploadBlob(tmp_file_path, container_client)
                            )
                await asyncio.gather(*tasks)
                log.info("Successfully uploaded all files to blob.")

        except Exception as e:
            log.error(f"An error occurred while uploading files to blob: {e}")

    async def uploadBlob(self, file_path, container_client):
        import os

        blob_name = os.path.basename(file_path)
        async with container_client.get_blob_client(blob_name) as blob_client:
            log.info(f"Uploading {file_path} to blob {blob_name}...")
            with open(file_path, "rb") as data:
                await blob_client.upload_blob(data, overwrite=True)

    def splitFile(self) -> None:
        log.info("Splitting file into chunks...")
        import os
        import mmap

        self.columns_in_csvs = {}
        self.total_lines = {}
        self.tmp_file_lists = {}
        for type, file_paths in self.file_paths_dict.items():
            for file_path in file_paths:
                # column names header
                columns_in_csv, newline_char = self.getColumnsInCSV(csv_path=file_path)
                self.columns_in_csvs[file_path] = columns_in_csv
                header_line = (
                    ",".join([f'"{col}"' for col in columns_in_csv]) + newline_char
                )

                original_file_name = os.path.basename(file_path)
                original_file_name_without_ext, ext = os.path.splitext(
                    original_file_name
                )
                temp_file_name = (
                    f"{original_file_name_without_ext}_part_{{count:05d}}{ext}"
                )

                total_lines = 0
                with open(file_path, "r+") as f:
                    mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    line_iterator = iter(mmapped_file.readline, b"")
                    temp_files = []
                    line_count = 0
                    file_count = 0

                    temp_file = self.createTempFile(
                        temp_file_name=temp_file_name,
                        count=file_count,
                        header_line=header_line,
                        add_header=False,
                    )
                    temp_files.append(temp_file.name)

                    for line in line_iterator:
                        temp_file.write(line)
                        line_count += 1
                        total_lines += 1
                        if line_count >= self.lines_per_chunk:
                            temp_file.close()
                            file_count += 1
                            line_count = 0
                            temp_file = self.createTempFile(
                                temp_file_name=temp_file_name,
                                count=file_count,
                                header_line=header_line,
                                add_header=True,
                            )
                            temp_files.append(temp_file.name)
                    temp_file.close()
                    mmapped_file.close()

                    self.total_lines[file_path] = total_lines - 1
                    self.tmp_file_lists[file_path] = temp_files

    def createTempFile(
        self,
        temp_file_name: str = "",
        count: int = 0,
        header_line: str = "",
        add_header: bool = True,
    ):
        import tempfile
        import os

        temp_file_path = os.path.join(
            tempfile.gettempdir(), temp_file_name.format(count=count)
        )
        temp_file = open(temp_file_path, "w+b")
        if add_header:
            temp_file.write(header_line.encode("utf-8"))
        return temp_file

    def getColumnsInCSV(self, csv_path: str = "") -> list:
        """
        Get the columns in a CSV file.

        Args:
            csv (str): CSV file path

        Returns:
            list: Columns in the CSV file
        """
        import csv

        newline_char = None
        columns = []

        with open(csv_path, "r", newline="") as f:
            # Detect the newline character
            sample = f.read(8192)
            if "\r\n" in sample:
                newline_char = "\r\n"
            elif "\n" in sample:
                newline_char = "\n"
            elif "\r" in sample:
                newline_char = "\r"

            # Reset the file pointer to the beginning
            f.seek(0)

            reader = csv.reader(f)
            columns = next(reader)

        return columns, newline_char


class TempTables:
    def __init__(
        self,
        columns_in_csvs: list = [],
        file_paths_dict: dict = {},
        pool: AsyncConnectionPool = None,
    ):
        import os

        self.file_paths_dict = file_paths_dict
        self.tbls_from_storage = {}
        self.columns_in_tbls_from_storage = {}
        self.id_map_tbls = {}
        for type, file_paths in self.file_paths_dict.items():
            for file_path in file_paths:
                columns_in_csv = columns_in_csvs[file_path]
                base_name_wo_ext, ext = os.path.splitext(os.path.basename(file_path))
                self.tbls_from_storage[file_path] = f"{base_name_wo_ext.lower()}_temp"
                self.columns_in_tbls_from_storage[file_path] = ",".join(
                    [f'"{column}" TEXT' for column in columns_in_csvs[file_path]]
                )
                if type == "nodes":
                    self.id_map_tbls[file_path] = f"{base_name_wo_ext.lower()}_id_map"
        self.pool = pool

    async def create(self) -> None:
        log.info("Creating temporary tables...")
        async with self.pool.connection() as conn:
            for type, file_paths in self.file_paths_dict.items():
                for file_path in file_paths:
                    tbl_from_storage = self.tbls_from_storage[file_path]
                    columns_in_tbl_from_storage = self.columns_in_tbls_from_storage[
                        file_path
                    ]
                    query = f"DROP TABLE IF EXISTS public.{tbl_from_storage};"
                    await conn.execute(query)
                    query = f"CREATE TABLE public.{tbl_from_storage} ({columns_in_tbl_from_storage});"
                    await conn.execute(query)

                    if type == "nodes":
                        id_map_tbl = self.id_map_tbls[file_path]
                        query = f"DROP TABLE IF EXISTS public.{id_map_tbl};"
                        await conn.execute(query)
                        query = f"CREATE TABLE public.{id_map_tbl} (entryID TEXT, id BIGINT);"
                        await conn.execute(query)

    async def delete(self) -> None:
        async with self.pool.connection() as conn:
            for type, file_paths in self.file_paths_dict.items():
                for file_path in file_paths:
                    tbl_from_storage = self.tbls_from_storage[file_path]
                    query = f"DROP TABLE public.{tbl_from_storage};"
                    await conn.execute(query)
                    if type == "nodes":
                        id_map_tbl = self.id_map_tbls[file_path]
                        query = f"DROP TABLE public.{id_map_tbl};"
                        await conn.execute(query)


class StorageLoader:
    def __init__(
        self,
        tmp_file_lists: dict = {},
        total_lines: dict = {},
        storage_account_name: str = "",
        blob_container_name: str = "",
        tbls_from_storage: dict = {},
        columns_in_tbls_from_storage: dict = {},
        columns_in_csvs: list = [],
        pool: AsyncConnectionPool = None,
    ):
        import os

        self.tmp_file_lists = {}
        for file_path, tmp_file_list in tmp_file_lists.items():
            self.tmp_file_lists[file_path] = list(
                map(lambda file_path: os.path.basename(file_path), tmp_file_list)
            )
        self.total_lines = total_lines
        self.storage_account_name = storage_account_name
        self.blob_container_name = blob_container_name
        self.tbls_from_storage = tbls_from_storage
        self.columns_in_tbls_from_storage = columns_in_tbls_from_storage
        self.columns_in_csvs = columns_in_csvs
        self.pool = pool

    async def load(self):
        import asyncio
        from psycopg.rows import namedtuple_row

        log.info("Loading files into temporary tables...")

        tasks = []
        for file_path, tmp_file_list in self.tmp_file_lists.items():
            tbl_from_storage = self.tbls_from_storage[file_path]
            columns_in_tbl_from_storage = self.columns_in_tbls_from_storage[file_path]
            for tmp_file in tmp_file_list:
                tasks.append(
                    self.executeQuery(
                        self.pool,
                        f"""INSERT INTO public.{tbl_from_storage}
                            SELECT *
                            FROM azure_storage.blob_get(
                                '{self.storage_account_name}',
                                '{self.blob_container_name}',
                                '{tmp_file}',
                                options := azure_storage.options_csv_get(header := 'true'))
                            AS res ({columns_in_tbl_from_storage});
                        """,
                    )
                )

        await asyncio.gather(*tasks)

        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                for file_path, tbl_from_storage in self.tbls_from_storage.items():
                    await cur.execute(f"SELECT COUNT(*) FROM {tbl_from_storage}")
                    result = await cur.fetchone()
                    total_rows_in_tbl = result.count
                    if total_rows_in_tbl == self.total_lines[file_path]:
                        log.info("Successfully loaded all files to temporary table.")
                    else:
                        log.error(
                            f"Total rows in the table '{tbl_from_storage}' is not equal to the total lines in the file."
                        )
                        raise ValueError

        log.info("Creating indexes...")
        tasks = []
        for file_path, tmp_file_list in self.tmp_file_lists.items():
            tbl_from_storage = self.tbls_from_storage[file_path]
            for col in self.columns_in_csvs[file_path]:
                tasks.append(
                    self.executeQuery(
                        self.pool,
                        f"CREATE INDEX ON public.{tbl_from_storage} USING BTREE ({col});",
                    )
                )
        await asyncio.gather(*tasks)

    async def executeQuery(
        self, pool: AsyncConnectionPool = None, query: str = ""
    ) -> None:
        try:
            async with pool.connection() as conn:
                await conn.set_autocommit(True)
                await conn.execute(query)
        except Exception as e:
            log.debug(f"Error: {e}, in {sys._getframe().f_code.co_name}.")


class GraphLoader:
    def __init__(
        self,
        tbls_from_storage: str = "",
        total_lines: int = 0,
        vertex_args: list = [],
        edge_args: list = [],
        columns_in_csvs: list = [],
        id_map_tbls: dict = {},
        graph_name: str = "",
        records_per_thread: int = 0,
        pool: AsyncConnectionPool = None,
    ):
        self.tbls_from_storage = tbls_from_storage
        self.total_lines = total_lines
        self.vertex_args = vertex_args
        self.edge_args = edge_args
        self.columns_in_csvs = columns_in_csvs
        self.id_map_tbls = id_map_tbls
        self.graph_name = graph_name
        self.records_per_thread = records_per_thread
        self.pool = pool

    async def load(self) -> None:
        """
        Load a graph data from the Temporary Table
        """
        log.info(f"Creating graph from table, '{self.tbls_from_storage}'...")
        import asyncio
        from psycopg.rows import namedtuple_row

        for vertex_arg in self.vertex_args:
            first_id = await self.getFirstId(
                graph_name=self.graph_name, label_type=vertex_arg["label"]
            )
            props_formatted = ",".join(
                [f'"{prop}":"%s"' for prop in vertex_arg["props"]]
            )

            log.info("Creating vertices...")
            tasks = [
                self.executeQuery(
                    self.pool,
                    f"""
                    INSERT INTO "{self.graph_name}"."{vertex_arg["label"]}" (properties)
                        SELECT format('{{"id":"%s", {props_formatted}}}', "{vertex_arg["id"]}", {",".join([f'"{prop}"' for prop in vertex_arg["props"]])})::agtype
                        FROM (
                            SELECT DISTINCT {",".join([f'"{item}"' for item in [vertex_arg["id"]] + vertex_arg["props"]])}
                            FROM {self.tbls_from_storage[vertex_arg["csv_path"]]}
                            OFFSET {offset} LIMIT {self.records_per_thread}
                        ) AS distinct_s;
                    INSERT INTO {self.id_map_tbls[vertex_arg["csv_path"]]} (entryID, id)
                        SELECT distinct_v."{vertex_arg["id"]}", {first_id} + {offset} + ROW_NUMBER() OVER () - 1
                        FROM (
                            SELECT DISTINCT "{vertex_arg["id"]}"
                            FROM {self.tbls_from_storage[vertex_arg["csv_path"]]}
                            OFFSET {offset} LIMIT {self.records_per_thread}
                        ) AS distinct_v;""",
                )
                for offset in range(
                    0, self.total_lines[vertex_arg["csv_path"]], self.records_per_thread
                )
            ]
            await asyncio.gather(*tasks)

            log.info(
                f"Creating indexes on {self.id_map_tbls[vertex_arg['csv_path']]}..."
            )
            tasks = [
                self.executeQuery(self.pool, query)
                for query in [
                    f"CREATE INDEX ON {self.id_map_tbls[vertex_arg['csv_path']]} USING BTREE (entryID);",
                ]
            ]
            await asyncio.gather(*tasks)

            log.info("Creating indexes for node...")
            tasks = [
                self.executeQuery(self.pool, query)
                for query in [
                    f'CREATE INDEX ON "{self.graph_name}"."{vertex_arg["label"]}" USING GIN (properties);',
                    f'CREATE INDEX ON "{self.graph_name}"."{vertex_arg["label"]}" USING BTREE (id);',
                ]
            ]
            await asyncio.gather(*tasks)

        log.info("Creating edges...")
        for edge_arg in self.edge_args:
            edge_type = edge_arg["type"]
            for csv_path in edge_arg["csv_paths"]:
                tbl_from_storage = self.tbls_from_storage[csv_path]
                async with self.pool.connection() as conn:
                    async with conn.cursor(row_factory=namedtuple_row) as cur:
                        await cur.execute(
                            f"SELECT start_vertex_type, end_vertex_type FROM {tbl_from_storage} LIMIT 1;"
                        )
                        row = await cur.fetchone()
                        start_vertex_type = row.start_vertex_type.lower()
                        end_vertex_type = row.end_vertex_type.lower()

                prop_cols = [
                    col
                    for col in self.columns_in_csvs[csv_path]
                    if col
                    not in [
                        "id",
                        "start_id",
                        "start_vertex_type",
                        "end_id",
                        "end_vertex_type",
                    ]
                ]
                prop_cols_joined = ""
                if prop_cols:
                    prop_cols_joined = "," + ",".join(
                        [f'"{prop}"' for prop in prop_cols]
                    )
                    prop_vals = (
                        ", format('{"
                        + ",".join([f'"{prop}":"%s"' for prop in prop_cols])
                        + "}', "
                        + ",".join([f'af."{prop}"' for prop in prop_cols])
                        + ")::agtype"
                    )
                tasks = [
                    self.executeQuery(
                        self.pool,
                        f"""
                        INSERT INTO "{self.graph_name}"."{edge_type}" (start_id, end_id{", properties" if prop_cols_joined else ""})
                        SELECT s_map.id::agtype::graphid, e_map.id::agtype::graphid {prop_vals if prop_cols_joined else ""}
                        FROM (
                            SELECT start_id, start_vertex_type, end_id, end_vertex_type {prop_cols_joined if prop_cols_joined else ""}
                            FROM {tbl_from_storage}
                            OFFSET {offset} LIMIT {self.records_per_thread}
                        ) AS af
                        JOIN {start_vertex_type}_id_map AS s_map ON af.start_id = s_map.entryID
                        JOIN {end_vertex_type}_id_map AS e_map ON af.end_id = e_map.entryID;""",
                    )
                    for offset in range(
                        0, self.total_lines[csv_path], self.records_per_thread
                    )
                ]
                await asyncio.gather(*tasks)

            log.info("Creating indexes for edge...")
            tasks = [
                self.executeQuery(self.pool, query)
                for query in [
                    f'CREATE INDEX ON "{self.graph_name}"."{edge_type}" USING BTREE (start_id);',
                    f'CREATE INDEX ON "{self.graph_name}"."{edge_type}" USING BTREE (end_id);',
                ]
            ]
            await asyncio.gather(*tasks)

    async def executeQuery(
        self, pool: AsyncConnectionPool = None, query: str = ""
    ) -> None:
        try:
            async with pool.connection() as conn:
                await conn.set_autocommit(True)
                await conn.execute("SET statement_timeout = '3600s';")
                await conn.execute(query)

        except Exception as e:
            log.debug(f"Error: {e}, in {sys._getframe().f_code.co_name}.")

    # avoid to affect agefreighter class
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

    # avoid to affect agefreighter class
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
