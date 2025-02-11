from agefreighter import AgeFreighter
from psycopg_pool import AsyncConnectionPool
import sys
import warnings
import logging
import io
from typing import Optional

log = logging.getLogger(__name__)


class AzureStorageFreighter(AgeFreighter):
    def __init__(self):
        super().__init__()
        # Make sure to declare attributes with non‐optional types (or check before use)
        self.subscription_id: str = ""
        self.location: str = ""
        self.storage_account_name: str = ""
        self.blob_container_name: str = ""
        self.access_key: str = ""
        self.pg_fqdn: str = ""
        self.pg_server_name: str = ""
        self.resource_group_name: str = ""
        self.progress: bool = False
        self.pool: AsyncConnectionPool

        # Ensure that the DSN is a string (if defined in the parent) so that re.match has a string argument.
        # For example, if AgeFreighter defines self.dsn as Optional[str], you might want:
        if self.dsn is None:
            raise ValueError("dsn must be defined as a non-None string")
        # Also, ensure that self.pool is assigned (or later assert it’s non-None)

    async def __aenter__(self):
        await super().__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await super().__aexit__(exc_type, exc, tb)
        if exc_type:
            print(f"Exception: {exc_type}, {exc}")

    async def load(
        self,
        csv_path: str = "",
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
        create_graph: bool = True,
        **kwargs,
    ) -> None:
        """
        Load a graph data to the PostgreSQL Flex with Azure Storage.

        Args:
            csv_path (str): CSV file path
            start_v_label (str): Start Vertex Label
            start_id (str): Start Vertex ID
            start_props (list): Start Vertex Properties
            edge_type (str): Edge Type
            edge_props (list): Edge Properties
            end_v_label (str): End Vertex Label
            end_id (str): End Vertex ID
            end_props (list): End Vertex Properties
            graph_name (str): The name of the graph to load the data into.
            chunk_size (int): The size of the chunks to create.
            create_graph (bool): Whether to create the graph.
            **kwargs: Additional keyword arguments

        Keyword Args:
            subscription_id (str): Azure Subscription ID

        Returns:
            None
        """
        log.debug("Loading data from Azure Storage")

        if "csv" in kwargs.keys():
            warnings.warn(
                "The 'csv' parameter is deprecated. Please use 'csv_path' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            csv_path = kwargs["csv"]

        if "progress" in kwargs.keys():
            self.progress = kwargs["progress"]

        CHUNK_MULTIPLIER = 10000
        TBL_FROM_STORAGE = "table_from_azure_storage"
        TBL_ID_MAP_S = "table_id_map_s"
        TBL_ID_MAP_E = "table_id_map_e"

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

        # Upload a CSV file to the blob container
        print("Uploading file...")
        bl = BlobUploader(
            storage_account_name=self.storage_account_name,
            blob_container_name=self.blob_container_name,
            access_key=self.access_key,
            file_path=csv_path,
            lines_per_chunk=chunk_size * CHUNK_MULTIPLIER,
        )
        await bl.upload()

        # Check if the columns contain arguments
        self.checkColumns(
            columns_in_csv=bl.columns_in_csv,
            columns_in_args=[
                start_id,
                *start_props,
                *edge_props,
                end_id,
                *end_props,
            ],
        )

        # Create temporary tables
        # They're named 'temp', but not the persistent tables.
        print("Creating temporary tables...")
        tt = TempTables(
            tbl_from_storage=TBL_FROM_STORAGE,
            columns_in_csv=bl.columns_in_csv,
            tbl_id_map_s=TBL_ID_MAP_S,
            tbl_id_map_e=TBL_ID_MAP_E,
            pool=self.pool,
        )
        await tt.create()

        # Load the data from the Azure Storage to the temporary table
        print("Loading files to temporary table...")
        sl = StorageLoader(
            file_list=bl.file_list,
            total_lines=bl.total_lines,
            storage_account_name=self.storage_account_name,
            blob_container_name=self.blob_container_name,
            table_name=TBL_FROM_STORAGE,
            columns_in_tbl_from_storage=tt.columns_in_tbl_from_storage,
            pool=self.pool,
        )
        await sl.load()

        # start to create a graph
        print("Creating a graph...")
        await self.setUpGraph(graph_name=graph_name, create_graph=create_graph)
        await self.createLabelType(label_type="vertex", value=start_v_label)
        await self.createLabelType(label_type="vertex", value=end_v_label)
        await self.createLabelType(label_type="edge", value=edge_type)

        gl = GraphLoader(
            tbl_from_storage=TBL_FROM_STORAGE,
            total_lines=bl.total_lines,
            tbl_id_map_s=TBL_ID_MAP_S,
            tbl_id_map_e=TBL_ID_MAP_E,
            start_v_label=start_v_label,
            start_id=start_id,
            start_props=start_props,
            edge_type=edge_type,
            edge_props=edge_props,
            end_v_label=end_v_label,
            end_id=end_id,
            end_props=end_props,
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
        match = pattern.match(self.dsn)
        if match:
            self.pg_fqdn = match.group("host")
        else:
            log.error("Failed to get the PostgreSQL FQDN.")
            raise ValueError
        self.pg_server_name = self.pg_fqdn.split(".")[0]
        servers = client.servers.list()
        pattern = re.compile(
            r"/subscriptions/[^/]+/resourceGroups/(?P<resourceGroup>[^/]+)/providers"
        )
        for server in servers:
            if server.fully_qualified_domain_name == self.pg_fqdn:
                if not server.id:
                    log.error("Failed to get the server ID.")
                    raise ValueError
                match = pattern.match(server.id)
                if match:
                    self.resource_group_name = match.group("resourceGroup")
                    self.location = server.location
                    return True
        return False

    @staticmethod
    def checkColumns(columns_in_csv: list = [], columns_in_args: list = []) -> None:
        """
        Check if the columns contain arguments.

        Args:
            columns_in_csv (list): Columns in the CSV file
            columns_in_args (list): Columns in the arguments

        Raises:
            ValueError: If the column is not in the CSV

        Returns:
            None
        """
        log.debug("Checking keys")
        for column in columns_in_args:
            if column not in columns_in_csv:
                log.error(f"Column '{column}' is not in the CSV.")
                raise ValueError


class AzureExtensions:
    def __init__(
        self,
        subscription_id: str = "",
        resource_group_name: str = "",
        pg_server_name: str = "",
        pool: Optional[AsyncConnectionPool] = None,
        extensions: list = [],
    ):
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        self.pg_server_name = pg_server_name
        if pool is None:
            raise ValueError("pool must be provided for StorageAccount")
        self.pool = pool
        self.extensions = extensions

    def enable(self) -> None:
        """
        Enable the Azure Storage Extension for PostgreSQL Flex Server.

        Returns:
            None
        """
        log.debug("Enabling Extensions")
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.postgresqlflexibleservers import PostgreSQLManagementClient
        from azure.mgmt.postgresqlflexibleservers.models import ConfigurationForUpdate

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
                    if (
                        configuration.value
                        and extension_name not in configuration.value
                    ):
                        enabled = False
                        log.info(f"Updating configuration '{configuration.value}'...")
                        new_value = f"{configuration.value},{extension_name}"

                        # Use the model class instead of a dict:
                        parameters = ConfigurationForUpdate(
                            value=new_value, source="user-override"
                        )

                        client.configurations.begin_update(
                            resource_group_name=self.resource_group_name,
                            server_name=self.pg_server_name,
                            configuration_name=configuration_name,
                            parameters=parameters,
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

        Returns:
            None
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
        pool: Optional[AsyncConnectionPool] = None,
    ):
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        self.location = location
        self.storage_account_name = ""
        self.blob_container_name = ""
        self.access_key = ""
        if pool is None:
            raise ValueError("pool must be provided for StorageAccount")
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

        Returns:
            None
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
        file_path: str = "",
        lines_per_chunk: int = 10000,
    ):
        self.storage_account_name = storage_account_name
        self.access_key = access_key
        self.blob_container_name = blob_container_name
        self.file_path = file_path
        self.lines_per_chunk = lines_per_chunk

    async def upload(self) -> None:
        """
        Upload a CSV file to the Blob Container.

        Returns:
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
                tasks = [
                    self.uploadBlob(file_path, container_client)
                    for file_path in self.file_list
                ]
                await asyncio.gather(*tasks)
                log.info("Successfully uploaded all files to blob.")

        except Exception as e:
            log.error(f"An error occurred while uploading files to blob: {e}")

    async def uploadBlob(self, file_path, container_client) -> None:
        """
        Upload a blob to the container.

        Args:
            file_path (str): File path
            container_client (ContainerClient): Container client

        Returns:
            None
        """
        import os

        blob_name = os.path.basename(file_path)
        async with container_client.get_blob_client(blob_name) as blob_client:
            log.info(f"Uploading {file_path} to blob {blob_name}...")
            with open(file_path, "rb") as data:
                await blob_client.upload_blob(data, overwrite=True)

    def splitFile(self) -> None:
        """
        Split a file into chunks.

        Returns:
            None
        """
        log.info("Splitting file into chunks...")
        import os
        import mmap

        # column names header
        columns_in_csv, newline_char = self.getColumnsInCSV(csv_path=self.file_path)
        self.columns_in_csv = columns_in_csv
        header_line = ",".join([f'"{col}"' for col in columns_in_csv]) + newline_char

        original_file_name = os.path.basename(self.file_path)
        original_file_name_without_ext, ext = os.path.splitext(original_file_name)
        temp_file_name = f"{original_file_name_without_ext}_part_{{count:05d}}{ext}"

        total_lines = 0
        with open(self.file_path, "r+") as f:
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

            self.total_lines = total_lines - 1
            self.file_list = temp_files

    def createTempFile(
        self,
        temp_file_name: str = "",
        count: int = 0,
        header_line: str = "",
        add_header: bool = True,
    ) -> io.BufferedRandom:
        """
        Create a temporary file.

        Args:
            temp_file_name (str): Temporary file name
            count (int): Count
            header_line (str): Header line
            add_header (bool): Add header

        Returns:
            Temporary file
        """
        import tempfile
        import os

        temp_file_path = os.path.join(
            tempfile.gettempdir(), temp_file_name.format(count=count)
        )
        temp_file = open(temp_file_path, "w+b")
        if add_header:
            temp_file.write(header_line.encode("utf-8"))
        return temp_file

    def getColumnsInCSV(self, csv_path: str = "") -> tuple:
        """
        Get the columns in a CSV file.

        Args:
            csv (str): CSV file path

        Returns:
            tuple: Columns in the CSV file
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
        tbl_from_storage: str = "",
        columns_in_csv: list = [],
        tbl_id_map_s: str = "",
        tbl_id_map_e: str = "",
        pool: Optional[AsyncConnectionPool] = None,
    ):
        self.tbl_from_storage = tbl_from_storage
        self.columns_in_csv = columns_in_csv
        self.columns_in_tbl_from_storage = ",".join(
            [f'"{column}" TEXT' for column in columns_in_csv]
        )
        self.tbl_id_map_s = tbl_id_map_s
        self.tbl_id_map_e = tbl_id_map_e
        if pool is None:
            raise ValueError("pool must be provided for StorageAccount")
        self.pool = pool

    async def create(self) -> None:
        """
        Create temporary tables.

        Returns
            None
        """
        log.info("Creating temporary tables...")
        async with self.pool.connection() as conn:
            # create a temporary table to store the data from the Azure Storage
            query = f"DROP TABLE IF EXISTS public.{self.tbl_from_storage};"
            await conn.execute(query)
            query = f"CREATE TABLE public.{self.tbl_from_storage} ({self.columns_in_tbl_from_storage});"
            await conn.execute(query)

            # create a temporary table to store the mapping between the entryID and the id
            query = f"DROP TABLE IF EXISTS public.{self.tbl_id_map_s};"
            await conn.execute(query)
            query = (
                f"CREATE TABLE public.{self.tbl_id_map_s} (entryID TEXT, id BIGINT);"
            )
            await conn.execute(query)
            query = f"DROP TABLE IF EXISTS public.{self.tbl_id_map_e};"
            await conn.execute(query)
            query = (
                f"CREATE TABLE public.{self.tbl_id_map_e} (entryID TEXT, id BIGINT);"
            )
            await conn.execute(query)

    async def delete(self) -> None:
        """
        Delete temporary tables.

        Returns:
            None
        """
        if self.pool is None:
            raise ValueError("pool must be provided for TempTables")
        async with self.pool.connection() as conn:
            query = f"DROP TABLE public.{self.tbl_from_storage};"
            await conn.execute(query)
            query = f"DROP TABLE public.{self.tbl_id_map_s};"
            await conn.execute(query)
            query = f"DROP TABLE public.{self.tbl_id_map_e};"
            await conn.execute(query)


class StorageLoader:
    def __init__(
        self,
        file_list: list = [],
        total_lines: int = 0,
        storage_account_name: str = "",
        blob_container_name: str = "",
        table_name: str = "",
        columns_in_tbl_from_storage: str = "",
        pool: Optional[AsyncConnectionPool] = None,
    ):
        import os

        self.file_list = list(
            map(lambda file_path: os.path.basename(file_path), file_list)
        )
        self.total_lines = total_lines
        self.storage_account_name = storage_account_name
        self.blob_container_name = blob_container_name
        self.table_name = table_name
        self.columns_in_tbl_from_storage = columns_in_tbl_from_storage
        self.pool = pool

    async def load(self) -> None:
        """
        Load files into the temporary table.

        Returns:
            None
        """
        import asyncio

        if self.pool is None:
            raise ValueError("pool must be provided for StorageLoader")

        log.info("Loading files into temporary table...")

        tasks = [
            self.executeQuery(
                self.pool,
                f"""INSERT INTO public.{self.table_name}
                    SELECT *
                    FROM azure_storage.blob_get(
                        '{self.storage_account_name}',
                        '{self.blob_container_name}',
                        '{file}',
                        options := azure_storage.options_csv_get(header := 'true'))
                    AS res ({self.columns_in_tbl_from_storage});
                """,
            )
            for file in self.file_list
        ]
        await asyncio.gather(*tasks)

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                result = await cur.fetchone()
                if result is None:
                    log.error("Failed to load files to temporary table.")
                    raise ValueError
                total_rows_in_tbl = result[0]
                if total_rows_in_tbl == self.total_lines:
                    log.info("Successfully loaded all files to temporary table.")
                else:
                    log.error(
                        f"Total rows in the table '{self.table_name}' is not equal to the total lines in the file."
                    )
                    raise ValueError

        log.info("Creating indexes...")
        tasks = [
            self.executeQuery(
                self.pool,
                f"CREATE INDEX ON public.{self.table_name} USING BTREE ({col});",
            )
            for col in self.columns_in_tbl_from_storage
        ]
        await asyncio.gather(*tasks)

    async def executeQuery(
        self, pool: Optional[AsyncConnectionPool] = None, query: str = ""
    ) -> None:
        """
        Execute a query.

        Args:
            pool (AsyncConnectionPool): Connection pool
            query (str): Query

        Returns:
            None
        """
        if pool is None:
            raise ValueError("pool must be provided for StorageLoader")
        try:
            async with pool.connection() as conn:
                await conn.set_autocommit(True)
                await conn.execute(query)
        except Exception as e:
            log.debug(f"Error: {e}, in {sys._getframe().f_code.co_name}.")


class GraphLoader:
    def __init__(
        self,
        tbl_from_storage: str = "",
        total_lines: int = 0,
        tbl_id_map_s: str = "",
        tbl_id_map_e: str = "",
        start_v_label: str = "",
        start_id: str = "",
        start_props: list = [],
        edge_type: str = "",
        edge_props: list = [],
        end_v_label: str = "",
        end_id: str = "",
        end_props: list = [],
        graph_name: str = "",
        records_per_thread: int = 0,
        pool: Optional[AsyncConnectionPool] = None,
        progress: bool = False,
    ):
        self.tbl_from_storage = tbl_from_storage
        self.total_lines = total_lines
        self.tbl_id_map_s = tbl_id_map_s
        self.tbl_id_map_e = tbl_id_map_e
        self.start_v_label = start_v_label
        self.start_id = start_id
        self.start_props = start_props
        self.edge_type = edge_type
        self.edge_props = edge_props
        self.end_v_label = end_v_label
        self.end_id = end_id
        self.end_props = end_props
        self.graph_name = graph_name
        self.records_per_thread = records_per_thread
        self.pool = pool
        self.progress = progress

    async def load(self) -> None:
        """
        Load a graph data from the Temporary Table

        Returns:
            None
        """
        log.info(f"Creating graph from table, '{self.tbl_from_storage}'...")
        import asyncio

        first_id_s = await self.getFirstId(
            graph_name=self.graph_name, label_type=self.start_v_label
        )
        first_id_e = await self.getFirstId(
            graph_name=self.graph_name, label_type=self.end_v_label
        )
        start_props_formatted = ",".join(
            [f'"{prop}":"%s"' for prop in self.start_props]
        )
        end_props_formatted = ",".join([f'"{prop}":"%s"' for prop in self.end_props])

        log.info("Creating start vertices...")
        tasks = [
            self.executeQuery(
                self.pool,
                f"""
                INSERT INTO "{self.graph_name}"."{self.start_v_label}" (properties)
                    SELECT format('{{"id":"%s", {start_props_formatted}}}', "{self.start_id}", {",".join([f'"{start_prop}"' for start_prop in self.start_props])})::agtype
                    FROM (
                        SELECT DISTINCT {",".join([f'"{item}"' for item in [self.start_id] + self.start_props])}
                        FROM {self.tbl_from_storage}
                        OFFSET {offset} LIMIT {self.records_per_thread}
                    ) AS distinct_s;
                INSERT INTO {self.tbl_id_map_s} (entryID, id)
                    SELECT distinct_s."{self.start_id}", {first_id_s} + {offset} + ROW_NUMBER() OVER () - 1
                    FROM (
                        SELECT DISTINCT "{self.start_id}"
                        FROM {self.tbl_from_storage}
                        OFFSET {offset} LIMIT {self.records_per_thread}
                    ) AS distinct_s;""",
            )
            for offset in range(0, self.total_lines, self.records_per_thread)
        ]
        await asyncio.gather(*tasks)

        log.info("Creating end vertices...")
        tasks = [
            self.executeQuery(
                self.pool,
                f"""
                INSERT INTO "{self.graph_name}"."{self.end_v_label}" (properties)
                    SELECT format('{{"id":"%s", {end_props_formatted}}}', "{self.end_id}", {",".join([f'"{end_prop}"' for end_prop in self.end_props])})::agtype
                    FROM (
                        SELECT DISTINCT {",".join([f'"{item}"' for item in [self.end_id] + self.end_props])}
                        FROM {self.tbl_from_storage}
                        OFFSET {offset} LIMIT {self.records_per_thread}
                    ) AS distinct_e;
                INSERT INTO {self.tbl_id_map_e} (entryID, id)
                    SELECT distinct_e."{self.end_id}", {first_id_e} + {offset} + ROW_NUMBER() OVER () - 1
                    FROM (
                        SELECT DISTINCT "{self.end_id}"
                        FROM {self.tbl_from_storage}
                        OFFSET {offset} LIMIT {self.records_per_thread}
                    ) AS distinct_e;""",
            )
            for offset in range(0, self.total_lines, self.records_per_thread)
        ]
        await asyncio.gather(*tasks)

        log.info(f"Creating indexes on {self.tbl_id_map_s} / {self.tbl_id_map_e}...")
        tasks = [
            self.executeQuery(self.pool, query)
            for query in [
                f"CREATE INDEX ON {self.tbl_id_map_s} USING BTREE (entryID);",
                f"CREATE INDEX ON {self.tbl_id_map_e} USING BTREE (entryID);",
            ]
        ]
        await asyncio.gather(*tasks)

        log.info("Creating edges...")
        if self.edge_props:
            prop_cols = "," + ",".join([f'"{prop}"' for prop in self.edge_props])
            prop_vals = (
                ", format('{"
                + ",".join([f'"{prop}":"%s"' for prop in self.edge_props])
                + "}', "
                + ",".join([f'af."{prop}"' for prop in self.edge_props])
                + ")::agtype"
            )
        tasks = [
            self.executeQuery(
                self.pool,
                f"""
                INSERT INTO "{self.graph_name}"."{self.edge_type}" (start_id, end_id{", properties" if self.edge_props else ""})
                SELECT s_map.id::agtype::graphid, e_map.id::agtype::graphid {prop_vals if self.edge_props else ""}
                FROM (
                    SELECT "{self.start_id}", "{self.end_id}" {prop_cols if self.edge_props else ""}
                    FROM {self.tbl_from_storage}
                    OFFSET {offset} LIMIT {self.records_per_thread}
                ) AS af
                JOIN {self.tbl_id_map_s} AS s_map ON af."{self.start_id}" = s_map.entryID
                JOIN {self.tbl_id_map_e} AS e_map ON af."{self.end_id}" = e_map.entryID;""",
            )
            for offset in range(0, self.total_lines, self.records_per_thread)
        ]
        await asyncio.gather(*tasks)

        log.info("Creating indexes for graph...")
        tasks = [
            self.executeQuery(self.pool, query)
            for query in [
                f'CREATE INDEX ON "{self.graph_name}"."{self.start_v_label}" USING GIN (properties);',
                f'CREATE INDEX ON "{self.graph_name}"."{self.start_v_label}" USING BTREE (id);',
                f'CREATE INDEX ON "{self.graph_name}"."{self.end_v_label}" USING GIN (properties);',
                f'CREATE INDEX ON "{self.graph_name}"."{self.end_v_label}" USING BTREE (id);',
                f'CREATE INDEX ON "{self.graph_name}"."{self.edge_type}" USING BTREE (start_id);',
                f'CREATE INDEX ON "{self.graph_name}"."{self.edge_type}" USING BTREE (end_id);',
            ]
        ]
        await asyncio.gather(*tasks)

    async def executeQuery(
        self, pool: Optional[AsyncConnectionPool] = None, query: str = ""
    ) -> None:
        """
        Execute a query.

        Args:
            pool (AsyncConnectionPool): Connection pool
            query (str): Query

        Returns:
            None
        """
        if pool is None:
            raise ValueError("pool must be provided for GraphLoader")
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
            graph_name (str): The name of the graph.
            label_type (str): The label type.

        Returns:
            int: The first id.
        """
        import numpy as np

        if self.pool is None:
            raise ValueError("pool must be provided for GraphLoader")

        graph_name = self.quotedGraphName(graph_name)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                relation = f'{graph_name}."{label_type}"'
                await cur.execute(
                    f"SELECT id FROM ag_label WHERE relation='{relation}'::regclass;"
                )
                row = await cur.fetchone()
                if row is None:
                    raise ValueError("No row returned from query.")

                ENTRY_ID_BITS = 32 + 16
                ENTRY_ID_MASK = np.uint64(0x0000FFFFFFFFFFFF)
                first_id = ((np.uint64(row[0])) << ENTRY_ID_BITS) | (
                    (np.uint64(1)) & ENTRY_ID_MASK
                )

                return int(first_id)

    # avoid to affect agefreighter class
    @staticmethod
    def quotedGraphName(graph_name: str = "") -> str:
        """
        Quote the graph name.

        Args:
            graph_name (str): The name of the graph.

        Returns:
            str: The quoted graph name
        """
        log.debug(
            f"Quoting graph name {graph_name}, in {sys._getframe().f_code.co_name}."
        )
        if graph_name.lower() != graph_name:
            return f'"{graph_name}"'
        return graph_name
