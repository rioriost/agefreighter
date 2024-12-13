from agefreighter import AgeFreighter

import logging

log = logging.getLogger(__name__)


class AzureStorageFreighter(AgeFreighter):
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
        drop_graph: bool = False,
        **kwargs,
    ):
        """
        Load a graph data to the PostgreSQL Flex with Azure Storage.

        Args:
            csv (str): CSV file path
            start_v_label (str): Start Vertex Label
            start_id (str): Start Vertex ID
            start_props (list): Start Vertex Properties
            edge_type (str): Edge Type
            end_v_label (str): End Vertex Label
            end_id (str): End Vertex ID
            end_props (list): End Vertex Properties
            graph_name (str): Graph Name
            chunk_size (int): Chunk Size
            drop_graph (bool): Drop Graph

        Keyword Args:
            subscription_id (str): Azure Subscription ID
        """
        log.debug("Loading data from Azure Storage")
        # optional
        if "subscription_id" in kwargs.keys():
            if self.isValidAzureSubscriptionID(kwargs["subscription_id"]):
                self.subscription_id = kwargs["subscription_id"]
            else:
                log.error("Invalid Azure Subscription ID.")
                return

        if not hasattr(self, "subscription_id"):
            if not await self.findAzureSubscriptionID():
                log.error("Azure Subscription ID is not set.")
                return

        if not await self.setParameters():
            log.error("Parameters are not set.")
            return

        # add azure_storage to azure.extensions and shared_preload_libraries
        await self.enableExtensions(extension_names=["azure_storage"])

        # CREATE azure_storage extension
        await self.createAzureExtensions(extension_names=["azure_storage"])

        # create a storage account and a container
        await self.createStorageAccount()

        # upload a CSV file to the blob container
        await self.uploadToBlob(csv)

        # add the storage account to the PostgreSQL Flex
        await self.addStorageAccount()

        columns_in_csv = self.getColumnsInCSV(csv)

        await self.setUpGraph(graph_name=graph_name, drop_graph=drop_graph)
        await self.createLabelType(label_type="vertex", value=start_v_label)
        await self.createLabelType(label_type="vertex", value=end_v_label)
        await self.createLabelType(label_type="edge", value=edge_type)

        await self.executeUDF(
            csv=csv,
            columns_in_csv=columns_in_csv,
            start_v_label=start_v_label,
            start_id=start_id,
            start_props=start_props,
            edge_type=edge_type,
            end_v_label=end_v_label,
            end_id=end_id,
            end_props=end_props,
            graph_name=graph_name,
            chunk_size=chunk_size,
        )

        await self.close()

    async def findAzureSubscriptionID(self) -> bool:
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

        client = PostgreSQLManagementClient(
            credential=DefaultAzureCredential(), subscription_id=self.subscription_id
        )
        self.fqdn = self.getServerFQDN()
        self.server_name = self.fqdn.split(".")[0]
        servers = client.servers.list()
        for server in servers:
            if server.fully_qualified_domain_name == self.fqdn:
                if resource_group_name := self.getResourceGroupName(server.id):
                    self.resource_group_name = resource_group_name
                    self.location = server.location
                    return True
        return False

    def getServerFQDN(self) -> str:
        """
        Get the PostgreSQL Flex Server Name.

        Returns:
            str: PostgreSQL Flex Server Name
        """
        import re

        if not self.dsn:
            return None
        pattern = re.compile(r"host=(?P<host>[^ ]+)")
        return pattern.match(self.dsn).group("host")

    @staticmethod
    def getResourceGroupName(id: str = "") -> str:
        """
        Get the Resource Group Name from the Azure Resource ID.

        Args:
            id (str): Azure Resource ID

        Returns:
            str: Resource Group Name
        """
        import re

        if not id:
            return None
        pattern = re.compile(
            r"/subscriptions/[^/]+/resourceGroups/(?P<resourceGroup>[^/]+)/providers"
        )
        return pattern.match(id).group("resourceGroup")

    async def enableExtensions(self, extension_names: list = []) -> None:
        """
        Enable the Azure Storage Extension for PostgreSQL Flex Server.

        Args:
            extension_names (list): Extension Names
        """
        log.debug("Enabling Extensions")
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.postgresqlflexibleservers import PostgreSQLManagementClient

        client = PostgreSQLManagementClient(
            credential=DefaultAzureCredential(), subscription_id=self.subscription_id
        )
        configuration_names = ["azure.extensions", "shared_preload_libraries"]
        enabled = True
        for extension_name in extension_names:
            for configuration_name in configuration_names:
                try:
                    configuration = client.configurations.get(
                        resource_group_name=self.resource_group_name,
                        server_name=self.server_name,
                        configuration_name=configuration_name,
                    )
                    if extension_name not in configuration.value:
                        enabled = False
                        log.info(f"Updating configuration '{configuration.value}'...")
                        new_value = f"{configuration.value},{extension_name}"
                        client.configurations.begin_update(
                            resource_group_name=self.resource_group_name,
                            server_name=self.server_name,
                            configuration_name=configuration_name,
                            parameters={"value": new_value, "source": "user-override"},
                        ).result()

                except Exception as e:
                    print(f"An error occurred: {e}")

        if not enabled:
            log.info(f"Restarting server '{self.server_name}'...")
            client.servers.begin_restart(
                resource_group_name=self.resource_group_name,
                server_name=self.server_name,
            ).result()

    async def createAzureExtensions(self, extension_names: list = []) -> None:
        """
        Create the Azure Extensions for PostgreSQL Flex Server.

        Args:
            extension_names (list): Extension Names
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                for extension_name in extension_names:
                    await cur.execute(
                        f"CREATE EXTENSION IF NOT EXISTS {extension_name}"
                    )

    async def createStorageAccount(self) -> None:
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
            credential=DefaultAzureCredential(), subscription_id=self.subscription_id
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

    async def uploadToBlob(self, csv: str = "") -> None:
        """
        Upload a CSV file to the Blob Container.

        Args:
            csv (str): CSV file path
        """
        from azure.storage.blob import BlobServiceClient
        import os

        blob_service_client = BlobServiceClient(
            account_url=f"https://{self.storage_account_name}.blob.core.windows.net",
            credential=self.access_key,
        )
        blob_client = blob_service_client.get_blob_client(
            container=self.blob_container_name, blob=os.path.basename(csv)
        )
        log.info(f"Uploading '{csv}' to '{self.blob_container_name}'...")
        with open(csv, "rb") as data:
            blob_client.upload_blob(data)

    async def addStorageAccount(self) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT azure_storage.account_add('{self.storage_account_name}', '{self.access_key}');"
                )

    def getColumnsInCSV(self, csv_path: str = "") -> list:
        """
        Get the columns in a CSV file.

        Args:
            csv (str): CSV file path

        Returns:
            list: Columns in the CSV file
        """
        import csv

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            return next(reader)

    async def executeUDF(
        self,
        csv: str = "",
        columns_in_csv: list = [],
        start_v_label: str = "",
        start_id: str = "",
        start_props: list = [],
        edge_type: str = "",
        end_v_label: str = "",
        end_id: str = "",
        end_props: list = [],
        graph_name: str = "",
        chunk_size: int = 0,
    ) -> None:
        """
        Execute a UDF to load a graph data from the Azure Storage.

        Args:
            csv (str): CSV file path
            columns_in_csv (list): Columns in the CSV file
            start_v_label (str): Start Vertex Label
            start_id (str): Start Vertex ID
            start_props (list): Start Vertex Properties
            edge_type (str): Edge Type
            end_v_label (str): End Vertex Label
            end_id (str): End Vertex ID
            end_props (list): End Vertex Properties
            graph_name (str): Graph Name
            chunk_size (int): Chunk Size
        """
        log.debug(
            "Creating a UDF to load a graph data from the Azure Storage and executing it"
        )
        import os
        import psycopg as pg

        chunk_multiplier = 10000

        func_name = "load_graph_from_azure_storage"
        columns_in_temp_table = ",".join(
            [f"{column} TEXT" for column in columns_in_csv]
        )
        start_props_formatted = ",".join([f'"{prop}":"%s"' for prop in start_props])
        end_props_formatted = ",".join([f'"{prop}":"%s"' for prop in end_props])

        udf_query = f"""CREATE OR REPLACE FUNCTION {func_name}()
        RETURNS VOID AS $$
        DECLARE
            chunk RECORD;
            chunk_size BIGINT := {chunk_size * chunk_multiplier};
            num_offset BIGINT := 0;
            total_rows BIGINT;

            ENTRY_ID_BITS INTEGER := 32 + 16;
            ENTRY_ID_MASK BIGINT := 0x0000FFFFFFFFFFFF;
            oid BIGINT;
            first_id_s BIGINT;
            first_id_e BIGINT;
        BEGIN
            SET search_path = ag_catalog, "$user", public;

            -- create a temporary table to store the data from the Azure Storage
            CREATE TEMP TABLE temp_from_azure_storage ({columns_in_temp_table});

            -- bulk load from the Azure Storage into the temporary table
            INSERT INTO temp_from_azure_storage
            SELECT *
            FROM azure_storage.blob_get(
                '{self.storage_account_name}',
                '{self.blob_container_name}',
                '{os.path.basename(csv)}',
                options := azure_storage.options_csv_get(header := 'true'))
            AS res ({columns_in_temp_table});

            -- create a temporary table to store the mapping between the entryID and the id
            CREATE TEMP TABLE temp_id_map (entryID TEXT, id BIGINT);

            SELECT COUNT(*) INTO total_rows FROM temp_from_azure_storage;

            -- determine the first id for the start vertex
            SELECT id INTO oid FROM ag_label WHERE name='{start_v_label}';
            first_id_s := ((oid << ENTRY_ID_BITS) | (1 & ENTRY_ID_MASK));

            -- determine the first id for the end vertex
            SELECT id INTO oid FROM ag_label WHERE name='{end_v_label}';
            first_id_e := ((oid << ENTRY_ID_BITS) | (1 & ENTRY_ID_MASK));

            WHILE num_offset < total_rows LOOP
                -- bulk insert the start vertices
                INSERT INTO "{graph_name}"."{start_v_label}" (properties)
                SELECT format('{{"id":"%s", {start_props_formatted}}}', {start_id}, {','.join(start_props)})::agtype
                FROM (
                    SELECT DISTINCT {','.join([start_id] + start_props)}
                    FROM temp_from_azure_storage
                    OFFSET num_offset LIMIT chunk_size
                ) AS distinct_s;

                -- bulk insert the mapping between the entryID and the id
                INSERT INTO temp_id_map (entryID, id)
                SELECT distinct_s.{start_id}, first_id_s + ROW_NUMBER() OVER () - 1
                FROM (
                    SELECT DISTINCT {start_id}
                    FROM temp_from_azure_storage
                    OFFSET num_offset LIMIT chunk_size
                ) AS distinct_s;

                -- bulk insert the end vertices
                INSERT INTO "{graph_name}"."{end_v_label}" (properties)
                SELECT format('{{"id":"%s", {end_props_formatted}}}', {end_id}, {','.join(end_props)})::agtype
                FROM (
                    SELECT DISTINCT {','.join([end_id] + end_props)}
                    FROM temp_from_azure_storage
                    OFFSET num_offset LIMIT chunk_size
                ) AS distinct_e;

                -- bulk insert the mapping between the entryID and the id
                INSERT INTO temp_id_map (entryID, id)
                SELECT distinct_e.{end_id}, first_id_e + ROW_NUMBER() OVER () - 1
                FROM (
                    SELECT DISTINCT {end_id}
                    FROM temp_from_azure_storage
                    OFFSET num_offset LIMIT chunk_size
                ) AS distinct_e;

                -- bulk insert the edge data
                INSERT INTO "{graph_name}"."{edge_type}" (start_id, end_id)
                SELECT s_map.id::agtype::graphid, e_map.id::agtype::graphid
                FROM (
                    SELECT DISTINCT {start_id}, {end_id}
                    FROM temp_from_azure_storage
                    OFFSET num_offset LIMIT chunk_size
                ) AS af
                JOIN temp_id_map AS s_map ON af.{start_id} = s_map.entryID
                JOIN temp_id_map AS e_map ON af.{end_id} = e_map.entryID;

                num_offset := num_offset + chunk_size;
            END LOOP;

            CREATE INDEX ON "{graph_name}"."{start_v_label}" USING GIN (properties);
            CREATE INDEX ON "{graph_name}"."{start_v_label}" USING BTREE (id);

            CREATE INDEX ON "{graph_name}"."{end_v_label}" USING GIN (properties);
            CREATE INDEX ON "{graph_name}"."{end_v_label}" USING BTREE (id);

            CREATE INDEX ON "{graph_name}"."{edge_type}" USING BTREE (start_id);
            CREATE INDEX ON "{graph_name}"."{edge_type}" USING BTREE (end_id);

        END;
        $$ LANGUAGE plpgsql;
        """
        log.debug(udf_query)
        with pg.connect(self.dsn_wo_options) as conn:
            with conn.cursor() as cur:
                cur.execute(udf_query)
                cur.execute(f"SELECT {func_name}();")
                result = cur.fetchall()
                log.debug(result)
