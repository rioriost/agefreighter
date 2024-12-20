from agefreighter import AgeFreighter

import logging

log = logging.getLogger(__name__)


class PGFreighter(AgeFreighter):
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
        source_pg_con_string: str = "",
        source_schema: str = "public",
        source_tables: list = [],
        id_map: dict = {},
        graph_name: str = "",
        chunk_size: int = 128,
        direct_loading: bool = False,
        create_graph: bool = False,
        use_copy: bool = True,
        **kwargs,
    ) -> None:
        """
        Load data from a PostgreSQL database.

        Args:
            source_pg_con_string (str): The connection string of the source PostgreSQL database.
            source_schema (str): The source schema.
            source_tables (list): The source tables.
            id_map (dict): The ID map.
            graph_name (str): The name of the graph to load the data into.
            chunk_size (int): The size of the chunks to create.
            direct_loading (bool): Whether to load the data directly.
            create_graph (bool): Whether to create the graph.
            use_copy (bool): Whether to use the COPY protocol to load the data.

        Returns:
            None
        """
        log.debug("Loading data from a PostgreSQL database")
        import pandas as pd
        import psycopg as pg
        from psycopg.rows import namedtuple_row

        CHUNK_MULTIPLIER = 10000

        try:
            with pg.connect(source_pg_con_string) as conn:
                with conn.cursor(row_factory=namedtuple_row) as cur:
                    await self.setUpGraph(
                        graph_name=graph_name, create_graph=create_graph
                    )
                    for src_table in source_tables.values():
                        if id_map.get(src_table) is not None:  # nodes
                            id_col_name = id_map[src_table]
                            await self.createLabelType(
                                label_type="vertex", value=src_table
                            )
                            cur.execute(
                                f'SELECT COUNT(*) FROM {source_schema}."{src_table}"'
                            )
                            cnt = cur.fetchone()[0]
                            for i in range(0, cnt, chunk_size * CHUNK_MULTIPLIER):
                                cur.execute(
                                    f'SELECT * FROM {source_schema}."{src_table}" LIMIT {chunk_size * CHUNK_MULTIPLIER} OFFSET {i}'
                                )
                                rows = cur.fetchall()
                                vertices = pd.DataFrame(rows)
                                vertices.rename(
                                    columns={id_col_name: "id"}, inplace=True
                                )
                                await self.createVertices(
                                    vertices=vertices,
                                    vertex_label=src_table,
                                    chunk_size=chunk_size,
                                    direct_loading=direct_loading,
                                    use_copy=use_copy,
                                )
                        else:  # edges
                            await self.createLabelType(
                                label_type="edge", value=src_table
                            )
                            cur.execute(
                                f'SELECT COUNT(*) FROM {source_schema}."{src_table}"'
                            )
                            cnt = cur.fetchone()[0]
                            for i in range(0, cnt, chunk_size * CHUNK_MULTIPLIER):
                                cur.execute(
                                    f'SELECT * FROM {source_schema}."{src_table}" LIMIT {chunk_size * CHUNK_MULTIPLIER} OFFSET {i}'
                                )
                                rows = cur.fetchall()
                                edges = pd.DataFrame(rows)
                                edge_props = [
                                    e
                                    for e in edges.columns
                                    if e not in ["start_id", "end_id"]
                                ]
                                edges.insert(0, "start_v_label", list(id_map.keys())[0])
                                edges.insert(0, "end_v_label", list(id_map.keys())[1])
                                await self.createEdges(
                                    edges=edges,
                                    edge_type=src_table,
                                    edge_props=edge_props,
                                    chunk_size=chunk_size,
                                    direct_loading=direct_loading,
                                    use_copy=use_copy,
                                )
        except Exception as e:
            raise e

        await self.close()
