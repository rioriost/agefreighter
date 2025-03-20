#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import logging

from agefreighter.cypherparser import CypherParser

from flask import Flask, jsonify, request, render_template  # type: ignore
from psycopg_pool import ConnectionPool

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)


class ConnectionStringParser:
    """Utility class to parse connection strings into a dictionary."""

    @staticmethod
    def parse(conn_str: str) -> dict:
        """Convert a connection string into a dictionary."""
        conn_dict = {}
        parts = conn_str.split()
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                conn_dict[key] = value
        return conn_dict


class CypherQueryFormatter:
    """Utility class for formatting Cypher queries for Apache AGE."""

    @staticmethod
    def format_query(graph_name: str, cypher_query: str) -> str:
        """
        Format the provided Cypher query for Apache AGE.

        Raises:
            ValueError: If the query is unsafe or incorrectly formatted.
        """
        if not CypherQueryFormatter.is_safe_cypher_query(cypher_query):
            raise ValueError("Unsafe query")

        # Append LIMIT 50 if no limit is specified.
        if "limit" not in cypher_query.lower():
            cypher_query += " LIMIT 50"

        returns = CypherQueryFormatter.get_return_values(cypher_query)

        # Check for parameterized query usage.
        if re.findall(r"\$(\w+)", cypher_query):
            raise ValueError("Parameterized query")

        if returns:
            ag_types = ", ".join([f"{r} agtype" for r in returns])
            return f"SELECT * FROM cypher('{graph_name}', $$ {cypher_query} $$) AS ({ag_types});"
        else:
            raise ValueError("No return values specified")

    @staticmethod
    def is_safe_cypher_query(cypher_query: str) -> bool:
        """
        Ensure the Cypher query does not contain dangerous commands.

        Returns:
            bool: True if safe, False otherwise.
        """
        tokens = cypher_query.split()
        unsafe_keywords = ["add", "create", "delete", "merge", "remove", "set"]
        return all(token.lower() not in unsafe_keywords for token in tokens)

    @staticmethod
    def get_return_values(cypher_query: str) -> list:
        parser = CypherParser()
        try:
            result = parser.parse(cypher_query)
        except Exception as e:
            log.error(f"Failed to parse Cypher query: {e}")
            return []

        for op, opr, *_ in result:
            if op == "RETURN" or op == "RETURN_DISTINCT":
                log.debug(f"Returning values from query: {opr}")
                results = []
                for v in opr:
                    if isinstance(v, str):
                        results.append(v.split(".")[0])
                    elif isinstance(v, tuple):
                        match v[0]:
                            case "alias":
                                results.append(v[-1])
                            case "property":
                                results.append(v[-1])
                            case "func_call":
                                results.append(v[1])
                            case "":
                                pass
                return list(set(results))

        return []


class DatabaseManager:
    """Class for managing database connections and queries."""

    def __init__(self, log_level: int = logging.INFO):
        log.setLevel(log_level)
        self.pool: ConnectionPool | None = None  # Initialize pool to None
        self.connection_info: dict = {}

    def connect(self, connection_info: dict) -> None:
        """
        Connect to the database using provided connection information.

        Raises:
            Exception: If the connection fails.
        """
        host = connection_info.get("host", "localhost")
        port = connection_info.get("port", "5432")
        dbname = connection_info.get("dbname", "")
        user = connection_info.get("user", "")
        password = connection_info.get("password", "")

        # Construct the connection string with search_path options.
        conn_str = (
            f"host={host} port={port} dbname={dbname} user={user} password={password} "
            f"options='-c search_path=ag_catalog,\"$user\",public'"
        )

        # Initialize the connection pool and test the connection.
        self.pool = ConnectionPool(conninfo=conn_str)
        assert self.pool is not None
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        self.connection_info = connection_info  # 保存接続情報
        logging.info("Database connection successful.")

    def execute_query(self, graph_name: str, cypher_query: str) -> dict:
        """
        Execute a formatted Cypher query and return nodes and edges.

        Raises:
            ValueError: For unsafe or invalid queries.
            Exception: For other unexpected errors.
        """
        if self.pool is None:
            raise ValueError("No database connection. Please connect first.")

        formatted_query = CypherQueryFormatter.format_query(graph_name, cypher_query)
        logging.info(f"Executing query: {formatted_query}")

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(formatted_query)
                try:
                    results = cur.fetchall()
                except Exception as fetch_error:
                    logging.error(f"Error fetching results: {fetch_error}")
                    results = []

        nodes = []
        edges = []
        # Process each result item to extract nodes and edges.
        for row in results:
            for item in row:
                if item.endswith("::vertex"):
                    try:
                        jsn = json.loads(item.rstrip("::vertex"))
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error for vertex: {e}")
                        continue
                    node = {
                        "id": jsn.get("id"),
                        "label": jsn.get("label"),
                        "properties": jsn.get("properties"),
                    }
                    nodes.append(node)
                elif item.endswith("::edge"):
                    try:
                        jsn = json.loads(item.rstrip("::edge"))
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error for edge: {e}")
                        continue
                    edge = {
                        "id": jsn.get("id"),
                        "label": jsn.get("label"),
                        "source": jsn.get("start_id"),
                        "target": jsn.get("end_id"),
                        "properties": jsn.get("properties"),
                    }
                    edges.append(edge)

        logging.info(f"Query result - Nodes: {nodes}, Edges: {edges}")
        return {"nodes": nodes, "edges": edges}

    def get_graph_info(self) -> list:
        """
        Retrieve information about graphs and their labels from the database.

        Returns:
            list: A list of graph information dictionaries.

        Raises:
            ValueError: If no database connection is established.
            Exception: For other unexpected errors.
        """
        if self.pool is None:
            raise ValueError("No database connection. Please connect first.")

        result: list = []
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # Retrieve the list of graphs.
                cur.execute(
                    "SELECT graphid, name, namespace FROM ag_graph ORDER BY name;"
                )
                graphs = cur.fetchall()
                for graph in graphs:
                    graphid, graph_name, namespace = graph
                    # Retrieve label information for each graph.
                    cur.execute(
                        "SELECT name, kind, relation FROM ag_label WHERE graph = %s;",
                        (graphid,),
                    )
                    labels = cur.fetchall()
                    nodes = []
                    edges = []
                    for label in labels:
                        label_name, kind, relation = label
                        # Skip system labels (starting with '_').
                        if not label_name.startswith("_"):
                            try:
                                count_query = f"SELECT COUNT(*) FROM {relation};"
                                cur.execute(count_query)
                                query_result = cur.fetchone()
                                if query_result:
                                    count = int(query_result[0])
                                else:
                                    count = 0
                            except Exception as count_error:
                                logging.warning(
                                    f"Error counting records for relation {relation}: {count_error}"
                                )
                                count = None

                            if kind == "v":
                                nodes.append(
                                    {
                                        "name": label_name,
                                        "relation": relation,
                                        "count": count,
                                    }
                                )
                            elif kind == "e":
                                edges.append(
                                    {
                                        "name": label_name,
                                        "relation": relation,
                                        "count": count,
                                    }
                                )
                    result.append(
                        {
                            "graphid": graphid,
                            "graph_name": graph_name,
                            "namespace": namespace,
                            "nodes": nodes,
                            "edges": edges,
                        }
                    )
        return result


db_manager = DatabaseManager()


@app.route("/")
def index():
    """Render the main page with default connection info."""
    default_conn_str = os.environ.get("PG_CONNECTION_STRING", "")
    default_conn = (
        ConnectionStringParser.parse(default_conn_str) if default_conn_str else {}
    )
    return render_template("index.html", default_conn=default_conn)


@app.route("/api/connect", methods=["POST"])
def connect_db():
    """API endpoint to connect to the database."""
    data = request.get_json()
    connection_info = data.get("connection", {})
    try:
        db_manager.connect(connection_info)
        return jsonify({"message": "Successfully connected to the database!"})
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/connection_status", methods=["GET"])
def connection_status():
    """API endpoint to return the connection status."""
    status = "connected" if db_manager.pool is not None else "disconnected"
    return jsonify({"status": status, "connection_info": db_manager.connection_info})


@app.route("/api/execute_query", methods=["POST"])
def execute_query_endpoint():
    """API endpoint to execute a Cypher query."""
    data = request.get_json()
    cypher_query = data.get("cypher_query", "")
    graph_name = data.get("graph_name", "")

    if not cypher_query:
        return jsonify({"error": "No Cypher query specified"}), 400

    try:
        result = db_manager.execute_query(graph_name, cypher_query)
        return jsonify(result)
    except ValueError as ve:
        logging.error(f"Query error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Unexpected error during query execution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph_info", methods=["GET"])
def graph_info():
    """API endpoint to retrieve graph information."""
    try:
        result = db_manager.get_graph_info()
        return jsonify(result)
    except ValueError as ve:
        logging.error(f"Graph info error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Unexpected error retrieving graph info: {e}")
        return jsonify({"error": str(e)}), 500
