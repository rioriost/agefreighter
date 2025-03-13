#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import tokenize
import ast
import urllib.request
import io
import importlib.resources
import logging
from typing import List, Tuple
from psycopg_pool import ConnectionPool

from agefreighter.cypherparser import CypherParser
from agefreighter.main import CONFIG_DIR

# Import the OpenAI client library after installing openai.
from openai import OpenAI  # type: ignore

# Configure logging (only errors will be printed)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Color definitions for pretty-printing output.
RED = "\033[0;31m"
GREEN = "\033[0;32m"
RESET = "\033[0m"


class QueryExtractor:
    """
    Extracts Gremlin and Cypher queries along with their line numbers from
    source code files. It supports Python source files (using AST or tokenize
    methods) and generic source files (Java, C#, etc.) using regular expressions.
    """

    def __init__(self, code: str, file_path: str, log_level: int = logging.INFO):
        """
        Initialize the extractor with source code and its file path/URL.
        """
        log.setLevel(log_level)
        self.code: str = code
        self.file_path: str = file_path

    def extract(self) -> List[Tuple[int, str]]:
        """
        Determine file type and choose the corresponding extraction method.
        Returns a list of tuples (line_number, query_literal).

        The method now checks for both Gremlin and Cypher indicators.
        """
        queries: List[Tuple[int, str]] = []
        # For Python files, check for library-specific imports or call patterns.
        if self.file_path.endswith(".py"):
            if (
                "from gremlin_python" in self.code
                or "import gremlin_python" in self.code
            ):
                queries.extend(self._extract_gremlin_queries())
            elif "neo4j" in self.code or "session.run" in self.code:
                queries.extend(self._extract_cypher_queries())
            else:
                # No clear indicator; try extracting both types from literals.
                queries.extend(self._extract_gremlin_from_literals())
                queries.extend(self._extract_cypher_from_literals())
        # For files with a .cypher extension, use the generic cypher extractor.
        elif self.file_path.endswith(".cypher"):
            queries.extend(self._extract_generic_cypher_literals())
        else:
            # For other generic source files, extract both Gremlin and Cypher literals.
            queries.extend(self._extract_generic_literals())
            queries.extend(self._extract_generic_cypher_literals())
        return queries

    def _extract_gremlin_from_literals(self) -> List[Tuple[int, str]]:
        """
        For Python files that do not import gremlin_python,
        tokenizes the source and looks for string literals starting with "g.".
        Returns a list of (line_number, literal) tuples.
        """
        results: List[Tuple[int, str]] = []
        try:
            tokens = tokenize.generate_tokens(io.StringIO(self.code).readline)
            for token in tokens:
                if token.type == tokenize.STRING:
                    literal = token.string
                    # Check if the literal (or with opening quote removed) starts with "g."
                    if literal.startswith("g.") or (
                        len(literal) > 1 and literal[1:].startswith("g.")
                    ):
                        results.append(
                            (token.start[0], literal.strip('"').replace('"', "'"))
                        )
        except tokenize.TokenError:
            sys.exit("Error while parsing Python file")

        return results

    def _extract_gremlin_queries(self) -> List[Tuple[int, str]]:
        """
        Uses an AST visitor to locate call nodes that originate from identifier 'g'.
        Returns a list of (line_number, query_string) tuples.
        """
        try:
            tree = ast.parse(self.code)
        except Exception:
            sys.exit("Error while parsing Python file")

        class GremlinQueryVisitor(ast.NodeVisitor):
            def __init__(self):
                self.queries: List[Tuple[int, str]] = []

            def visit_Call(self, node):
                if self._starts_with_g(node.func):
                    try:
                        query_str = ast.unparse(node)
                    except Exception:
                        query_str = self._node_to_str(node)
                    lineno = getattr(node, "lineno", -1)
                    self.queries.append((lineno, query_str))
                self.generic_visit(node)

            def _starts_with_g(self, node):
                """Check recursively if the AST node starts with 'g'."""
                if isinstance(node, ast.Name):
                    return node.id == "g"
                elif isinstance(node, ast.Attribute):
                    return self._starts_with_g(node.value)
                elif isinstance(node, ast.Call):
                    return self._starts_with_g(node.func)
                return False

            def _node_to_str(self, node):
                """Fallback conversion for an AST node."""
                return "<Gremlin query representation>"

        visitor = GremlinQueryVisitor()
        visitor.visit(tree)
        return visitor.queries

    def _extract_generic_literals(self) -> List[Tuple[int, str]]:
        """
        Removes comment blocks from non-Python source code files and
        uses regular expressions to find double-quoted literals starting with "g.".
        Returns a list of (line_number, literal) tuples.
        """
        code_no_comments = re.sub(r"//.*", "", self.code)
        code_no_comments = re.sub(r"/\*.*?\*/", "", code_no_comments, flags=re.DOTALL)

        results: List[Tuple[int, str]] = []
        string_pattern = r'"g\.(?:\\.|[^"\\])*"'
        for match in re.finditer(string_pattern, code_no_comments):
            literal = match.group(0)
            lineno = code_no_comments.count("\n", 0, match.start()) + 1
            results.append((lineno, literal))
        return results

    def _extract_cypher_from_literals(self) -> List[Tuple[int, str]]:
        """
        For Python files without clear Neo4j imports, tokenizes the source
        and looks for string literals starting with a common Cypher keyword.
        Returns a list of (line_number, literal) tuples.
        """
        results: List[Tuple[int, str]] = []
        cypher_keywords = ("MATCH", "CREATE", "MERGE", "CALL", "WITH", "RETURN")
        try:
            tokens = tokenize.generate_tokens(io.StringIO(self.code).readline)
            for token in tokens:
                if token.type == tokenize.STRING:
                    literal = token.string
                    # Remove the quotes and any leading whitespace.
                    stripped_literal = literal.strip("'\"").lstrip()
                    # Check if the literal starts with any of the cypher keywords.
                    for keyword in cypher_keywords:
                        if stripped_literal.upper().startswith(keyword):
                            results.append((token.start[0], literal))
                            break
        except tokenize.TokenError:
            sys.exit("Error while parsing Python file")
        return results

    def _extract_cypher_queries(self) -> List[Tuple[int, str]]:
        """
        Uses an AST visitor to locate call nodes (e.g. session.run(...)) where the first
        argument is a string literal starting with a common Cypher keyword.
        Returns a list of (line_number, query_string) tuples.
        """
        try:
            tree = ast.parse(self.code)
        except Exception:
            sys.exit("Error while parsing Python file")

        class CypherQueryVisitor(ast.NodeVisitor):
            def __init__(self):
                self.queries: List[Tuple[int, str]] = []

            def visit_Call(self, node):
                # Check for calls of the form: some_object.run(query, ...)
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    query_str = node.args[0].value
                    # Check if query_str starts with a common Cypher keyword.
                    if (
                        query_str.strip()
                        .upper()
                        .startswith(
                            ("MATCH", "CREATE", "MERGE", "CALL", "WITH", "RETURN")
                        )
                    ):
                        lineno = getattr(node, "lineno", -1)
                        self.queries.append((lineno, query_str))
                self.generic_visit(node)

        visitor = CypherQueryVisitor()
        visitor.visit(tree)
        return visitor.queries

    def _extract_generic_cypher_literals(self) -> List[Tuple[int, str]]:
        """
        Removes comment blocks from non-Python source code files and uses
        regular expressions to find double-quoted literals starting with a Cypher keyword.
        Returns a list of (line_number, literal) tuples.
        """
        code_no_comments = re.sub(r"//.*", "", self.code)
        code_no_comments = re.sub(r"/\*.*?\*/", "", code_no_comments, flags=re.DOTALL)

        results: List[Tuple[int, str]] = []
        # Regex to match a double-quoted string starting with a common Cypher keyword.
        # Using re.IGNORECASE to match regardless of case.
        string_pattern = r'"(?:MATCH|CREATE|MERGE|CALL|WITH|RETURN)\b(?:\\.|[^"\\])*"'
        for match in re.finditer(string_pattern, code_no_comments, re.IGNORECASE):
            literal = match.group(0)
            lineno = code_no_comments.count("\n", 0, match.start()) + 1
            results.append((lineno, literal))
        return results


class CacheManager:
    """
    Manages caching of Gremlin-to-Cypher conversion results.
    Cache is stored in a JSON file at ~/.converter_cache.
    """

    def __init__(self, log_level: int = logging.INFO):
        log.setLevel(log_level)
        self.cache_file = os.path.join(
            os.path.expanduser("~"), CONFIG_DIR, ".converter_cache"
        )
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """
        Loads the cache from disk.
        If the cache file does not exist, deploy a default one.
        """
        if not os.path.exists(self.cache_file):
            self.deploy_converter_cache()
        try:
            with open(self.cache_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            sys.exit("Error loading cache file: " + str(e))

    def _save_cache(self) -> None:
        """
        Saves the cache dictionary to disk.
        """
        try:
            with open(self.cache_file, "w", encoding="utf-8") as file:
                json.dump(self.cache, file, ensure_ascii=False, indent=4)
        except Exception as e:
            sys.exit("Error saving cache file: " + str(e))

    def add_search_result(self, keyword: str, result: str) -> None:
        """
        Add a new conversion result to the cache.
        """
        self.cache[keyword] = result
        self._save_cache()

    def get_search_result(self, keyword: str) -> str:
        """
        Retrieve a conversion result from the cache.
        Returns None if not found.
        """
        return self.cache.get(keyword, None)

    def deploy_converter_cache(self) -> None:
        """
        Deploy the initial .converter_cache file from packaged data if it exists.
        """
        target_path = self.cache_file
        try:
            # Open the packaged cache file using importlib.resources.
            with importlib.resources.open_binary(
                "agefreighter.packaged_data", "converter_cache"
            ) as resource_file:
                cache_data = resource_file.read()
            with open(target_path, "wb") as f:
                f.write(cache_data)
            print(f".converter_cache was deployed to {target_path}.")
        except FileNotFoundError:
            print(".converter_cache not found in packaged data.")
        except Exception as e:
            print(f"Error deploying cache: {e}")


class GremlinToCypherConverter:
    """
    Uses the OpenAI API to convert a Gremlin query into its equivalent Cypher query.
    """

    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", log_level: int = logging.INFO
    ):
        """
        Initialize the converter by retrieving the API key from the environment.
        """
        log.setLevel(log_level)
        self.api_key = api_key
        # Initialize the OpenAI client with the API key.
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def convert(self, gremlin_query: str) -> str:
        """
        Convert the supplied Gremlin query to a Cypher query.
        Returns the Cypher query as a string.
        """
        # Create a prompt that includes the Gremlin query.
        prompt = f"""Convert the following Gremlin query to an equivalent Cypher query:

{gremlin_query}
"""
        try:
            # Call the OpenAI Chat Completion API.
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that converts Gremlin queries "
                            "into Cypher queries. You reply with only the converted Cypher query."
                            "Never include any additional text or explanations."
                            "Never include carriage returns or line breaks."
                            "Never use 'GROUP BY' in the Cypher query."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=0.0,  # Deterministic conversions.
            )

            logging.debug(f"API response: {chat_completion}")

            # Extract and return the Cypher query.
            cypher_query = chat_completion.choices[0].message.content
            if cypher_query:
                return cypher_query
            else:
                raise ValueError("No Cypher query found in the API response.")
        except Exception as e:
            print(f"Error during conversion: {e}")
            return ""


class DryRunner:
    def __init__(self, dsn: str, graph_name: str, log_level: int = logging.INFO):
        log.setLevel(log_level)
        self.dsn = dsn + " options='-c search_path=ag_catalog,\"$user\",public'"
        self.pool = ConnectionPool(self.dsn)
        self.pool.open()
        self.graph_name = graph_name

    def __del__(self):
        self.pool.close()

    def run(self, query: str) -> str:
        with self.pool.connection() as conn:
            with conn.cursor() as cursor:
                check_query = f"SELECT * FROM ag_graph WHERE name='{self.graph_name}'"
                try:
                    cursor.execute(check_query)
                    if cursor.fetchone():
                        log.error(f"Graph '{self.graph_name}' already exists")
                        sys.exit(1)
                except Exception as e:
                    print(f"Error checking graph: {e}")
                    return ""
                queries = [
                    f"SELECT create_graph('{self.graph_name}');",
                    query,
                    f"SELECT drop_graph('{self.graph_name}', true);",
                ]
                try:
                    cursor.execute("\n".join(queries))
                    return f"{GREEN}[Query executed successfully]{RESET}"
                except Exception as e:
                    return f"{RED}[Error executing query: {e}]{RESET}"


class ConverterController:
    """
    Controller class to coordinate reading input, extracting Gremlin queries,
    converting them to Cypher, and outputting the results.
    """

    def __init__(
        self,
        query_language: str,
        api_key: str,
        model: str,
        dryrun: bool,
        dsn: str,
        graph_name: str,
        query: str,
        filepath: str,
        url: str,
        log_level: int = logging.INFO,
    ):
        """
        Initialize with parsed command line arguments.
        """
        log.setLevel(log_level)
        self.query_language = query_language
        if query_language == "gremlin":
            self.converter = GremlinToCypherConverter(api_key=api_key, model=model)
        elif query_language == "cypher":
            pass
        else:
            raise ValueError(f"Unsupported query language: {query_language}")
        self.dryrun = dryrun
        if self.dryrun:
            self.runner = DryRunner(dsn=dsn, graph_name=graph_name, log_level=log_level)
        self.graph_name = graph_name
        self.query = query
        self.filepath = filepath
        self.url = url
        self.cache_manager = CacheManager()

    def _read_code(self) -> Tuple[str, str]:
        """
        Reads code from a file, URL, or directly from the -g/--gremlin argument.
        Returns a tuple (code, path), where path is the file or URL or "direct query".
        """
        code = ""
        path = ""
        if self.filepath:
            # Read code from the provided file path.
            if os.path.exists(self.filepath):
                path = self.filepath
                try:
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        code = f.read()
                except Exception as e:
                    sys.exit("Failed to read the file: " + str(e))
            else:
                sys.exit(f"File not found: {self.filepath}")
        elif self.url:
            # Read code from the provided URL.
            try:
                with urllib.request.urlopen(self.url) as response:
                    code = response.read().decode("utf-8")
                path = self.url
            except Exception as e:
                sys.exit("Failed to fetch the file: " + str(e))
        elif self.query:
            # Direct conversion of the single Gremlin query.
            path = "direct query"
            # Replace smart quotes with standard double quotes.
            code = self.query.replace("“", '"').replace("”", '"')
        else:
            sys.exit("No valid input provided.")
        return code.strip('"'), path

    def process(self) -> None:
        """
        Main processing method:
          1. Reads the code or query.
          2. Extracts Gremlin queries.
          3. For each query, check cache and then convert if needed.
          4. Prints the converted Cypher queries.
        """
        code, path = self._read_code()

        extracted: List[Tuple[int, str]] = []
        # If input is a file or URL, extract potential queries
        if path != "direct query":
            extractor = QueryExtractor(code, path)
            extracted = extractor.extract()
            if not extracted:
                sys.exit(
                    f"No Gremlin or Cypher queries found in the source code: {path}"
                )
        else:
            # Wrap the supplied query in a list with an artificial line number.
            extracted = [(1, code)]

        age_queries: dict = {}
        # Process each extracted query.
        for lineno, query in extracted:
            # First try retrieving a cached conversion.
            cypher_result = self.cache_manager.get_search_result(query)
            if cypher_result:
                age_queries[f"line {lineno}, {query}"] = self.format_cypher(
                    cypher_result
                )
            else:
                if self.query_language == "gremlin":
                    # Convert the query using OpenAI API.
                    cypher_query = self.converter.convert(query)
                    if cypher_query:
                        self.cache_manager.add_search_result(query, cypher_query)
                        age_queries[f"line {lineno}, {query}"] = self.format_cypher(
                            cypher_query
                        )
                    else:
                        age_queries[f"line {lineno}, {query}"] = self.format_cypher("")
                elif self.query_language == "cypher":
                    age_queries[f"line {lineno}, {query}"] = self.format_cypher(query)

        # Print all conversion results.
        print("Converted Cypher queries for Apache AGE:\n")
        for src, age_query in age_queries.items():
            print(f"{src} ->\n{age_query}\n")
            if self.dryrun:
                query_result = self.runner.run(age_query)
                print(query_result)

    def format_cypher(self, cypher_query: str) -> str:
        """
        Format a Cypher query
        If self.args.age is True, format the query for Apache AGE.
        """
        logging.debug(cypher_query)
        # Failed to convert
        if not cypher_query:
            return f"{RED}[Failed]{RESET}"

        returns = self.get_return_values(cypher_query)
        matches = re.findall(r"\$(\w+)", cypher_query)
        stored_procedure = ""
        parameter = ""
        execution = ""
        if matches:  # create stored procedure
            stored_procedure = (
                "DEALLOCATE ALL; PREPARE cypher_stored_procedure(agtype) AS "
            )
            parameter = ", $1"
            execution = (
                "EXECUTE cypher_stored_procedure('{"
                + ", ".join([f'"{match}": 12345' for match in matches])
                + "}');"
            )
        if returns:
            ag_types = ", ".join([f"{r} agtype" for r in returns])
            age_query = f"{stored_procedure}SELECT * FROM cypher('{self.graph_name}', $$ {cypher_query} $${parameter}) AS ({ag_types});{execution}"
        else:
            age_query = (
                f"SELECT * FROM cypher('{self.graph_name}', $$ {cypher_query} $$);"
            )

        return age_query

    @staticmethod
    def get_return_values(cypher_query: str) -> list:
        parser = CypherParser()
        try:
            result = parser.parse(cypher_query)
        except Exception as e:
            logging.error(f"Failed to parse Cypher query: {cypher_query}")
            logging.error(f"Error: {e}")
            return []

        if result:
            for op, opr, *_ in result:
                if op == "RETURN":
                    return opr

        return []
