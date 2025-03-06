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
    Extracts Gremlin queries along with their line numbers from
    source code files. It supports Python source files (using AST
    or tokenize methods) and generic source files (Java, C#, etc.)
    using regular expressions.
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
        """
        # For Python files, decide which extraction method to use.
        if self.file_path.endswith(".py"):
            if (
                "from gremlin_python" in self.code
                or "import gremlin_python" in self.code
            ):
                return self._extract_gremlin_queries()
            else:
                return self._extract_gremlin_from_literals()
        else:
            return self._extract_generic_literals()

    def _extract_gremlin_from_literals(self) -> List[Tuple[int, str]]:
        """
        For Python files that do not import gremlin_python,
        tokenizes the source and looks for string literals starting with "g.".
        Returns a list of (line_number, literal) tuples.
        """
        results: List[Tuple[int, str]] = []
        try:
            # Convert the code string to bytes for tokenize.
            tokens = tokenize.generate_tokens(io.StringIO(self.code).readline)
            for token in tokens:
                if token.type == tokenize.STRING:
                    literal = token.string
                    # Check if the literal (or with opening quote removed) starts with "g."
                    if literal.startswith("g.") or (
                        len(literal) > 1 and literal[1:].startswith("g.")
                    ):
                        # token.start returns (line, column); we want the first element.
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
                        # Try to convert the AST node back into actual source code.
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
        # Remove single-line and multi-line comments.
        code_no_comments = re.sub(r"//.*", "", self.code)
        code_no_comments = re.sub(r"/\*.*?\*/", "", code_no_comments, flags=re.DOTALL)

        results: List[Tuple[int, str]] = []
        # RegEx to match a double-quoted string starting with "g." (handles escaped quotes)
        string_pattern = r'"g\.(?:\\.|[^"\\])*"'
        for match in re.finditer(string_pattern, code_no_comments):
            literal = match.group(0)
            # Calculate the line number based on newline count before the match.
            lineno = code_no_comments.count("\n", 0, match.start()) + 1
            results.append((lineno, literal))
        return results


class CacheManager:
    """
    Manages caching of Gremlin-to-Cypher conversion results.
    Cache is stored in a JSON file at ~/.g2c_cache.
    """

    def __init__(self, log_level: int = logging.INFO):
        log.setLevel(log_level)
        self.cache_file = os.path.join(os.path.expanduser("~"), ".g2c_cache")
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """
        Loads the cache from disk.
        If the cache file does not exist, deploy a default one.
        """
        if not os.path.exists(self.cache_file):
            self.deploy_g2c_cache()
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

    def deploy_g2c_cache(self) -> None:
        """
        Deploy the initial .g2c_cache file from packaged data if it exists.
        """
        target_path = self.cache_file
        try:
            # Open the packaged cache file using importlib.resources.
            with importlib.resources.open_binary(
                "agefreighter.packaged_data", "g2c_cache"
            ) as resource_file:
                cache_data = resource_file.read()
            with open(target_path, "wb") as f:
                f.write(cache_data)
            print(f".g2c_cache was deployed to {target_path}.")
        except FileNotFoundError:
            print(".g2c_cache not found in packaged data.")
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


class GremlinConverterController:
    """
    Controller class to coordinate reading input, extracting Gremlin queries,
    converting them to Cypher, and outputting the results.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        dryrun: bool,
        dsn: str,
        graph_name: str,
        gremlin_query: str,
        filepath: str,
        url: str,
        log_level: int = logging.INFO,
    ):
        """
        Initialize with parsed command line arguments.
        """
        log.setLevel(log_level)
        self.converter = GremlinToCypherConverter(api_key=api_key, model=model)
        self.dryrun = dryrun
        if self.dryrun:
            self.runner = DryRunner(dsn=dsn, graph_name=graph_name, log_level=log_level)
        self.graph_name = graph_name
        self.gremlin_query = gremlin_query
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
        elif self.gremlin_query:
            # Direct conversion of the single Gremlin query.
            path = "direct query"
            # Replace smart quotes with standard double quotes.
            code = self.gremlin_query.replace("“", '"').replace("”", '"')
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
                sys.exit(f"No Gremlin queries found in the source code: {path}")
        else:
            # Wrap the supplied query in a list with an artificial line number.
            extracted = [(1, code)]

        cypher_queries: dict = {}
        # Process each extracted query.
        for lineno, query in extracted:
            # First try retrieving a cached conversion.
            cypher_result = self.cache_manager.get_search_result(query)
            if cypher_result:
                cypher_queries[f"line {lineno}, {query}"] = self.format_cypher(
                    cypher_result
                )
            else:
                # Convert the query using OpenAI API.
                cypher_query = self.converter.convert(query)
                if cypher_query:
                    # Add to cache.
                    self.cache_manager.add_search_result(query, cypher_query)
                    cypher_queries[f"line {lineno}, {query}"] = self.format_cypher(
                        cypher_query
                    )
                else:
                    cypher_queries[f"line {lineno}, {query}"] = self.format_cypher("")

        # Print all conversion results.
        print("Converted Cypher queries:\n")
        for src, cypher_query in cypher_queries.items():
            print(f"{src} ->\n{cypher_query}\n")

    def format_cypher(self, cypher_query: str) -> str:
        """
        Format a Cypher query
        If self.args.age is True, format the query for Apache AGE.
        """
        # Failed to conver
        if not cypher_query:
            return f"{RED}[Failed]{RESET}"

        cypher_for_age = self.format_for_age(cypher_query)
        if not self.dryrun:
            return f"{cypher_for_age}"

        # for dry run
        if self.dryrun:
            res = self.runner.run(cypher_for_age)
            return f"{cypher_for_age}\n{res}"

        # standard cypher query
        return f"{cypher_query}"

    def format_for_age(self, cypher_query: str) -> str:
        logging.debug(cypher_query)
        returns = self.extract_return_values(cypher_query)
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
            return f"{stored_procedure}SELECT * FROM cypher('{self.graph_name}', $$ {cypher_query} $${parameter}) AS ({ag_types});{execution}"
        else:
            return f"SELECT * FROM cypher('{self.graph_name}', $$ {cypher_query} $$);"

    @staticmethod
    def extract_return_values(cypher_query: str) -> list:
        match = (
            re.search(r"(?i)(?<=\breturn\b)(.*)$", cypher_query)
            or re.search(r"(?i)(?<=\bdelete\b)(.*)$", cypher_query)
            or re.search(r"(?i)(?<=\bset\b)(.*)$", cypher_query)
        )
        return_parts: list = []
        if match:
            return_parts = [x.strip() for x in match.group(1).strip().split(",")]
        return_values: list = []
        pattern = re.compile(r"([A-Za-z0-9_]\w*)\s*(?=\()")  # extract function name
        num_pattern = re.compile(r"^[0-9\.]+$")
        for return_part in return_parts:
            tokens = return_part.split()
            next_is_alias = False
            return_value = ""
            for token in tokens:
                if token.lower() == "as":
                    next_is_alias = True
                    continue
                elif token.lower() == "distinct":
                    next_is_alias = True
                    continue
                elif token.lower() == "order":
                    break
                elif token.lower() == "group":  # Cypher doesn't support GROUP BY
                    break
                elif token.lower() == "by":
                    break
                elif token.lower() == "desc":
                    break
                elif token.lower() == "asc":
                    break
                elif token.lower() == "limit":
                    break
                elif token.lower() == "skip":
                    break
                elif token == "=":
                    break
                else:
                    return_value = token
                if next_is_alias:
                    return_value = token
                    break
            match = pattern.match(return_value)  # extract function name
            if match:
                return_value = match.group(1)
            match = num_pattern.match(return_value)
            if not match:
                return_value = return_value.split(".")[0]
            if not re.search(r"[(){}]", return_value):
                return_values.append(return_value)
        return return_values
