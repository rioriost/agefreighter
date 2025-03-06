#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import argcomplete
import asyncio
import importlib.util
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Optional

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def show_completion_instructions():
    print(
        "\nTo enable shell completion, add the following to your shell configuration:"
    )
    print("\n- bash (~/.bashrc):")
    print('    eval "$(register-python-argcomplete agefreighter)"')
    print("\n- zsh (~/.zshrc, ~/.zprofile):")
    print("    autoload -U compinit && compinit")
    print('    eval "$(register-python-argcomplete agefreighter)"\n')


def check_first_run():
    home = os.path.expanduser("~")
    marker_file = os.path.join(home, ".agefreighter_first_run")
    if not os.path.exists(marker_file):
        show_completion_instructions()
        try:
            with open(marker_file, "w") as f:
                f.write("agefreighter has been executed.")
        except Exception as e:
            log.error(f"Failed to create first run marker file: {e}")


def check_and_install(package_name: str, module_name: Optional[str] = None):
    """
    package_name: name to install via pip or uv
    module_name: name to import (defaults to package_name)
    """
    module_name = module_name or package_name
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None

    if spec is not None:
        log.debug(f"Module '{module_name}' is already available.")
        return

    answer = (
        input(f"required module '{module_name}' is not installed. Install it? [Y/n]: ")
        .strip()
        .lower()
    )
    if answer not in ("", "y", "yes"):
        print(f"'{package_name}' installation was denied. Exiting subcommand.")
        sys.exit(1)

    # Try installing with pip first
    install_cmd = [sys.executable, "-m", "pip", "install", package_name]
    try:
        subprocess.run(install_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        if "No module named pip" in (e.stderr or ""):
            print("pip module is not available. Trying with uv...")
            # Fallback: try installing with uv
            alt_install_cmd = ["uv", "add", "--dev", package_name]
            try:
                subprocess.run(alt_install_cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to install '{package_name}' using uv.")
                sys.exit(1)
        else:
            print(f"Failed to install '{package_name}'.")
            sys.exit(1)


def run_flask(flask_port: int, log_level: int = logging.INFO):
    from agefreighter.view import app

    app.logger.setLevel(log_level)
    # Run the Flask app on the desired port
    app.run(port=flask_port)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="AGEFreighter, a tool to export data from various sources and load it into Apache AGE."
    )
    # PostgreSQL connection arguments
    parser.add_argument(
        "--graphname",
        type=str,
        default="FROM_NEO4J",
        help="Name of a new graph on Apache AGE",
    )
    parser.add_argument(
        "--pg-con-str",
        type=str,
        default=os.environ.get("PG_CONNECTION_STRING", ""),
        help="Connection string of the Azure Database for PostgreSQL",
    )
    parser.add_argument(
        "--pg-min-connections",
        type=int,
        default=4,
        help="Minimum number of connections to PostgreSQL",
    )
    parser.add_argument(
        "--pg-max-connections",
        type=int,
        default=64,
        help="Maximum number of connections to PostgreSQL",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="subparser", required=True)

    # "completion" subcommand
    subparsers.add_parser("completion", help="Show Completion Instructions")

    # "load" subcommand
    parser_load = subparsers.add_parser("load", help="Load data into Apache AGE")
    parser_load.add_argument(
        "--source-type",
        type=str,
        choices=["neo4j", "cosmosdb", "pgsql", "csv"],
        default="neo4j",
        help="Source type of the graph data",
    )
    parser_load.add_argument(
        "--trial",
        action="store_true",
        default=False,
        help="Extract only 100 edges per relationship type",
    )
    parser_load.add_argument(
        "--save-temps",
        action="store_true",
        default=False,
        help="Save data from source as CSV files into a directory",
    )
    parser_load.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for exporting data",
    )
    parser_load.add_argument(
        "--progress", action="store_true", default=True, help="Show progress"
    )
    # CSV-specific arguments
    parser_load.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to the configuration file",
    )
    # Neo4j-specific arguments
    parser_load.add_argument(
        "--neo4j-uri",
        type=str,
        default=os.environ.get("NEO4J_URI", ""),
        help="Neo4j URI",
    )
    parser_load.add_argument(
        "--neo4j-user",
        type=str,
        default=os.environ.get("NEO4J_USER", ""),
        help="Neo4j username",
    )
    parser_load.add_argument(
        "--neo4j-password",
        type=str,
        default=os.environ.get("NEO4J_PASSWORD", ""),
        help="Neo4j password",
    )
    parser_load.add_argument(
        "--neo4j-database",
        type=str,
        default="",
        help="Neo4j database",
    )
    # Cosmos-specific arguments
    parser_load.add_argument(
        "--cosmos-endpoint",
        type=str,
        default=os.environ.get("COSMOS_ENDPOINT", ""),
        help="Cosmos endpoint",
    )
    parser_load.add_argument(
        "--cosmos-key",
        type=str,
        default=os.environ.get("COSMOS_KEY", ""),
        help="Cosmos key",
    )
    parser_load.add_argument(
        "--cosmos-database",
        type=str,
        default=os.environ.get("COSMOS_DATABASE", ""),
        help="Cosmos database",
    )
    parser_load.add_argument(
        "--cosmos-container",
        type=str,
        default=os.environ.get("COSMOS_CONTAINER", ""),
        help="Cosmos container",
    )
    parser_load.add_argument(
        "--src-pg-con-str",
        type=str,
        default=os.environ.get("SRC_PG_CONNECTION_STRING", ""),
        help="Source PostgreSQL connection string",
    )

    # "view" subcommand
    parser_view = subparsers.add_parser("view", help="View data in Apache AGE")
    parser_view.add_argument(
        "--flask-port",
        type=int,
        default="5000",
        help="Port to run the server on",
    )

    # "parse" subcommand
    parser_parse = subparsers.add_parser("parse", help="Parse a cypher query")
    parser_parse.add_argument(
        "cypher_query",
        type=str,
        default="MATCH (n) RETURN n",
        help="Cypher query to be parsed",
    )

    # "generate" subcommand
    parser_generate = subparsers.add_parser("generate", help="Generate dummy data")
    parser_generate.add_argument(
        "--pattern-no",
        type=int,
        default=1,
        help="Pattern number to generate",
    )

    # "convert" subcommand
    parser_convert = subparsers.add_parser(
        "convert", help="Convert Gremlin queries to Cypher queries."
    )
    parser_convert.add_argument(
        "-k",
        "--openai-api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key to use.",
    )
    parser_convert.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use.",
    )
    parser_convert.add_argument(
        "-d",
        "--dryrun",
        action="store_true",
        default=False,
        help="Dry run with PostgreSQL.",
    )
    parser_convert.add_argument(
        "--pg-con-str-for-dryrun",
        type=str,
        default=os.environ.get("PG_CONNECTION_STRING", ""),
        help="Connection string of the Azure Database for PostgreSQL",
    )
    parser_convert.add_argument(
        "--graph-for-dryrun",
        type=str,
        default="GRAPH_FOR_DRYRUN",
        help="Graph name for dry run with PostgreSQL.",
    )
    group_convert = parser_convert.add_mutually_exclusive_group(required=True)
    group_convert.add_argument(
        "-g", "--gremlin", type=str, help="The Gremlin query to convert."
    )
    group_convert.add_argument(
        "-f",
        "--filepath",
        type=str,
        help="Path to the source code file (.py, .java, .cs, .txt)",
    )
    group_convert.add_argument(
        "-u",
        "--url",
        type=str,
        help="URL to the source code file (.py, .java, .cs, .txt)",
    )

    # "prepare" subcommand
    parser_prepare = subparsers.add_parser(
        "prepare", help="Prepare data for testing AGEFreighter."
    )
    parser_prepare.add_argument(
        "--target-type",
        type=str,
        choices=["neo4j", "cosmosdb", "pgsql"],
        default="neo4j",
        help="Targeted type of the source of graph data.",
    )
    parser_prepare.add_argument(
        "--data-dir",
        type=str,
        default="docs",
        help="Directory containing the data files",
    )
    parser_prepare.add_argument(
        "--base-file",
        type=str,
        default="customer_product_bought.csv",
        help="Base file name for the data files",
    )
    # Neo4j-specific arguments
    parser_prepare.add_argument(
        "--neo4j-uri",
        type=str,
        default=os.environ.get("NEO4J_URI", ""),
        help="Neo4j URI",
    )
    parser_prepare.add_argument(
        "--neo4j-user",
        type=str,
        default=os.environ.get("NEO4J_USER", ""),
        help="Neo4j username",
    )
    parser_prepare.add_argument(
        "--neo4j-password",
        type=str,
        default=os.environ.get("NEO4J_PASSWORD", ""),
        help="Neo4j password",
    )
    parser_prepare.add_argument(
        "--neo4j-database",
        type=str,
        default="",
        help="Neo4j database",
    )
    # Cosmos-specific arguments
    parser_prepare.add_argument(
        "--cosmos-gremlin-endpoint",
        type=str,
        default=os.environ.get("COSMOS_GREMLIN_ENDPOINT", ""),
        help="Cosmos Gremlinendpoint",
    )
    parser_prepare.add_argument(
        "--cosmos-key",
        type=str,
        default=os.environ.get("COSMOS_KEY", ""),
        help="Cosmos key",
    )
    parser_prepare.add_argument(
        "--cosmos-database",
        type=str,
        default=os.environ.get("COSMOS_DATABASE", ""),
        help="Cosmos database",
    )
    parser_prepare.add_argument(
        "--cosmos-container",
        type=str,
        default=os.environ.get("COSMOS_CONTAINER", ""),
        help="Cosmos container",
    )
    parser_prepare.add_argument(
        "--src-pg-con-str",
        type=str,
        default=os.environ.get("SRC_PG_CONNECTION_STRING", ""),
        help="Source PostgreSQL connection string",
    )
    argcomplete.autocomplete(parser)
    return parser.parse_args()


async def async_main() -> None:
    """
    Main async function to export from Neo4j and load data into AGE.
    """
    args = parse_arguments()

    # Adjust logging level based on the debug flag
    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled.")
    else:
        log.setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)

    if args.pg_min_connections > args.pg_max_connections:
        log.error("min_connections cannot be greater than max_connections")
        sys.exit(1)
    if not args.pg_con_str:
        log.error(
            "PG_CONNECTION_STRING not set. Set via environment variable or argument."
        )
        sys.exit(1)

    match args.subparser:
        case "completion":
            show_completion_instructions()
            sys.exit(0)
        case "load":
            match args.source_type:
                case "neo4j":
                    if not (args.neo4j_uri and args.neo4j_user and args.neo4j_password):
                        log.error(
                            "NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD not set. Set via environment variable or argument."
                        )
                        sys.exit(1)

                    check_and_install("neo4j")
                    from agefreighter.neo4jexporter import Neo4jExporter

                    try:
                        async with Neo4jExporter(
                            dsn=args.pg_con_str,
                            min_connections=args.pg_min_connections,
                            max_connections=args.pg_max_connections,
                            uri=args.neo4j_uri,
                            user=args.neo4j_user,
                            password=args.neo4j_password,
                            database=args.neo4j_database,
                            trial=args.trial,
                            save_temps=args.save_temps,
                            progress=args.progress,
                            graph_name=args.graphname,
                            chunk_size=args.chunk_size,
                            log_level=logging.DEBUG if args.debug else logging.INFO,
                        ) as exporter:
                            await exporter.export()
                            await exporter.copy()
                    except Exception as e:
                        log.error("An error occurred during export: %s", e)
                        sys.exit(1)
                case "csv":
                    if not args.config:
                        log.error(
                            "Config file is required for CSV export. See samples in configs directory."
                        )
                        sys.exit(1)
                    if not os.path.exists(os.path.abspath(args.config)):
                        log.error("Config file '%s' does not exist.", args.config)
                        sys.exit(1)

                    from agefreighter.csvexporter import CSVExporter

                    try:
                        async with CSVExporter(
                            dsn=args.pg_con_str,
                            min_connections=args.pg_min_connections,
                            max_connections=args.pg_max_connections,
                            config=os.path.abspath(args.config),
                            trial=args.trial,
                            save_temps=args.save_temps,
                            progress=args.progress,
                            graph_name=args.graphname,
                            chunk_size=args.chunk_size,
                            log_level=logging.DEBUG if args.debug else logging.INFO,
                        ) as exporter:
                            await exporter.export()
                            await exporter.copy()
                    except Exception as e:
                        log.error("An error occurred during export: %s", e)
                        sys.exit(1)
                case "cosmosdb":
                    if not (
                        args.cosmos_endpoint
                        and args.cosmos_key
                        and args.cosmos_database
                        and args.cosmos_container
                    ):
                        log.error(
                            "COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, and COSMOS_CONTAINER not set. Set via environment variable or argument."
                        )
                        sys.exit(1)

                    check_and_install("azure-cosmos", "azure.cosmos")
                    from agefreighter.cosmosnosqlexporter import CosmosNoSQLExporter

                    try:
                        async with CosmosNoSQLExporter(
                            dsn=args.pg_con_str,
                            min_connections=args.pg_min_connections,
                            max_connections=args.pg_max_connections,
                            cosmos_endpoint=args.cosmos_endpoint,
                            cosmos_key=args.cosmos_key,
                            cosmos_database=args.cosmos_database,
                            cosmos_container=args.cosmos_container,
                            trial=args.trial,
                            save_temps=args.save_temps,
                            progress=args.progress,
                            graph_name=args.graphname,
                            chunk_size=args.chunk_size,
                            log_level=logging.DEBUG if args.debug else logging.INFO,
                        ) as exporter:
                            await exporter.export()
                            await exporter.copy()
                    except Exception as e:
                        log.error("An error occurred during export: %s", e)
                        sys.exit(1)
                case "pgsql":
                    raise NotImplementedError(
                        "PostgreSQL export is not yet implemented"
                    )
                    if not (args.src_pg_con_str):
                        log.error(
                            "SRC_PG_CON_STR not set. Set via environment variable or argument."
                        )
                        sys.exit(1)

                    from agefreighter.pgsqlexporter import PGSQLExporter  # type: ignore

                    try:
                        async with PGSQLExporter(
                            dsn=args.pg_con_str,
                            min_connections=args.pg_min_connections,
                            max_connections=args.pg_max_connections,
                            src_dsn=args.src_pg_con_str,
                            trial=args.trial,
                            save_temps=args.save_temps,
                            progress=args.progress,
                            graph_name=args.graphname,
                            chunk_size=args.chunk_size,
                            log_level=logging.DEBUG if args.debug else logging.INFO,
                        ) as exporter:
                            await exporter.export()
                            await exporter.copy()
                    except Exception as e:
                        log.error("An error occurred during export: %s", e)
                        sys.exit(1)
                case _:
                    log.error("Source type '%s' is not implemented.", args.source_type)
                    sys.exit(1)

        case "view":
            check_and_install("flask")

            flask_thread = threading.Thread(
                target=run_flask,
                args=(
                    args.flask_port,
                    logging.DEBUG if args.debug else logging.INFO,
                ),
            )
            flask_thread.daemon = True  # Ensure thread exits when main program exits
            flask_thread.start()

            try:
                # Keep the main thread alive so the Flask app keeps running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Received interrupt. Exiting...")

        case "parse":
            if not args.cypher_query:
                log.error("Cypher query not provided.")
                sys.exit(1)

            check_and_install("ply")
            from agefreighter.cypherparser import CypherParser

            parser = CypherParser(
                log_level=logging.DEBUG if args.debug else logging.INFO,
            )
            print(parser.parse(args.cypher_query))

        case "generate":
            check_and_install("faker")
            import agefreighter.generator as generator

            await generator.main(
                pattern_no=args.pattern_no,
                log_level=logging.DEBUG if args.debug else logging.INFO,
            )

        case "convert":
            if not args.openai_api_key:
                log.error(
                    "OPENAI_API_KEY not set. Set via environment variable or argument."
                )
                sys.exit(1)
            check_and_install("openai")
            from agefreighter.g2c import GremlinConverterController

            controller = GremlinConverterController(
                api_key=args.openai_api_key,
                model=args.model,
                dryrun=args.dryrun,
                dsn=args.pg_con_str_for_dryrun,
                graph_name=args.graph_for_dryrun,
                gremlin_query=args.gremlin,
                filepath=args.filepath,
                url=args.url,
                log_level=logging.DEBUG if args.debug else logging.INFO,
            )
            controller.process()

        case "prepare":
            check_and_install("pandas")
            from agefreighter.csvdatamanager import CsvDataManager

            csv_manager = CsvDataManager(
                data_dir=args.data_dir,
                base_file=args.base_file,
                log_level=logging.DEBUG if args.debug else logging.INFO,
            )  # Create a CSV data manager instance
            match args.target_type:
                case "neo4j":
                    if not (args.neo4j_uri and args.neo4j_user and args.neo4j_password):
                        log.error(
                            "NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD not set. Set via environment variable or argument."
                        )
                        sys.exit(1)
                    check_and_install("neo4j")
                    from agefreighter.neo4jloader import Neo4jLoader  # type: ignore

                    neo4j_loader = Neo4jLoader(
                        csv_manager=csv_manager,
                        neo4j_uri=args.neo4j_uri,
                        neo4j_user=args.neo4j_user,
                        neo4j_password=args.neo4j_password,
                        log_level=logging.DEBUG if args.debug else logging.INFO,
                    )
                    await neo4j_loader.load_data()
                case "pgsql":
                    from agefreighter.pgsqlloader import PgsqlLoader  # type: ignore

                    pgsql_loader = PgsqlLoader(
                        csv_manager=csv_manager,
                        src_dsn=args.src_pg_con_str,
                        log_level=logging.DEBUG if args.debug else logging.INFO,
                    )
                    await pgsql_loader.load_data()
                case "cosmosdb":
                    if not (
                        args.cosmos_gremlin_endpoint
                        and args.cosmos_key
                        and args.cosmos_database
                        and args.cosmos_container
                    ):
                        log.error(
                            "COSMOS_GREMLIN_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, and COSMOS_CONTAINER not set. Set via environment variable or argument."
                        )
                        sys.exit(1)

                    check_and_install("gremlinpython")
                    from agefreighter.cosmosgremlinloader import CosmosGremlinLoader  # type: ignore

                    cosmos_loader = CosmosGremlinLoader(
                        csv_manager=csv_manager,
                        cosmos_gremlin_endpoint=args.cosmos_gremlin_endpoint,
                        cosmos_key=args.cosmos_key,
                        cosmos_database=args.cosmos_database,
                        cosmos_container=args.cosmos_container,
                        log_level=logging.DEBUG if args.debug else logging.INFO,
                    )
                    await cosmos_loader.load_data()
                case _:
                    log.error("No valid target type provided.")
                    sys.exit(1)
        case _:
            log.error("No valid subcommand provided.")
            sys.exit(1)


def main() -> None:
    """
    Entry point: Set the appropriate event loop policy and run the async_main function.
    """
    check_first_run()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(async_main())
    except asyncio.CancelledError:
        # Clean up if necessary
        pass
    except KeyboardInterrupt:
        log.error("KeyboardInterrupt received. Shutting down.")
        sys.exit(1)
    except Exception as e:
        log.error("An unhandled exception occurred: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
