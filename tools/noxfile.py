import nox

nox.options.pythons = ["3.13"]
nox.options.default_venv_backend = "uv"


@nox.session(python=["3.13"], tags=["clean"])
def clean(session):
    with session.chdir("../"):
        session.run(
            "uv",
            "run",
            "rm",
            "-rf",
            ".nox",
            ".coverage*",
            ".pytest_cache",
            ".mypy_cache",
            "__pycache__",
            "tests/__pycache__",
            "src/agefreighter/__pycache__",
            "src/agefreighter/parsetab.py",
            "dist",
        )


@nox.session(python=["3.13"], tags=["dist"])
def dist(session):
    with session.chdir("../"):
        session.run("uv", "build")


@nox.session(python=["3.13"], tags=["docker"])
def docker(session):
    with session.chdir("../"):
        session.run("uv", "run", "tools/build_docker.py")


@nox.session(python=["3.13"], tags=["lint"])
def lint(session):
    with session.chdir("../"):
        session.install("ruff")
        session.run("uv", "run", "ruff", "check", "src")
        session.run("uv", "run", "ruff", "format", "src")


@nox.session(python=["3.13"], tags=["mypy"])
def mypy(session):
    with session.chdir("../"):
        session.install(".")
        session.install(
            "mypy", "types-aiofiles", "pandas-stubs", "ply", "faker", "neo4j", "shtab"
        )
        session.run("uv", "run", "mypy", "src")


@nox.session(python=["3.13"], tags=["pytest"])
def pytest(session):
    with session.chdir("../"):
        session.install(".")
        session.install(
            "pytest",
            "pytest-cov",
            "neo4j",
            "azure-cosmos",
            "openai",
            "pandas",
            "ply",
            "faker",
            "flask",
            "gremlinpython",
            "shtab",
        )
        test_files = [
            "tests/test_agefreighter.py",
            "tests/test_converter.py",
            "tests/test_cosmosgremlinloader.py",
            "tests/test_cosmosnosqlexporter.py",
            "tests/test_csvdatamanager.py",
            "tests/test_csvexporter.py",
            "tests/test_generator.py",
            "tests/test_main.py",
            "tests/test_neo4jexporter.py",
            "tests/test_neo4jloader.py",
            "tests/test_pgsqlexporter.py",
            "tests/test_view.py",
        ]
        session.run(
            "uv",
            "run",
            "pytest",
            "--maxfail=1",
            "--cov=agefreighter",
            "--cov-report=html",
            *test_files,
        )


@nox.session(python=["3.13"], tags=["post_clean"])
def post_clean(session):
    with session.chdir("../"):
        session.run(
            "uv",
            "run",
            "rm",
            "-rf",
            "dummy_config.json",
            ".nox",
            ".coverage*",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "__pycache__",
            "tests/__pycache__",
            "src/agefreighter/__pycache__",
            "src/agefreighter/parsetab.py",
            "tab_replaced*",
        )
