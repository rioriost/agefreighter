import nox

nox.options.python = "3.13"
nox.options.default_venv_backend = "uv"


@nox.session(python=["3.13"], tags=["lint"])
def lint(session):
    session.install("ruff")
    session.run("uv", "run", "ruff", "check", "src")
    session.run("uv", "run", "ruff", "format", "src")


@nox.session(python=["3.13"], tags=["mypy"])
def mypy(session):
    session.install(".")
    session.install(
        "mypy",
        "types-aiofiles",
        "pandas-stubs",
        "ply",
        "faker",
        "neo4j",
    )
    session.run("uv", "run", "mypy", "src")


@nox.session(python=["3.13"], tags=["pytest"])
def pytest(session):
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
    )
    test_files = [
        "tests/test_agefreighter.py",
        "tests/test_cosmosgremlinloader.py",
        "tests/test_cosmosnosqlexporter.py",
        "tests/test_csvdatamanager.py",
        "tests/test_csvexporter.py",
        "tests/test_g2c.py",
        "tests/test_generator.py",
        "tests/test_main.py",
        "tests/test_neo4jexporter.py",
        "tests/test_neo4jloader.py",
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
