import nox

nox.options.python = "3.9"
nox.options.default_venv_backend = "uv"


@nox.session(python=["3.9"], tags=["lint"])
def lint(session):
    session.install("ruff")
    session.run("uv", "run", "ruff", "check")
    session.run("uv", "run", "ruff", "format")


@nox.session(python=["3.9"], tags=["mypy"])
def mypy(session):
    session.install(".")
    session.install("mypy", "pandas-stubs", "pyarrow-stubs", "types-networkx")
    session.run("uv", "run", "mypy", "src")


# @nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"], tags=["pytest"])
@nox.session(python=["3.9"], tags=["pytest-agefreighter"])
def pytest_agefreighter(session):
    session.install(".")
    session.install("pytest", "pytest-cov")
    test_files = ["tests/test_agefreighter.py"]
    session.run(
        "uv",
        "run",
        "pytest",
        "--maxfail=1",
        "--cov=agefreighter",
        "--cov-report=term",
        *test_files,
    )


@nox.session(python=["3.9"], tags=["pytest-csvfreighter"])
def pytest_csvfreighter(session):
    session.install(".")
    session.install("pytest", "pytest-cov")
    test_files = ["tests/test_csvfreighter.py"]
    session.run(
        "uv",
        "run",
        "pytest",
        "--maxfail=1",
        "--cov=agefreighter",
        "--cov-report=term",
        *test_files,
    )


@nox.session(python=["3.9"], tags=["pytest-multicsvfreighter"])
def pytest_multicsvfreighter(session):
    session.install(".")
    session.install("pytest", "pytest-cov")
    test_files = ["tests/test_multicsvfreighter.py"]
    session.run(
        "uv",
        "run",
        "pytest",
        "--maxfail=1",
        "--cov=agefreighter",
        "--cov-report=term",
        *test_files,
    )
