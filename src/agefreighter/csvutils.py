from typing import Any, Mapping

DEFAULT_CSV_DELIMITER = ","


def normalize_csv_delimiter(delimiter: Any) -> str:
    """Return a delimiter accepted by CSV readers."""
    if not isinstance(delimiter, str):
        raise ValueError("CSV delimiter must be a string")
    if delimiter == r"\t":
        delimiter = "\t"
    if len(delimiter) != 1:
        raise ValueError("CSV delimiter must be a single character")
    if delimiter in {'"', "\r", "\n"}:
        raise ValueError("CSV delimiter cannot be a line break or double quote")
    return delimiter


def get_csv_delimiter(config: Mapping[str, Any]) -> str:
    """Read and validate an optional delimiter from a CSV config object."""
    return normalize_csv_delimiter(config.get("delimiter", DEFAULT_CSV_DELIMITER))
