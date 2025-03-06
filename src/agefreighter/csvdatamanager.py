#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Generator

import pandas as pd

# Configure logging; default to INFO (overridable by the --debug flag)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CsvDataManager:
    """
    Manage CSV file operations, including loading a DataFrame and chunking.
    """

    def __init__(
        self,
        data_dir: str,
        base_file: str,
        log_level: int = logging.INFO,
    ) -> None:
        log.setLevel(log_level)
        if data_dir:
            self.csv_file = os.path.abspath(os.path.join(data_dir, base_file))
        log.info("CsvDataManager initialized")

    def get_dataframe(self) -> pd.DataFrame:
        """
        Read the CSV file and return a DataFrame.
        """
        return pd.read_csv(self.csv_file)

    @staticmethod
    def get_chunks(
        df: pd.DataFrame, chunk_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Yield chunks of the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to be chunked.
            chunk_size (int): Number of rows per chunk.

        Yields:
            Generator[pd.DataFrame, None, None]: Chunks of the DataFrame.
        """
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i : i + chunk_size].copy()
