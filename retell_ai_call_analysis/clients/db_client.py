import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


class SQLiteClient:
    """
    A client for interacting with SQLite databases.
    Optimized for handling large datasets with connection pooling and efficient data operations.
    """

    def __init__(self, db_path: str | Path):
        """
        Initialize the SQLite client with the database path.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self):
        """
        Context manager for database connections.
        Automatically handles connection creation and cleanup.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable foreign keys support
            conn.execute("PRAGMA foreign_keys = ON")
            # For better performance with large transactions
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
        finally:
            conn.close()

    def execute_query(
        self, query: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a query and return the results as a list of dictionaries.

        Args:
            query: SQL query to execute
            params: Parameters to substitute in the query

        Returns:
            List of dictionaries representing the query results
        """
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return [dict(row) for row in cursor.fetchall()]

    def execute_write(self, query: str, params: tuple | None = None) -> int:
        """
        Execute a write operation (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query to execute
            params: Parameters to substitute in the query

        Returns:
            Number of rows affected
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            conn.commit()
            return cursor.rowcount

    def execute_many(self, query: str, params_list: list[tuple]) -> int:
        """
        Execute a write operation with multiple parameter sets.
        Optimized for bulk operations.

        Args:
            query: SQL query to execute
            params_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount

    def create_table(
        self,
        table_name: str,
        columns: dict[str, str],
        primary_key: str | None = None,
        if_not_exists: bool = True,
    ) -> None:
        """
        Create a new table in the database.

        Args:
            table_name: Name of the table to create
            columns: Dictionary mapping column names to their SQL types
            primary_key: Name of the primary key column (if any)
            if_not_exists: Whether to add IF NOT EXISTS to the query
        """
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""

        column_defs = []
        for col_name, col_type in columns.items():
            if primary_key and col_name == primary_key:
                column_defs.append(f"{col_name} {col_type} PRIMARY KEY")
            else:
                column_defs.append(f"{col_name} {col_type}")

        query = f"CREATE TABLE {exists_clause}{table_name} ({', '.join(column_defs)})"

        with self.connection() as conn:
            conn.execute(query)
            conn.commit()

    def insert_data(self, table_name: str, data: dict[str, Any]) -> int:
        """
        Insert a single row of data into a table.

        Args:
            table_name: Name of the table
            data: Dictionary mapping column names to values

        Returns:
            ID of the inserted row
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()))
            conn.commit()
            return cursor.lastrowid

    def insert_many(self, table_name: str, data_list: list[dict[str, Any]]) -> int:
        """
        Insert multiple rows of data into a table.

        Args:
            table_name: Name of the table
            data_list: List of dictionaries mapping column names to values

        Returns:
            Number of rows inserted
        """
        if not data_list:
            return 0

        # Ensure all dictionaries have the same keys
        columns = data_list[0].keys()
        placeholders = ", ".join(["?"] * len(columns))
        query = (
            f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        )

        # Convert list of dicts to list of tuples
        values = [tuple(d.values()) for d in data_list]

        return self.execute_many(query, values)

    def read_to_pandas(self, query: str, params: tuple | None = None) -> pd.DataFrame:
        """
        Execute a query and return the results as a pandas DataFrame.
        Optimized for large data processing.

        Args:
            query: SQL query to execute
            params: Parameters to substitute in the query

        Returns:
            pandas DataFrame containing the query results
        """
        with self.connection() as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            return pd.read_sql_query(query, conn)

    def write_pandas_to_table(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"
    ) -> None:
        """
        Write a pandas DataFrame to a database table.

        Args:
            df: pandas DataFrame to write
            table_name: Name of the target table
            if_exists: How to behave if the table exists ('fail', 'replace', or 'append')
        """
        with self.connection() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if the table exists, False otherwise
        """
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (table_name,))
            return cursor.fetchone() is not None
