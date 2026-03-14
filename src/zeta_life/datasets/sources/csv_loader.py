"""CSV data source using only stdlib csv module.

Loads numeric data from CSV files for feeding into consciousness systems.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class CSVSource:
    """Load numeric data from a CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file.
    columns : list[str | int] | None
        Column names (str) or indices (int) to select. None = all columns.
    normalize : bool
        If True, z-score normalize each column after loading.
    skip_header : bool
        If True, treat the first row as a header.
    """

    def __init__(
        self,
        csv_path: str | Path,
        columns: list[str | int] | None = None,
        normalize: bool = True,
        skip_header: bool = True,
    ) -> None:
        self._path = Path(csv_path)
        self._columns = columns
        self._normalize = normalize
        self._skip_header = skip_header
        self._header: list[str] | None = None
        self._data: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"csv_{self._path.stem}"

    @property
    def n_features(self) -> int:
        data = self._ensure_loaded()
        return data.shape[1]

    @property
    def n_samples(self) -> int:
        data = self._ensure_loaded()
        return data.shape[0]

    @property
    def feature_names(self) -> list[str]:
        self._ensure_loaded()
        if self._header is not None:
            return self._header
        return [f"col_{i}" for i in range(self.n_features)]

    def load(self) -> np.ndarray:
        """Load and return data as (n_samples, n_features) array."""
        self._data = None  # force reload
        return self._ensure_loaded()

    def _ensure_loaded(self) -> np.ndarray:
        if self._data is not None:
            return self._data

        rows: list[list[str]] = []
        with open(self._path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header_row = None
            if self._skip_header:
                header_row = next(reader, None)
            for row in reader:
                if row:
                    rows.append(row)

        if not rows:
            raise ValueError(f"No data rows found in {self._path}")

        # Resolve column indices
        col_indices = self._resolve_columns(header_row, len(rows[0]))

        # Extract selected columns and convert to float
        data = []
        for row in rows:
            try:
                data.append([float(row[i]) for i in col_indices])
            except (ValueError, IndexError):
                continue  # skip non-numeric rows

        if not data:
            raise ValueError(f"No valid numeric data in {self._path}")

        self._data = np.array(data, dtype=np.float64)

        # Store selected header names
        if header_row is not None:
            self._header = [header_row[i] for i in col_indices]

        if self._normalize:
            std = self._data.std(axis=0)
            std[std == 0] = 1.0
            self._data = (self._data - self._data.mean(axis=0)) / std

        return self._data

    def _resolve_columns(
        self, header_row: list[str] | None, n_cols: int
    ) -> list[int]:
        if self._columns is None:
            return list(range(n_cols))

        indices: list[int] = []
        for col in self._columns:
            if isinstance(col, int):
                if col < 0 or col >= n_cols:
                    raise ValueError(f"Column index {col} out of range (0-{n_cols-1})")
                indices.append(col)
            elif isinstance(col, str):
                if header_row is None:
                    raise ValueError(
                        f"Cannot select column by name '{col}' without header"
                    )
                try:
                    indices.append(header_row.index(col))
                except ValueError:
                    raise ValueError(
                        f"Column '{col}' not found in header: {header_row}"
                    ) from None
            else:
                raise TypeError(f"Column selector must be str or int, got {type(col)}")
        return indices
