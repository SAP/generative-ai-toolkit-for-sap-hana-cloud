"""Dataset preparation tools for conversational data ingestion workflows."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Type

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml import dataframe as hana_dataframe

from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder

logger = logging.getLogger(__name__)


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _qualified_table_name(table_name: str, schema_name: Optional[str] = None) -> str:
    if schema_name:
        return f"{_quote_identifier(schema_name)}.{_quote_identifier(table_name)}"
    return _quote_identifier(table_name)


def _default_output_table(source_table: str, suffix: str, explicit_name: Optional[str] = None) -> str:
    return explicit_name or f"{source_table}_{suffix}"


def _validate_split_request(
    *,
    split_mode: str,
    order_by: Optional[str],
) -> Optional[str]:
    if split_mode != "time_ordered":
        return (
            "Error: only split_mode=time_ordered is currently supported. "
            "This toolkit currently focuses on forecasting workflows, so dataset splits must preserve chronology."
        )
    if split_mode == "time_ordered" and not order_by:
        return "Error: order_by is required for time_ordered splits"
    return None


class ImportCSVToTableInput(BaseModel):
    """Input schema for importing a CSV file into a HANA table."""

    csv_path: str = Field(description="The local CSV file path. It must be accessible to the running Python process.")
    table_name: str = Field(description="The target HANA table name.")
    schema_name: Optional[str] = Field(default=None, description="Optional schema name for the target table.")
    force: bool = Field(default=False, description="Whether to overwrite the target table if it already exists.")
    delimiter: str = Field(default=",", description="Field delimiter used by the CSV file.")
    encoding: str = Field(default="utf-8", description="File encoding used when reading the CSV file.")
    has_header: bool = Field(default=True, description="Whether the first CSV row contains column names.")
    date_columns: Optional[List[str]] = Field(
        default=None,
        description="Optional list of columns to parse as datetime values before uploading to HANA.",
    )
    date_format: Optional[str] = Field(
        default=None,
        description="Optional datetime parsing format applied to date_columns when provided.",
    )
    preview_rows: int = Field(default=5, description="Number of preview rows to include in the tool response.")


class SplitTableForForecastingInput(BaseModel):
    """Input schema for splitting a forecasting table into train/test/validation tables."""

    source_table: str = Field(description="The source HANA table name to split.")
    schema_name: Optional[str] = Field(default=None, description="Optional schema name of the source table.")
    split_mode: str = Field(
        default="time_ordered",
        description=(
            "Split strategy. Currently only time_ordered is supported because the toolkit focuses on forecasting "
            "workflows where chronology must be preserved."
        ),
    )
    train_ratio: float = Field(default=0.7, description="Fraction of rows to place into the training set.")
    test_ratio: float = Field(default=0.2, description="Fraction of rows to place into the test set.")
    validation_ratio: float = Field(default=0.1, description="Fraction of rows to place into the validation set.")
    order_by: Optional[str] = Field(
        default=None,
        description=(
            "Ordering column used for time_ordered splits. Required for time_ordered mode and recommended for "
            "time series workflows so train, test, and validation tables preserve temporal order."
        ),
    )
    train_table: Optional[str] = Field(default=None, description="Optional output table name for the training split.")
    test_table: Optional[str] = Field(default=None, description="Optional output table name for the test split.")
    validation_table: Optional[str] = Field(
        default=None,
        description="Optional output table name for the validation split.",
    )
    force: bool = Field(default=False, description="Whether to overwrite existing output tables.")


class ImportCSVToTableTool(BaseTool):
    """Import a local CSV file into a HANA table."""

    name: str = "import_csv_to_table"
    description: str = (
        "Import a local CSV file into a HANA table, with optional datetime parsing for selected columns."
    )
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = ImportCSVToTableInput
    return_direct: bool = False

    def __init__(self, connection_context: ConnectionContext, return_direct: bool = False) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]

        csv_path = Path(str(kwargs.get("csv_path") or "")).expanduser().resolve()
        table_name = kwargs.get("table_name")
        schema_name = kwargs.get("schema_name")
        force = bool(kwargs.get("force", False))
        delimiter = kwargs.get("delimiter", ",")
        encoding = kwargs.get("encoding", "utf-8")
        has_header = bool(kwargs.get("has_header", True))
        date_columns = kwargs.get("date_columns") or []
        date_format = kwargs.get("date_format")
        preview_rows = max(0, int(kwargs.get("preview_rows", 5)))

        if not table_name:
            return "Error: table_name is required"
        if not csv_path.exists() or not csv_path.is_file():
            return f"Error: csv_path does not exist or is not a file: {csv_path}"

        header = 0 if has_header else None
        try:
            pdf = pd.read_csv(csv_path, sep=delimiter, encoding=encoding, header=header)
        except Exception as exc:
            logger.exception("Failed to read CSV file")
            return f"Operation failed while reading CSV: {exc}"

        if not has_header:
            pdf.columns = [f"COLUMN_{idx + 1}" for idx in range(len(pdf.columns))]

        missing_date_columns = [col for col in date_columns if col not in pdf.columns]
        if missing_date_columns:
            return f"Error: date_columns not found in CSV: {', '.join(missing_date_columns)}"

        try:
            for column in date_columns:
                pdf[column] = pd.to_datetime(pdf[column], format=date_format, errors="raise")

            hana_dataframe.create_dataframe_from_pandas(
                connection_context=self.connection_context,
                pandas_df=pdf,
                table_name=table_name,
                schema=schema_name,
                force=force,
                drop_exist_tab=force,
                disable_progressbar=True,
            )
        except Exception as exc:
            logger.exception("Failed to import CSV to HANA table")
            return f"Operation failed while importing CSV to HANA: {exc}"

        summary = {
            "csv_path": str(csv_path),
            "table_name": table_name,
            "schema_name": schema_name,
            "rows_imported": int(len(pdf.index)),
            "columns": [str(col) for col in pdf.columns],
            "date_columns": list(date_columns),
            "preview": pdf.head(preview_rows).to_dict(orient="records") if preview_rows > 0 else [],
        }
        return json.dumps(summary, cls=_CustomEncoder, ensure_ascii=False, indent=2)

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class SplitTableForForecastingTool(BaseTool):
    """Split a source HANA table into time-ordered train/test/validation tables for forecasting."""

    name: str = "split_table_for_forecasting"
    description: str = (
        "Split a HANA table into train/test/validation tables for forecasting using time-ordered logic. "
        "The current toolkit scope is forecasting, so chronology is preserved by design."
    )
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = SplitTableForForecastingInput
    return_direct: bool = False

    def __init__(self, connection_context: ConnectionContext, return_direct: bool = False) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]

        source_table = kwargs.get("source_table")
        schema_name = kwargs.get("schema_name")
        split_mode = str(kwargs.get("split_mode") or "time_ordered").strip().lower()
        train_ratio = float(kwargs.get("train_ratio", 0.7))
        test_ratio = float(kwargs.get("test_ratio", 0.2))
        validation_ratio = float(kwargs.get("validation_ratio", 0.1))
        order_by = kwargs.get("order_by")
        force = bool(kwargs.get("force", False))

        if not source_table:
            return "Error: source_table is required"

        ratios = [train_ratio, test_ratio, validation_ratio]
        if any(ratio < 0 for ratio in ratios):
            return "Error: train_ratio, test_ratio, and validation_ratio must be non-negative"
        if abs(sum(ratios) - 1.0) > 1e-9:
            return "Error: train_ratio + test_ratio + validation_ratio must equal 1.0"

        validation_error = _validate_split_request(
            split_mode=split_mode,
            order_by=order_by,
        )
        if validation_error:
            return validation_error

        train_table = _default_output_table(source_table, "TRAIN", kwargs.get("train_table"))
        test_table = _default_output_table(source_table, "TEST", kwargs.get("test_table"))
        validation_table = _default_output_table(source_table, "VALIDATION", kwargs.get("validation_table"))

        try:
            source_df = self.connection_context.table(source_table, schema=schema_name)
        except Exception as exc:
            logger.exception("Failed to access source table")
            return f"Operation failed while loading source table: {exc}"

        try:
            summary = self._run_time_ordered_split(
                source_df=source_df,
                source_table=source_table,
                schema_name=schema_name,
                order_by=order_by,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                train_table=train_table,
                test_table=test_table,
                validation_table=validation_table,
                force=force,
            )
        except Exception as exc:
            logger.exception("Failed to split source table")
            return f"Operation failed while splitting the source table: {exc}"

        return json.dumps(summary, cls=_CustomEncoder, ensure_ascii=False, indent=2)

    def _run_time_ordered_split(
        self,
        *,
        source_df,
        source_table: str,
        schema_name: Optional[str],
        order_by: str,
        train_ratio: float,
        test_ratio: float,
        train_table: str,
        test_table: str,
        validation_table: str,
        force: bool,
    ) -> dict:
        columns = list(source_df.columns)
        selected_columns = ", ".join(_quote_identifier(column) for column in columns)
        projected_columns = ", ".join(f'base.{_quote_identifier(column)}' for column in columns)
        source_name = _qualified_table_name(source_table, schema_name)
        order_expr = _quote_identifier(order_by)

        train_boundary = train_ratio
        test_boundary = train_ratio + test_ratio

        train_sql = f"""
WITH base AS (
    SELECT
        {selected_columns},
        ROW_NUMBER() OVER (ORDER BY {order_expr}) AS "__ROW_NUM",
        COUNT(*) OVER () AS "__TOTAL_ROWS"
    FROM {source_name}
)
SELECT {projected_columns}
FROM base
WHERE "__ROW_NUM" <= FLOOR("__TOTAL_ROWS" * {train_boundary})
""".strip()
        test_sql = f"""
WITH base AS (
    SELECT
        {selected_columns},
        ROW_NUMBER() OVER (ORDER BY {order_expr}) AS "__ROW_NUM",
        COUNT(*) OVER () AS "__TOTAL_ROWS"
    FROM {source_name}
)
SELECT {projected_columns}
FROM base
WHERE "__ROW_NUM" > FLOOR("__TOTAL_ROWS" * {train_boundary})
  AND "__ROW_NUM" <= FLOOR("__TOTAL_ROWS" * {test_boundary})
""".strip()
        validation_sql = f"""
WITH base AS (
    SELECT
        {selected_columns},
        ROW_NUMBER() OVER (ORDER BY {order_expr}) AS "__ROW_NUM",
        COUNT(*) OVER () AS "__TOTAL_ROWS"
    FROM {source_name}
)
SELECT {projected_columns}
FROM base
WHERE "__ROW_NUM" > FLOOR("__TOTAL_ROWS" * {test_boundary})
""".strip()

        self.connection_context.sql(train_sql).smart_save(train_table, schema=schema_name, force=force)
        self.connection_context.sql(test_sql).smart_save(test_table, schema=schema_name, force=force)
        self.connection_context.sql(validation_sql).smart_save(validation_table, schema=schema_name, force=force)

        return {
            "source_table": source_table,
            "schema_name": schema_name,
            "split_mode": "time_ordered",
            "split_guidance": "Time-ordered split used so the forecasting workflow preserves chronology.",
            "order_by": order_by,
            "ratios": {
                "train": train_ratio,
                "test": test_ratio,
                "validation": 1.0 - test_boundary,
            },
            "output_tables": {
                "train": train_table,
                "test": test_table,
                "validation": validation_table,
            },
            "row_counts": {
                "train": int(self.connection_context.table(train_table, schema=schema_name).count()),
                "test": int(self.connection_context.table(test_table, schema=schema_name).count()),
                "validation": int(self.connection_context.table(validation_table, schema=schema_name).count()),
            },
        }

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class SplitTableForModelingTool(SplitTableForForecastingTool):
    """Deprecated compatibility alias for the forecasting split tool."""

    name: str = "split_table_for_modeling"
    description: str = (
        "Deprecated alias for split_table_for_forecasting. "
        "Split a HANA table into train/test/validation tables for forecasting using time-ordered logic."
    )
