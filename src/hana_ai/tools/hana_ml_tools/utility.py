"""
Utility functions for the HANA ML tools.
"""
import os
import shutil
import json
import re
from pathlib import Path
import logging
from datetime import datetime, date
from typing import Optional, Union, Any
from pandas import Timestamp
from numpy import int64
from hana_ml.model_storage import ModelStorage
#pylint: disable=too-many-nested-blocks, unexpected-keyword-arg, invalid-name

logger = logging.getLogger(__name__)

def convert_cap_to_hdi(source_dir, target_dir, archive=True):
    """
    Convert a CAP project structure to an HDI structure.
    Parameters
    ----------
    source_dir : str
        The source directory containing the CAP project files.
    target_dir : str
        The target directory where the HDI structure will be created.
    archive : bool, optional
        If True, the function will create an archive of the source directory.
        Default is True.
    """
    target_path = Path(target_dir)
    if target_path.exists() and target_path.is_dir():
        if any(target_path.iterdir()):
            if archive:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                archive_path = f"archive_{target_dir}_{timestamp}.tar.gz"
                shutil.make_archive(archive_path, 'gztar', target_dir)
                # delete the target directory after archiving including subdirectories except the archive
                for item in target_path.iterdir():
                    if item.name != f"{target_dir}.tar.gz":
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                logger.info("Created archive: %s", archive_path)
            else:
                logger.info("Target directory %s already exists and is not empty.", target_dir)
                raise FileExistsError(f"The target_dir {target_dir} is not empty. Please provide an empty directory.")
    db_src = os.path.join(Path(target_dir), "db", "src")
    db_cfg = os.path.join(Path(target_dir), "db", "cfg")
    srv_dir = os.path.join(Path(target_dir), "srv")
    os.makedirs(db_src, exist_ok=True)
    os.makedirs(db_cfg, exist_ok=True)
    os.makedirs(srv_dir, exist_ok=True)
    cap_db = Path(os.path.join(Path(source_dir), "db"))
    src_files = Path(os.path.join(cap_db, "src")).glob("*")
    for file in src_files:
        if file.suffix == ".cds":
            target_file = os.path.join(db_src, f"{file.stem}.hdbcds")
            shutil.copy2(file, target_file)
        else:
            shutil.copy2(file, os.path.join(db_src, file.name))
    for cds_file in cap_db.glob("*.cds"):
        target_file = os.path.join(db_src, f"{cds_file.stem}.hdbcds")
        shutil.copy2(cds_file, target_file)
    srv_source = Path(os.path.join(Path(source_dir), "srv"))
    if srv_source.exists():
        shutil.copytree(srv_source, srv_dir, dirs_exist_ok=True)
    hdi_config = os.path.join(db_cfg, ".hdiconfig")
    with open(hdi_config, "w") as f:
        json.dump({
            "file": {
                "path": os.path.join("db", "src"),
                "build_plugins": [
                    {"plugin": "com.sap.hana.di.cds"},
                    {"plugin": "com.sap.hana.di.procedure"},
                    {"plugin": "com.sap.hana.di.synonym"},
                    {"plugin": "com.sap.hana.di.grant"}
                ]
            }
        }, f, indent=2)

class _CustomEncoder(json.JSONEncoder):
    """
    This class is used to encode the model attributes into JSON string.
    """
    def default(self, obj): #pylint: disable=arguments-renamed
        if isinstance(obj, (Timestamp, datetime, date)):
            # Convert Timestamp, datetime or date to ISO string
            return obj.isoformat()
        elif isinstance(obj, (int64, int)):
            # Convert numpy int64 or Python int to Python int
            return int(obj)
        # Let other types use the default handler
        return super().default(obj)


def add_stopping_hint(x : str):
    """Added the hint for stopping the execution when an error message is returned."""
    return (x + ". Please stop the execution and return.").replace("..", ".")


def _hana_safe_identifier(text: Any) -> str:
    """Normalize a segment used to build a HANA table identifier.

    HANA folds unquoted identifiers to upper case at parse time, but the
    ``smart_save``/``save`` helpers in ``hana_ml`` quote the target table name.
    That means when a user-supplied fragment like ``my_hana_ai_model`` is
    embedded verbatim into a table identifier, the table is created
    case-sensitively (``..._my_hana_ai_model_8``) yet any downstream
    ``SELECT ... FROM PREDICT_RESULT_..._my_hana_ai_model_8`` written without
    quotes is folded to ``..._MY_HANA_AI_MODEL_8`` and no longer matches.

    Uppercasing every fragment before assembling the identifier keeps the
    stored table name aligned with HANA's default folding so that downstream
    unquoted references (issued by the agent, tools, or SQL written by the
    user) resolve without needing to double-quote the name. Already-uppercase
    inputs are idempotent under this transform.
    """
    return str(text).upper() if text is not None else text

def generate_model_storage_version(ms : ModelStorage, version: Union[int, str, None], name: str) -> int:
    """Generate the model storage version."""
    ms._create_metadata_table()
    if version is None:
        version = ms._get_new_version_no(name)
        if version is None:
            version = 1
        else:
            version = int(version)
    return version

def _create_temp_table(conn, select_statement: str, tool_name: str, additional_info: str = None) -> str:
    """
    Create a temporary table in the HANA database.
    Parameters
    ----------
    conn : Connection
        The HANA connection object.
    select_statement : str
        The SQL select statement to create the temporary table.
    tool_name : str
        The name of the tool to create a unique temporary table name.
    additional_info : str, optional
        Additional information to append to the table name.
    Returns
    -------
    str
        The SQL statement to select from the temporary table.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    if additional_info:
        additional_info = f"_{additional_info}_"
    else:
        additional_info = "_"
    table_name = f"#{tool_name}{additional_info}{timestamp}".upper()
    create_temp_table_sql = f"CREATE LOCAL TEMPORARY TABLE {table_name} AS ({select_statement})"
    conn.execute_sql(create_temp_table_sql)
    return f"SELECT * FROM {table_name}"


def normalize_column_list(columns: Union[None, str, list, tuple]) -> list[str]:
    """Normalize optional column input into a flat ordered list of column names."""
    if columns is None:
        return []
    if isinstance(columns, str):
        parts = [part.strip() for part in re.split(r"[;,]", columns) if part.strip()]
        return parts if parts else [columns.strip()]
    normalized: list[str] = []
    for column in columns:
        text = str(column).strip()
        if text:
            normalized.append(text)
    return normalized


def is_predict_feature_mismatch_error(exc: Exception) -> bool:
    """Detect PAL/HANA errors that indicate predict-table features do not match the trained model."""
    text = str(exc).lower()
    markers = (
        "feature number of predict table does not match the trained model",
        "predict table features do not match the trained model",
        "predict table does not match the trained model",
        "invalid table:$tab$",
        "73001007",
    )
    return any(marker in text for marker in markers)


def build_repaired_predict_dataframe(predict_df, *, key: str, exog=None, group_key: Optional[str] = None, add_placeholder: bool = False):
    """Return a predict DataFrame containing only the columns required for inference.

    The returned tuple is (dataframe, kept_columns, missing_columns).
    """
    required_columns: list[str] = []
    for column in [group_key, key, *normalize_column_list(exog)]:
        if column and column not in required_columns:
            required_columns.append(column)

    missing_columns = [column for column in required_columns if column not in predict_df.columns]
    if missing_columns:
        return predict_df, required_columns, missing_columns

    repaired_df = predict_df.select(*required_columns)
    if add_placeholder and len(repaired_df.columns) == 1:
        repaired_df = repaired_df.add_constant("PLACEHOLDER", 0)
    return repaired_df, required_columns, []


def format_predict_mismatch_diagnostic(*, predict_table: str, predict_schema: Optional[str], original_columns: list[str], kept_columns: list[str], missing_columns: list[str], key: str, exog=None, group_key: Optional[str] = None, original_error: Optional[str] = None) -> str:
    """Build a structured error payload for predict-table schema mismatches."""
    context_columns = [column for column in [group_key, key, *normalize_column_list(exog)] if column]
    analysis = (
        "The predict table structure does not match the trained model. "
        "For forecasting prediction, the predict input should usually contain only the time key"
        + (", the group key" if group_key else "")
        + (", and any explicit exogenous columns." if context_columns else ".")
    )
    payload = {
        "error": "Prediction table features do not match the trained model.",
        "error_category": "predict_table_feature_mismatch",
        "input_predict_table": predict_table,
        "input_predict_schema": predict_schema,
        "predict_table_columns": original_columns,
        "columns_required_for_retry": kept_columns,
        "missing_required_columns": missing_columns,
        "analysis": analysis,
        "suggested_fix": "Create or use a predict table that contains only the required columns and retry the prediction.",
    }
    if original_error:
        payload["original_error"] = original_error
    return json.dumps(payload, cls=_CustomEncoder)
