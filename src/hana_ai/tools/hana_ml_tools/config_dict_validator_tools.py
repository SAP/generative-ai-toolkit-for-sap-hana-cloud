"""
This module exposes the SAP HANA PAL AutoML **pipeline catalogue** and
**config_dict** SQL procedures as LangChain tools.

Instead of maintaining a hand-authored schema on the client, the tools call
the PAL procedures directly so the source of truth stays inside HANA:

* :class:`GetPALPipelineInfo` — wraps ``_SYS_AFL.PAL_PIPELINE_INFO`` and
  returns the catalogue of operators, categories and parameters supported by
  the current HANA instance.
* :class:`GetAutoMLConfigDict` — wraps ``_SYS_AFL.PAL_AUTOML_CONFIG`` in
  *read* mode. Given a ``pipeline_type`` and a ``config_dict`` payload
  (``'default'`` / ``'light'`` / ``'empty'`` template or a JSON object), it
  returns the resolved ``config_dict`` as JSON along with the per-operator
  ``INFO`` rows.
* :class:`ModifyAutoMLConfigDict` — wraps ``_SYS_AFL.PAL_AUTOML_CONFIG`` in
  *modify* mode. It accepts the optional ``CONFIG_REMOVE`` / ``CONFIG_ADD``
  / ``CONFIG_REPLACE`` / ``CONFIG_MODIFY`` rows plus ``VERIFY_CONFIG`` and
  returns the resulting ``config_dict``. When ``verify=True`` (the default)
  PAL raises on invalid entries and the tool surfaces the error verbatim.

Reference: SAP Help Portal — `Pipeline Operator
<https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-predictive-analysis-library/pipeline-operator-pipeline-operator>`_.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from hdbcli import dbapi

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.pal_base import PALBase
from hana_ml.algorithms.pal.sqlgen import ParameterTable
from hana_ml.ml_base import try_drop
from hana_ml.algorithms.pal.auto_ml import get_pipeline_info

from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PAL_AUTOML_CONFIG accepts these template keywords in the CONFIG_DICT slot.
# 'light' is only supported for classifier / regressor pipelines; PAL raises
# on an unsupported combination — we do not shadow that check on the client.
_TEMPLATE_KEYWORDS = {"default", "light", "empty"}

# Valid pipeline types accepted by PAL_AUTOML_CONFIG's PIPELINE_TYPE row.
_PIPELINE_TYPES = {"classifier", "regressor", "timeseries"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_config_payload(payload: Any) -> Optional[str]:
    """Coerce a ``config_dict``-ish payload into the string PAL expects.

    * ``None`` returns ``None`` (caller skips the parameter row).
    * A template keyword (``default`` / ``light`` / ``empty``) passes through,
      normalised to lower-case.
    * A ``dict`` is JSON-encoded.
    * Any other string is treated as a pre-formatted JSON payload and returned
      unchanged.
    """
    if payload is None:
        return None
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.lower() in _TEMPLATE_KEYWORDS:
            return stripped.lower()
        return stripped
    if isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False)
    # Fall back to json.dumps so callers get a helpful HANA error rather than
    # a Python TypeError deep in PAL.
    return json.dumps(payload, ensure_ascii=False, default=str)


def _call_pal_automl_config(
    connection_context: ConnectionContext,
    *,
    pipeline_type: str,
    config_dict: Any = None,
    config_remove: Any = None,
    config_add: Any = None,
    config_replace: Any = None,
    config_modify: Any = None,
    verify: bool = False,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Invoke ``_SYS_AFL.PAL_AUTOML_CONFIG`` and return the resolved config.

    Parameters
    ----------
    connection_context : ConnectionContext
        Live HANA connection.
    pipeline_type : str
        One of ``'classifier'``, ``'regressor'``, ``'timeseries'``.
    config_dict, config_remove, config_add, config_replace, config_modify : any, optional
        Values for the corresponding PAL parameter rows. ``dict`` inputs are
        JSON-encoded; template strings (``'default'`` / ``'light'`` /
        ``'empty'``) pass through unchanged; ``None`` values skip the row.
    verify : bool, optional
        When ``True`` emit ``VERIFY_CONFIG=1`` so PAL raises on invalid
        entries. When ``False`` (default) PAL best-effort ignores unknown
        keys.

    Returns
    -------
    (result_json, info_rows, error)
        * ``result_json`` — the resolved ``config_dict`` as a JSON string
          (chunks reassembled in ``ROW_INDEX`` order). Empty string on error.
        * ``info_rows`` — list of ``{"operator", "type", "config"}`` dicts
          extracted from the ``INFO`` output table. Empty list on error.
        * ``error`` — ``None`` on success; PAL's error message otherwise.

    Notes
    -----
    Row order matches the SAP Help spec: ``PIPELINE_TYPE, CONFIG_DICT,
    CONFIG_REMOVE, CONFIG_ADD, CONFIG_REPLACE, CONFIG_MODIFY, VERIFY_CONFIG``.
    PAL applies the modifiers in that same order.
    """
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    outputs = [
        '#PAL_AUTOML_CONFIG_RESULT_{}'.format(unique_id),
        '#PAL_AUTOML_CONFIG_INFO_{}'.format(unique_id),
    ]

    param_rows: List[Tuple[str, Optional[int], Optional[float], Optional[str]]] = []
    param_rows.append(('PIPELINE_TYPE', None, None, pipeline_type))

    for row_name, value in (
        ('CONFIG_DICT', config_dict),
        ('CONFIG_REMOVE', config_remove),
        ('CONFIG_ADD', config_add),
        ('CONFIG_REPLACE', config_replace),
        ('CONFIG_MODIFY', config_modify),
    ):
        encoded = _encode_config_payload(value)
        if encoded is not None:
            param_rows.append((row_name, None, None, encoded))

    if verify:
        param_rows.append(('VERIFY_CONFIG', 1, None, None))

    try:
        PALBase()._call_pal_auto(
            connection_context,
            'PAL_AUTOML_CONFIG',
            ParameterTable().with_data(param_rows),
            *outputs,
        )
    except dbapi.Error as db_err:
        logger.error(str(db_err))
        try_drop(connection_context, outputs)
        return "", [], str(db_err)
    except Exception as exc:  # pragma: no cover - safety net
        logger.error(str(exc))
        try_drop(connection_context, outputs)
        return "", [], str(exc)

    try:
        result_df = connection_context.table(outputs[0]).collect()
        info_df = connection_context.table(outputs[1]).collect()
    finally:
        try_drop(connection_context, outputs)

    # RESULT chunks are already assembled in ROW_INDEX order; concatenate the
    # CONTENT column to rebuild the full JSON payload (each row up to 5000
    # chars per SAP Help spec).
    if not result_df.empty:
        content_col = result_df.columns[1]
        index_col = result_df.columns[0]
        result_json = "".join(
            str(x) for x in result_df.sort_values(by=index_col)[content_col].tolist()
        )
    else:
        result_json = ""

    info_rows: List[Dict[str, Any]] = []
    if not info_df.empty:
        cols = list(info_df.columns)
        # Expect NAME, TYPE, CONFIG columns.
        name_col = cols[0]
        type_col = cols[1] if len(cols) > 1 else None
        config_col = cols[2] if len(cols) > 2 else None
        for _, row in info_df.iterrows():
            info_rows.append({
                "operator": row[name_col],
                "type": row[type_col] if type_col else None,
                "config": row[config_col] if config_col else None,
            })

    return result_json, info_rows, None


def _validate_pipeline_type(pipeline_type: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Normalise / validate the ``pipeline_type`` argument.

    Returns ``(normalised, error)``. ``error`` is a user-facing message when
    the input is missing or not one of ``_PIPELINE_TYPES``.
    """
    if not pipeline_type:
        return None, "pipeline_type is required (one of {}).".format(sorted(_PIPELINE_TYPES))
    normalised = str(pipeline_type).strip().lower()
    if normalised not in _PIPELINE_TYPES:
        return None, (
            "pipeline_type '{}' is not supported. "
            "Expected one of {}.".format(pipeline_type, sorted(_PIPELINE_TYPES))
        )
    return normalised, None


def _parse_config_json(raw: str) -> Any:
    """Best-effort JSON parse; return the raw string if it is not JSON."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


# ---------------------------------------------------------------------------
# Tool 1 — GetPALPipelineInfo
# ---------------------------------------------------------------------------

class GetPALPipelineInfoInput(BaseModel):
    """
    Input schema for :class:`GetPALPipelineInfo`.
    """
    operator: Optional[str] = Field(
        description="Only return the entry for this operator name (case-insensitive).",
        default=None,
    )
    category: Optional[str] = Field(
        description="Only return operators whose CATEGORY column matches this value.",
        default=None,
    )
    include_parameters: Optional[bool] = Field(
        description="Include the PARAMETER column (defaults to True).",
        default=True,
    )


class GetPALPipelineInfo(BaseTool):
    """
    Query ``_SYS_AFL.PAL_PIPELINE_INFO`` for the catalogue of operators
    supported by the current HANA instance.

    The response mirrors the PAL_PIPELINE_INFO output — one row per operator
    with its name, category and parameter descriptor. Optional filters narrow
    the payload to a single operator or category, useful when the agent only
    needs to know the parameters for one operator before calling
    :class:`ModifyAutoMLConfigDict`.

    Returns
    -------
    str
        A JSON string with:

        - ``operators`` : list of ``{"NAME", "CATEGORY", ...}`` rows.
        - ``count`` : number of rows returned.
        - ``error`` (optional): populated when ``PAL_PIPELINE_INFO`` is not
          available in the target HANA instance.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 20 50
                :header-rows: 1

                * - Field
                  - Description
                * - operator
                  - Optional operator-name filter.
                * - category
                  - Optional category filter (e.g. ``'Transformer'``, ``'Regressor'``).
                * - include_parameters
                  - Include the PARAMETER column. Defaults to True.
    """
    name: str = "get_pal_pipeline_info"
    """Name of the tool."""
    description: str = (
        "Fetch the catalogue of AutoML pipeline operators supported by this "
        "HANA instance via PAL_PIPELINE_INFO. Returns operator names, "
        "categories and their parameter descriptors. Call this to discover "
        "which operators are available before building a config_dict with "
        "modify_automl_config_dict."
    )
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = GetPALPipelineInfoInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        """Use the tool."""
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]

        operator = kwargs.get("operator")
        category = kwargs.get("category")
        include_parameters = kwargs.get("include_parameters")
        if include_parameters is None:
            include_parameters = True

        info_df = get_pipeline_info(self.connection_context)
        if info_df is False:
            return json.dumps(
                {"error": "PAL_PIPELINE_INFO is not available in this HANA instance."},
                cls=_CustomEncoder,
            )

        pdf = info_df.collect()
        rows = pdf.to_dict(orient="records")

        if operator:
            op_l = str(operator).strip().lower()
            rows = [r for r in rows if str(r.get("NAME", "")).lower() == op_l]
        if category:
            cat_l = str(category).strip().lower()
            rows = [r for r in rows if str(r.get("CATEGORY", "")).lower() == cat_l]
        if not include_parameters:
            for r in rows:
                r.pop("PARAMETER", None)

        return json.dumps(
            {"operators": rows, "count": len(rows)},
            cls=_CustomEncoder,
        )

    async def _arun(self, **kwargs) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)


# ---------------------------------------------------------------------------
# Tool 2 — GetAutoMLConfigDict
# ---------------------------------------------------------------------------

class GetAutoMLConfigDictInput(BaseModel):
    """
    Input schema for :class:`GetAutoMLConfigDict`.
    """
    pipeline_type: str = Field(
        description=(
            "AutoML pipeline type: one of 'classifier', 'regressor', 'timeseries'."
        ),
        default="timeseries",
    )
    config_dict: Optional[Union[str, dict]] = Field(
        description=(
            "The starting config_dict. Accepts the template keywords 'default', "
            "'light' (classifier/regressor only) or 'empty', a JSON string, or a dict. "
            "Defaults to 'default'."
        ),
        default="default",
    )


class GetAutoMLConfigDict(BaseTool):
    """
    Return the resolved ``config_dict`` for a given AutoML pipeline type.

    Calls ``_SYS_AFL.PAL_AUTOML_CONFIG`` with the supplied ``PIPELINE_TYPE``
    and ``CONFIG_DICT`` rows and no modifiers. This is the safe read-only
    entry-point: use it to fetch PAL's ``'default'`` / ``'light'`` /
    ``'empty'`` templates or to inspect a custom ``config_dict`` as PAL sees
    it (parameter values normalised, unknown keys stripped when
    ``VERIFY_CONFIG`` is off — which is the case here).

    Returns
    -------
    str
        A JSON string with:

        - ``pipeline_type`` : the normalised pipeline type.
        - ``config_dict`` : the resolved config_dict as a JSON object (or
          the raw string when PAL returns non-JSON content).
        - ``operators`` : per-operator INFO rows
          (``[{"operator", "type", "config"}]``).
        - ``error`` (optional): populated when the call fails.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 20 50
                :header-rows: 1

                * - Field
                  - Description
                * - pipeline_type
                  - AutoML pipeline type: 'classifier', 'regressor' or 'timeseries'.
                * - config_dict
                  - Template keyword ('default', 'light', 'empty') or a JSON object / string.
    """
    name: str = "get_automl_config_dict"
    """Name of the tool."""
    description: str = (
        "Fetch the resolved AutoML config_dict for a pipeline via PAL_AUTOML_CONFIG. "
        "Use this to obtain PAL's built-in 'default'/'light'/'empty' template as JSON, "
        "or to inspect a custom config_dict as PAL sees it. Read-only — VERIFY_CONFIG "
        "is off, so PAL best-effort ignores unknown keys. Follow up with "
        "modify_automl_config_dict when you need to add / remove / replace / modify "
        "operators or verify the schema strictly."
    )
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = GetAutoMLConfigDictInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        """Use the tool."""
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]

        pipeline_type, err = _validate_pipeline_type(kwargs.get("pipeline_type"))
        if err:
            return json.dumps({"error": err}, cls=_CustomEncoder)

        config_dict = kwargs.get("config_dict", "default")
        result_json, info_rows, call_err = _call_pal_automl_config(
            self.connection_context,
            pipeline_type=pipeline_type,
            config_dict=config_dict,
            verify=False,
        )
        if call_err is not None:
            return json.dumps(
                {"error": "pal_automl_config_failed", "detail": call_err},
                cls=_CustomEncoder,
            )

        return json.dumps(
            {
                "pipeline_type": pipeline_type,
                "config_dict": _parse_config_json(result_json),
                "operators": info_rows,
            },
            cls=_CustomEncoder,
        )

    async def _arun(self, **kwargs) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)


# ---------------------------------------------------------------------------
# Tool 3 — ModifyAutoMLConfigDict
# ---------------------------------------------------------------------------

class ModifyAutoMLConfigDictInput(BaseModel):
    """
    Input schema for :class:`ModifyAutoMLConfigDict`.
    """
    pipeline_type: str = Field(
        description="AutoML pipeline type: 'classifier', 'regressor' or 'timeseries'.",
        default="timeseries",
    )
    config_dict: Optional[Union[str, dict]] = Field(
        description=(
            "Starting config_dict. Accepts a template keyword ('default', 'light', "
            "'empty'), a JSON string, or a dict. Defaults to 'default'."
        ),
        default="default",
    )
    config_remove: Optional[Union[str, dict, list]] = Field(
        description="Estimator configs to remove (JSON object/string).",
        default=None,
    )
    config_add: Optional[Union[str, dict]] = Field(
        description="Estimator configs to add (JSON object/string).",
        default=None,
    )
    config_replace: Optional[Union[str, dict]] = Field(
        description="Estimator configs to replace (JSON object/string).",
        default=None,
    )
    config_modify: Optional[Union[str, dict]] = Field(
        description="Estimator configs to modify (JSON object/string).",
        default=None,
    )
    verify: Optional[bool] = Field(
        description=(
            "When True (default) PAL raises on invalid entries "
            "(VERIFY_CONFIG=1). Set to False to let PAL best-effort ignore "
            "unknown keys."
        ),
        default=True,
    )


class ModifyAutoMLConfigDict(BaseTool):
    """
    Verify (and, if needed, amend) an AutoML ``config_dict`` via
    ``_SYS_AFL.PAL_AUTOML_CONFIG``.

    **Preferred usage — explicit config_dict.** Build the full ``config_dict``
    yourself from the catalogue returned by :class:`GetPALPipelineInfo`, pass
    it as ``config_dict=<full JSON object>`` with ``verify=True`` and *no*
    ``config_add`` / ``config_remove`` / ``config_replace`` / ``config_modify``
    modifiers. The returned ``config_dict`` field is the PAL-normalized dict —
    hand that verbatim to the fit tool. This keeps the search space explicit
    and auditable, and any invalid operator / parameter is surfaced up-front
    with ``{"error": "invalid_config_dict", "detail": <pal message>}`` so the
    agent can fix it and re-verify.

    **Delta usage — modifiers on top of a template.** The optional
    ``CONFIG_REMOVE`` / ``CONFIG_ADD`` / ``CONFIG_REPLACE`` / ``CONFIG_MODIFY``
    rows exist for tweaking a starting template. PAL applies them in the
    order documented at the SAP Help portal: ``CONFIG_DICT → CONFIG_REMOVE →
    CONFIG_ADD → CONFIG_REPLACE → CONFIG_MODIFY``. Because the layering is
    less predictable than an explicit dict, reach for these modifiers only
    when you really want a delta against ``'default'`` / ``'light'`` /
    ``'empty'``.

    Returns
    -------
    str
        A JSON string with:

        - ``pipeline_type`` : the normalised pipeline type.
        - ``config_dict`` : the resolved config_dict as a JSON object.
        - ``operators`` : per-operator INFO rows.
        - ``error`` / ``detail`` (optional): populated when PAL rejects the
          modification. ``error == 'invalid_config_dict'`` when
          ``verify=True`` and PAL raises.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 20 50
                :header-rows: 1

                * - Field
                  - Description
                * - pipeline_type
                  - AutoML pipeline type: 'classifier', 'regressor' or 'timeseries'.
                * - config_dict
                  - Starting template or custom JSON. Defaults to 'default'.
                * - config_remove
                  - Estimator configs to remove (dict / JSON string).
                * - config_add
                  - Estimator configs to add.
                * - config_replace
                  - Estimator configs to replace.
                * - config_modify
                  - Estimator configs to modify in place.
                * - verify
                  - When True (default) PAL raises on invalid entries (VERIFY_CONFIG=1).
    """
    name: str = "modify_automl_config_dict"
    """Name of the tool."""
    description: str = (
        "Verify (and normalize) an AutoML config_dict against the PAL schema via "
        "PAL_AUTOML_CONFIG. PREFERRED USAGE: build the full explicit config_dict "
        "yourself from the operators / parameters returned by "
        "`get_pal_pipeline_info`, pass it as config_dict=<full JSON object> with "
        "verify=True and NO config_add / config_remove / config_replace / "
        "config_modify modifiers; the returned `config_dict` field is the "
        "PAL-normalized dict — hand that verbatim to the fit tool. This keeps the "
        "search space explicit and auditable. On PAL rejection the tool returns "
        "{\"error\": \"invalid_config_dict\", \"detail\": <pal message>} — fix the "
        "offending operator / parameter and re-verify. The optional "
        "CONFIG_REMOVE / CONFIG_ADD / CONFIG_REPLACE / CONFIG_MODIFY rows exist "
        "for tweaking on top of a template (PAL runs them in the order "
        "CONFIG_DICT -> REMOVE -> ADD -> REPLACE -> MODIFY) but the layering is "
        "less predictable than an explicit dict, so reach for them only when you "
        "really want a delta against `default`/`light`/`empty`."
    )
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ModifyAutoMLConfigDictInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        """Use the tool."""
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]

        pipeline_type, err = _validate_pipeline_type(kwargs.get("pipeline_type"))
        if err:
            return json.dumps({"error": err}, cls=_CustomEncoder)

        verify_raw = kwargs.get("verify")
        verify = True if verify_raw is None else bool(verify_raw)

        result_json, info_rows, call_err = _call_pal_automl_config(
            self.connection_context,
            pipeline_type=pipeline_type,
            config_dict=kwargs.get("config_dict", "default"),
            config_remove=kwargs.get("config_remove"),
            config_add=kwargs.get("config_add"),
            config_replace=kwargs.get("config_replace"),
            config_modify=kwargs.get("config_modify"),
            verify=verify,
        )
        if call_err is not None:
            payload = {"detail": call_err}
            payload["error"] = "invalid_config_dict" if verify else "pal_automl_config_failed"
            return json.dumps(payload, cls=_CustomEncoder)

        return json.dumps(
            {
                "pipeline_type": pipeline_type,
                "config_dict": _parse_config_json(result_json),
                "operators": info_rows,
            },
            cls=_CustomEncoder,
        )

    async def _arun(self, **kwargs) -> str:
        """Use the tool asynchronously."""
        return self._run(**kwargs)
