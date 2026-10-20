"""Opt-in Python execution for fully trusted hana-ml DataFrame workflows.

This tool is intended for agent-driven data analysis when an operation is easier
expressed as a short hana-ml DataFrame snippet, but only when the host application
can trust all code submitted to it.

Trust boundary
--------------
- Execution is disabled unless the host explicitly sets allow_dangerous_code=True.
- This tool is not included in HANAMLToolkit's default tools.
- Code runs in the host process with its filesystem, network and database access.
- cc, dataframe, pd and returned DataFrames expose unrestricted capabilities;
    sql() and table() are convenience helpers, not authorization checks.
- The AST checks and reduced builtins are best-effort guardrails, NOT a sandbox.

Never enable this tool for untrusted code or untrusted agent/MCP callers. There
is no process isolation, resource limit or execution timeout.
"""

from __future__ import annotations

import ast
import io
import json
import logging
from contextlib import redirect_stdout
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field, PrivateAttr
try:
    from langchain_core.tools import BaseTool
except Exception:  # pragma: no cover
    from hana_ai.langchain_compat import BaseTool

from hana_ml import ConnectionContext
from hana_ml import dataframe as hana_dataframe

logger = logging.getLogger(__name__)


class PythonExecInput(BaseModel):
    """Input schema for trusted, unsandboxed Python execution."""

    code: str = Field(
        description=(
            "Fully trusted Python code to execute in the host process (NOT sandboxed). "
            "Requires explicit host-side opt-in; tool arguments cannot enable execution. "
            "Write hana-ml DataFrame operations using the provided variables: cc, dataframe, pd, sql(), table(). "
            "Assign your final output to a variable named `result` (recommended)."
        )
    )
    result_var: str = Field(
        default="result",
        description="Name of the variable to return from the execution environment.",
    )
    max_output_chars: int = Field(
        default=6000,
        description="Maximum characters to return (stdout + formatted result).",
    )
    head_rows: int = Field(
        default=20,
        description="If the result is a hana-ml DataFrame, collect at most this many rows.",
    )


_FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "help",
}


class _BestEffortAstValidator(ast.NodeVisitor):
    """Reject some unsafe syntax; this is not a security boundary."""

    def __init__(self) -> None:
        """Create a validator instance."""
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> Any:  # noqa: N802
        """Reject import statements."""
        self.errors.append("Import statements are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:  # noqa: N802
        """Reject import-from statements."""
        self.errors.append("Import statements are not allowed")

    def visit_With(self, node: ast.With) -> Any:  # noqa: N802
        """Reject with-statements."""
        self.errors.append("with-statements are not allowed")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Any:  # noqa: N802
        """Reject async with-statements."""
        self.errors.append("with-statements are not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> Any:  # noqa: N802
        """Reject lambda expressions."""
        self.errors.append("lambda is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> Any:  # noqa: N802
        """Reject private/dunder attribute access."""
        # Block private/dunder attribute access
        if isinstance(node.attr, str) and node.attr.startswith("_"):
            self.errors.append("Access to private attributes is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: N802
        """Reject calls to known-dangerous builtins."""
        # Block obvious dangerous builtins by name
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in _FORBIDDEN_CALLS:
            self.errors.append(f"Call to forbidden builtin: {fn.id}")
        self.generic_visit(node)


def _truncate(text: str, limit: int) -> str:
    """Truncate a string to a character limit."""
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= limit else (s[: max(0, limit - 3)] + "...")


def _format_result(obj: Any, *, head_rows: int) -> str:
    """Format returned result into a compact, readable string."""
    if obj is None:
        return "(result is None)"

    # hana-ml DataFrame heuristic
    try:
        cls_name = obj.__class__.__name__
        mod_name = obj.__class__.__module__
        if cls_name == "DataFrame" and "hana_ml" in mod_name and hasattr(obj, "head") and hasattr(obj, "collect"):
            pdf = obj.head(head_rows).collect()
            try:
                return pdf.to_markdown(index=False)
            except Exception:
                return str(pdf)
    except Exception:
        pass

    # pandas DataFrame
    try:
        import pandas as pd  # local import allowed in host env

        if isinstance(obj, pd.DataFrame):
            return obj.head(head_rows).to_markdown(index=False)
    except Exception:
        pass

    # dict/list: pretty json
    if isinstance(obj, (dict, list)):
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return str(obj)

    return str(obj)


class PythonHanaMLExecTool(BaseTool):
    """Execute fully trusted Python only after explicit host-side opt-in.

    Setting allow_dangerous_code=True accepts in-process code execution with
    the host's permissions. It does not enable a sandbox or SQL authorization.
    """

    name: str = "python_hanaml_exec"
    description: str = (
        "DANGEROUS: execute fully trusted Python in the host process for hana-ml DataFrame operations. "
        "NOT sandboxed: code has the host's filesystem, network and database permissions. "
        "Disabled unless explicitly enabled by the host; never use with untrusted callers or code. "
        "Assign your final output to `result`."
    )
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = PythonExecInput
    return_direct: bool = False
    # Configuration belongs to the trusted host, never to the tool input schema.
    _allow_dangerous_code: bool = PrivateAttr(default=False)

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False,
        *,
        allow_dangerous_code: bool = False,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )
        # Require an actual True, rather than coercing strings such as "false".
        self._allow_dangerous_code = allow_dangerous_code is True
        if self._allow_dangerous_code:
            logger.warning(
                "python_hanaml_exec enabled for fully trusted code only; "
                "execution is not sandboxed and uses the host's permissions"
            )

    def _run(self, **kwargs: Any) -> str:
        # Fail closed before interpreting any caller-controlled code or options.
        if self._allow_dangerous_code is not True:
            return (
                "Error: python_hanaml_exec is disabled. This tool is not a sandbox. "
                "Only the host application can opt in with allow_dangerous_code=True "
                "when constructing the tool for fully trusted code and callers. "
                "Tool-call arguments cannot enable execution."
            )

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]

        code = str(kwargs.get("code") or "")
        result_var = str(kwargs.get("result_var") or "result")
        max_output_chars = int(kwargs.get("max_output_chars") or 6000)
        head_rows = int(kwargs.get("head_rows") or 20)

        if not code.strip():
            return "Error: code is required"

        # Parse + validate
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"

        validator = _BestEffortAstValidator()
        validator.visit(tree)
        if validator.errors:
            return "Error: code rejected by best-effort syntax checks.\n" + "\n".join(
                f"- {e}" for e in validator.errors
            )

        # Reduced builtins are a convenience guardrail, not containment.
        safe_builtins: Dict[str, Any] = {
            "True": True,
            "False": False,
            "None": None,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "enumerate": enumerate,
            "str": str,
            "int": int,
            "float": float,
            "print": print,
        }

        # Full-capability environment: only reachable after trusted host opt-in.
        try:
            import pandas as pd  # host env dependency
        except Exception:
            pd = None  # type: ignore

        cc = self.connection_context

        def sql(query: str):
            return cc.sql(query)

        def table(name: str, schema: Optional[str] = None):
            return cc.table(name, schema=schema)

        def head_collect(df: Any, n: int = 20):
            if hasattr(df, "head") and hasattr(df, "collect"):
                return df.head(int(n)).collect()
            raise TypeError("head_collect expects a hana-ml DataFrame")

        env: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "cc": cc,
            "dataframe": hana_dataframe,
            "pd": pd,
            "sql": sql,
            "table": table,
            "head_collect": head_collect,
        }

        stdout_buf = io.StringIO()
        try:
            compiled = compile(tree, filename="<python_hanaml_exec>", mode="exec")
            with redirect_stdout(stdout_buf):
                # pylint: disable=exec-used
                exec(compiled, env, env)  # noqa: S102
        except Exception as exc:
            logger.exception("python_hanaml_exec failed")
            out = stdout_buf.getvalue()
            msg = f"ExecutionError: {exc}\n"
            if out.strip():
                msg += "\n[stdout]\n" + out
            return _truncate(msg, max_output_chars)

        out = stdout_buf.getvalue()
        result = env.get(result_var)
        formatted = _format_result(result, head_rows=head_rows)

        msg_parts = []
        if out.strip():
            msg_parts.append("[stdout]\n" + out.strip())
        msg_parts.append(f"[result:{result_var}]\n" + formatted)
        return _truncate("\n\n".join(msg_parts), max_output_chars)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)
