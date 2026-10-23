"""
SQL Agent for working with hana-ml objects.

The following function is available:

    * :func `create_hana_sql_agent`
"""
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
import re
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)
from sqlalchemy import MetaData

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import BasePromptTemplate
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.tools import BaseTool
from hana_ml.dataframe import ConnectionContext

from hana_ai.langchain_compat import AgentType


class _ReadOnlySQLDatabase(SQLDatabase):
    """V-171: enforce read-only access for the LLM SQL agent at the code level.

    The LangChain ``SQLDatabaseToolkit`` otherwise exposes full DML/DDL. Any generated
    statement that is not a single ``SELECT``/``WITH`` query is rejected here, so the agent
    cannot mutate or drop data even when the underlying HANA connection is privileged.
    This is defense-in-depth; production should also run under a least-privilege,
    schema-scoped read-only DB role.
    """

    _FORBIDDEN_KEYWORDS = {
        "alter", "call", "create", "delete", "do", "drop",
        "grant", "insert", "into", "lock", "merge", "revoke",
        "truncate", "update", "upsert",
    }
    _IDENTIFIER = r'(?:"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_$#]*)'
    _QUALIFIED_IDENTIFIER = rf'{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})?'
    _TABLE_REF = re.compile(
        rf"\b(?:from|join)\s+(?P<name>{_QUALIFIED_IDENTIFIER})",
        re.IGNORECASE,
    )
    _CTE_NAME = re.compile(
        rf"(?:\bwith|,)\s*(?P<name>{_IDENTIFIER})\s+as\s*\(",
        re.IGNORECASE,
    )
    _COMMA_JOIN = re.compile(
        rf"\bfrom\s+{_QUALIFIED_IDENTIFIER}(?:\s+(?:as\s+)?{_IDENTIFIER})?\s*,",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *args,
        allowed_schema: Optional[str] = None,
        allowed_tables: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        normalized_schema = (
            self._normalize_identifier(allowed_schema) if allowed_schema else None
        )
        if normalized_schema and len(normalized_schema) != 1:
            raise ValueError("allowed_schema must contain exactly one SQL identifier")
        self._allowed_schema = normalized_schema[0] if normalized_schema else None
        self._allowed_tables = {
            self._normalize_identifier(table) for table in (allowed_tables or [])
        }
        super().__init__(*args, **kwargs)

    @classmethod
    def _normalize_identifier(cls, value: str) -> tuple[str, ...]:
        """Return HANA-resolved components without quoted-identifier case collisions."""
        identifier_match = re.fullmatch(
            rf"\s*(?P<first>{cls._IDENTIFIER})"
            rf"(?:\s*\.\s*(?P<second>{cls._IDENTIFIER}))?\s*",
            str(value),
        )
        if not identifier_match:
            raise ValueError(f"invalid SQL allowlist identifier: {value!r}")

        parts = []
        for raw_part in identifier_match.group("first", "second"):
            if raw_part is None:
                continue
            # HANA folds unquoted identifiers to uppercase; quoted identifiers keep
            # exact case. This keeps SENSITIVE distinct from quoted lowercase names.
            if raw_part.startswith('"'):
                normalized = raw_part[1:-1].replace('""', '"')
            else:
                normalized = raw_part.upper()
            if not normalized or "\x00" in normalized:
                raise ValueError(f"invalid SQL allowlist identifier: {value!r}")
            parts.append(normalized)
        return tuple(parts)

    @staticmethod
    def _mask_literals_and_reject_comments(statement: str) -> str:
        masked = []
        index = 0
        state = "normal"
        while index < len(statement):
            char = statement[index]
            if state == "string":
                masked.append(" ")
                if char == "'":
                    if index + 1 < len(statement) and statement[index + 1] == "'":
                        masked.append(" ")
                        index += 1
                    else:
                        state = "normal"
            elif state == "identifier":
                # Preserve quoted identifiers for the object allowlist while ensuring that
                # apostrophes and comment markers inside them cannot alter scanner state.
                masked.append(char)
                if char == '"':
                    if index + 1 < len(statement) and statement[index + 1] == '"':
                        masked.append('"')
                        index += 1
                    else:
                        state = "normal"
            else:
                if char == "'":
                    state = "string"
                    masked.append(" ")
                elif char == '"':
                    state = "identifier"
                    masked.append(char)
                elif statement.startswith("--", index) or statement.startswith("/*", index):
                    raise ValueError("read-only SQL agent: SQL comments are not allowed")
                else:
                    masked.append(char)
            index += 1
        if state != "normal":
            raise ValueError(
                "read-only SQL agent: unterminated string literal or quoted identifier"
            )
        return "".join(masked)

    @classmethod
    def _top_level_statement_verb(cls, masked_statement: str) -> Optional[str]:
        """Return SELECT, or the verb following a top-level WITH CTE list."""
        keyword_text = re.sub(r'"(?:[^"]|"")*"', " ", masked_statement)
        tokens = re.findall(
            r"[A-Za-z_][A-Za-z0-9_$#]*|[(),]", keyword_text.lower()
        )
        if not tokens:
            return None
        if tokens[0] != "with":
            return tokens[0]

        depth = 0
        saw_as = False
        completed_cte = False
        for token in tokens[1:]:
            if depth:
                if token == "(":
                    depth += 1
                elif token == ")":
                    depth -= 1
                    if depth == 0:
                        completed_cte = True
                continue
            if completed_cte:
                if token == ",":
                    completed_cte = False
                    saw_as = False
                    continue
                if token not in {"(", ")"}:
                    return token
            elif token == "as":
                saw_as = True
            elif token == "(" and saw_as:
                depth = 1
                saw_as = False
        return None

    def _enforce_object_allowlist(self, masked_statement: str) -> None:
        if self._COMMA_JOIN.search(masked_statement):
            raise ValueError(
                "read-only SQL agent: comma-separated table references are not allowed"
            )
        cte_names = {
            self._normalize_identifier(match.group("name"))
            for match in self._CTE_NAME.finditer(masked_statement)
        }
        table_refs = [
            self._normalize_identifier(match.group("name"))
            for match in self._TABLE_REF.finditer(masked_statement)
        ]
        for table_ref in table_refs:
            if table_ref in cte_names:
                continue
            schema = table_ref[-2] if len(table_ref) == 2 else None
            table = (table_ref[-1],)
            if self._allowed_schema:
                if schema != self._allowed_schema:
                    raise ValueError(
                        "read-only SQL agent: every table must be qualified with the allowed schema"
                    )
            elif schema and table_ref not in self._allowed_tables:
                raise ValueError(
                    "read-only SQL agent: schema-qualified tables must be explicitly allowlisted"
                )
            if self._allowed_tables and (
                table not in self._allowed_tables and table_ref not in self._allowed_tables
            ):
                display_name = ".".join(table_ref)
                raise ValueError(
                    f"read-only SQL agent: table {display_name!r} is not allowlisted"
                )
            if not self._allowed_schema and not self._allowed_tables:
                raise ValueError("read-only SQL agent: no table allowlist is configured")

    def run(self, command, *args, **kwargs):  # type: ignore[override]
        if not isinstance(command, str):
            raise ValueError("read-only SQL agent: command must be SQL text")
        stmt = (command or "").strip()
        masked_stmt = self._mask_literals_and_reject_comments(stmt)
        # Reject stacked/multiple statements (allow a single optional trailing semicolon).
        if ";" in masked_stmt.rstrip().rstrip(";"):
            raise ValueError("read-only SQL agent: multiple/stacked statements are not allowed")
        if self._top_level_statement_verb(masked_stmt) != "select":
            raise ValueError("read-only SQL agent: only SELECT/WITH queries are permitted")
        keyword_text = re.sub(r'"(?:[^"]|"")*"', " ", masked_stmt)
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_$#]*", keyword_text.lower()))
        forbidden = sorted(words & self._FORBIDDEN_KEYWORDS)
        if forbidden:
            raise ValueError(
                f"read-only SQL agent: forbidden keyword {forbidden[0]!r}"
            )
        self._enforce_object_allowlist(masked_stmt)
        return super().run(command, *args, **kwargs)


class _sql_toolkit(object):
    def __init__(self, llm, db, tools=None):
        self.tools = tools
        self.llm = llm
        self.db = db

    @property
    def dialect(self) -> str:
        """Return string representation of SQL dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        if self.tools is None:
            return SQLDatabaseToolkit(llm=self.llm, db=self.db).get_tools()
        return self.tools + SQLDatabaseToolkit(llm=self.llm, db=self.db).get_tools()

    def get_context(self) -> dict:
        """Return db context that you may want in agent prompt."""
        return self.db.get_context()

def create_hana_sql_agent(
    llm: any,
    connection_context: ConnectionContext,
    tools: BaseTool = None,
    agent_type: Optional[
        Union[AgentType, Literal["openai-tools", "tool-calling"]]
    ] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    format_instructions: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    extra_tools: Sequence[BaseTool] = (),
    *,
    prompt: Optional[BasePromptTemplate] = None,
    schema: Optional[str] = None,
    include_tables: Optional[Sequence[str]] = None,
    **kwargs: Any,
):
    """Create a HANA SQL agent.

    Parameters
    ----------
    llm: any
        The language model to use.
    tools: BaseTool
        The tools to use.
    agent_type: Union[AgentType, Literal["openai-tools", "tool-calling"]], optional
        The type of agent to create.
    callback_manager: BaseCallbackManager, optional
        The callback manager to use.
    prefix: str, optional
        The prefix to use.
    suffix: str, optional
        The suffix to use.
    format_instructions: str, optional
        The format instructions to use.
    input_variables: List[str], optional
        The input variables to use.
    top_k: int
        The top k to use.
    max_iterations: int, optional
        The max iterations to use.
    max_execution_time: float, optional
        The max execution time to use.
    early_stopping_method: str
        The early stopping method to use.
    verbose: bool
        The verbose to use.
    agent_executor_kwargs: Dict[str, Any], optional
        The agent executor kwargs to use.
    extra_tools: Sequence[BaseTool]
        The extra tools to use.
    db: SQLDatabase, optional
        The database to use.
    connection_context: ConnectionContext
        The connection context to use.
    prompt: BasePromptTemplate, optional
        The prompt to use.
    schema: str, optional
        The only schema the agent may query. Queries must qualify table names with it.
    include_tables: Sequence[str], optional
        Explicit table allowlist. At least one of schema or include_tables is required.
    kwargs: Any
        The kwargs to use.

    examples
    --------
    Assume cc is a connection to a SAP HANA instance:

    >>> from hana_ai.agents.hana_sql_agent import create_hana_sql_agent
    >>> from hana_ai.tools.code_template_tools import GetCodeTemplateFromVectorDB
    >>> from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

    >>> hana_vec = HANAMLinVectorEngine(connection_context=cc, table_name="hana_vec_hana_ml_sql_knowledge")
    >>> hana_vec.create_knowledge(option='sql')
    >>> code_tool = GetCodeTemplateFromVectorDB()
    >>> code_tool.set_vectordb(vectordb=hana_vec)
    >>> agent_executor = create_hana_sql_agent(llm=llm, connection_context=cc, tools=[code_tool], include_tables=["IRIS_DATA_FULL_TBL"], verbose=True)
    >>> agent_executor.invoke("show me the min and max value of sepalwidthcm in the table iris_data_full_tbl?")
    """
    if agent_type is None and AgentType is not None:
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    if not schema and not include_tables:
        raise ValueError("schema or include_tables is required for SQL agent access")
    include_table_list = list(include_tables or [])
    engine = connection_context.to_sqlalchemy()
    metadata = MetaData()

    # Reflect tables WITH engine and case sensitivity
    metadata.reflect(
        bind=engine,
        views=True,
        case_sensitive=True,
        schema=schema,
        only=include_table_list or None,
    )
    db = _ReadOnlySQLDatabase(
        engine,
        metadata=metadata,
        schema=schema,
        include_tables=include_table_list or None,
        allowed_schema=schema,
        allowed_tables=include_table_list,
    )
    toolkit = _sql_toolkit(llm=llm, db=db, tools=tools)
    return create_sql_agent(llm=llm,
                            toolkit=toolkit,
                            agent_type=agent_type,
                            callback_manager=callback_manager,
                            prefix=prefix,
                            suffix=suffix,
                            format_instructions=format_instructions,
                            input_variables=input_variables,
                            top_k=top_k,
                            max_iterations=max_iterations,
                            max_execution_time=max_execution_time,
                            early_stopping_method=early_stopping_method,
                            verbose=verbose,
                            agent_executor_kwargs=agent_executor_kwargs,
                            extra_tools=extra_tools,
                            db=None,
                            prompt=prompt,
                            **kwargs)

def create_hana_sql_toolkit(
    llm: any,
    connection_context: ConnectionContext,
    *,
    schema: Optional[str] = None,
    include_tables: Optional[Sequence[str]] = None,
):
    """
    Create a HANA SQL toolkit.

    Parameters
    ----------
    llm: any
        The language model to use.
    connection_context: ConnectionContext
        The connection context to use.
    schema: str, optional
        The only schema the toolkit may query.
    include_tables: Sequence[str], optional
        Explicit table allowlist. At least one of schema or include_tables is required.
    """
    if not schema and not include_tables:
        raise ValueError("schema or include_tables is required for SQL toolkit access")
    include_table_list = list(include_tables or [])
    engine = connection_context.to_sqlalchemy()
    metadata = MetaData()

    # Reflect tables WITH engine and case sensitivity
    metadata.reflect(
        bind=engine,
        views=True,
        case_sensitive=True,
        schema=schema,
        only=include_table_list or None,
    )
    db = _ReadOnlySQLDatabase(
        engine,
        metadata=metadata,
        schema=schema,
        include_tables=include_table_list or None,
        allowed_schema=schema,
        allowed_tables=include_table_list,
    )
    return _sql_toolkit(llm=llm, db=db)
