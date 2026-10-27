"""Shared read-only SQL validation for LLM- and MCP-supplied statements."""

import re
from typing import Any, List


_SQL_WRITE_KEYWORDS = {
    "ALTER", "CALL", "CREATE", "DELETE", "DO", "DROP", "EXEC", "EXECUTE",
    "GRANT", "INSERT", "INTO", "MERGE", "REVOKE", "TRUNCATE",
    "UPDATE", "UPSERT",
}


def ensure_hana_ml_single_statement(sql_text: str) -> str:
    """Reject text that ``hana-ml`` would split before the database parses it.

    ``ConnectionContext.sql()`` and ``execute_sql()`` use ``str.split(";")``.
    SQL quoting therefore cannot make a semicolon safe at either sink.  Builders
    that cannot use DBAPI parameters must call this on the final SQL string.
    """
    if not isinstance(sql_text, str) or not sql_text.strip():
        raise ValueError("SQL input did not contain a statement")
    if ";" in sql_text:
        raise ValueError(
            "semicolons are not allowed in SQL passed to hana-ml execution APIs"
        )
    return sql_text


def _readonly_sql_tokens(sql: str) -> List[str]:
    """Return tokens outside literals/quoted identifiers and reject SQL comments."""
    visible = []
    index = 0
    state = "normal"
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "normal":
            if char == "'":
                state = "string"
                visible.append(" ")
            elif char == '"':
                state = "identifier"
                visible.append(" ")
            elif (char == "-" and next_char == "-") or (
                char == "/" and next_char == "*"
            ):
                raise ValueError("SQL comments are not allowed in generated SQL")
            else:
                visible.append(char)
        elif state == "string":
            if char == "'" and next_char == "'":
                visible.extend((" ", " "))
                index += 1
            elif char == "'":
                state = "normal"
                visible.append(" ")
            else:
                visible.append(" ")
        else:
            if char == '"' and next_char == '"':
                visible.extend((" ", " "))
                index += 1
            elif char == '"':
                state = "normal"
                visible.append(" ")
            else:
                visible.append(" ")
        index += 1
    if state != "normal":
        raise ValueError("unterminated SQL string or quoted identifier")
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[(),;]", "".join(visible).upper())


def _top_level_statement_verb(tokens: List[str]) -> str:
    """Return SELECT, or the verb following a top-level WITH CTE list."""
    if not tokens:
        return ""
    if tokens[0] != "WITH":
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
        elif token == "AS":
            saw_as = True
        elif token == "(" and saw_as:
            depth = 1
            saw_as = False
    return ""


def ensure_readonly_select(sql_text: str) -> str:
    """Accept exactly one comment-free SELECT/WITH statement."""
    if not isinstance(sql_text, str) or not sql_text.strip():
        raise ValueError("SQL input did not contain a statement")
    text = sql_text.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    # hana-ml ConnectionContext.sql() uses a raw str.split(";") before it creates
    # the DataFrame. It does not honor SQL string/identifier quoting and executes
    # every preceding fragment immediately. Therefore no semicolon may reach it,
    # including one that would be a legitimate character inside a SQL literal.
    ensure_hana_ml_single_statement(text)
    tokens = _readonly_sql_tokens(text)
    if not tokens or _top_level_statement_verb(tokens) != "SELECT":
        raise ValueError("SQL must contain only a SELECT or WITH statement")
    forbidden = sorted(_SQL_WRITE_KEYWORDS.intersection(tokens))
    if forbidden:
        raise ValueError(
            f"SQL contains forbidden operation(s): {', '.join(forbidden)}"
        )
    if re.search(r"\bFOR\s+UPDATE\b", " ".join(tokens)):
        raise ValueError("SELECT FOR UPDATE is not read-only")
    return text


def readonly_dataframe(connection_context: Any, sql_text: str) -> Any:
    """Validate untrusted SQL before handing it to hana-ml ConnectionContext.sql."""
    return connection_context.sql(ensure_readonly_select(sql_text))
