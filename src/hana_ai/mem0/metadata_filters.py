"""Shared, fail-closed metadata checks for memory search and export results.

These checks supplement backend filtering (including backends without filter
support); they never replace the manager's mandatory tenant/session predicates.
The memory API's list-membership $contains is retained, not HANA full-text search.
"""

from datetime import date, datetime
from fnmatch import fnmatchcase
import operator
from typing import Any, Dict, Optional


_COMPARISONS = {
    "$eq": operator.eq, "$ne": operator.ne,
    "$gt": operator.gt, "$gte": operator.ge,
    "$lt": operator.lt, "$lte": operator.le,
}


def _compare(actual: Any, expected: Any, operation: str) -> bool:
    """Compare scalar/date operands without coercing tenant IDs or nulls."""
    if expected is None:
        if operation == "$eq":
            return actual is None
        return operation == "$ne" and actual is not None
    if actual is None:
        # SQL comparisons with NULL do not match, including <> and NOT IN.
        return False
    if isinstance(expected, dict):
        if expected.get("type") != "date":
            return False
        expected = date.fromisoformat(expected["date"])
        if isinstance(actual, datetime):
            actual = actual.date()
        elif isinstance(actual, str):
            actual = date.fromisoformat(actual)
    numeric = (
        type(actual) in (int, float) and type(expected) in (int, float)
    )
    if not numeric and type(actual) is not type(expected):
        return False
    if not isinstance(expected, (str, int, float, bool, date)):
        return False
    if isinstance(expected, bool) and operation not in {"$eq", "$ne"}:
        return False
    return _COMPARISONS[operation](actual, expected)


def _matches_operator(actual: Any, operation: str, expected: Any) -> bool:
    """Evaluate one field predicate; malformed operands cannot match."""
    try:
        if operation in _COMPARISONS:
            return _compare(actual, expected, operation)
        if operation in {"$in", "$nin"}:
            if actual is None or not isinstance(expected, (list, tuple)) or not expected:
                return False
            if any(value is None for value in expected):
                return False
            comparison = "$eq" if operation == "$in" else "$ne"
            matches = [_compare(actual, value, comparison) for value in expected]
            # Incomparable/invalid operands must not become matches by negation.
            return any(matches) if operation == "$in" else all(matches)
        if operation == "$between":
            return (
                isinstance(expected, (list, tuple)) and len(expected) == 2
                and _compare(actual, expected[0], "$gte")
                and _compare(actual, expected[1], "$lte")
            )
        if operation == "$contains":
            wanted = expected if isinstance(expected, list) else [expected]
            values = actual if isinstance(actual, list) else [actual]
            return all(value in values for value in wanted)
        if operation == "$like":
            if not isinstance(actual, str) or not isinstance(expected, str):
                return False
            # SQL LIKE uses % and _. Escape glob metacharacters so brackets,
            # stars and question marks in metadata remain literal characters.
            escapes = {"%": "*", "_": "?", "*": "[*]", "?": "[?]", "[": "[[]"}
            pattern = "".join(escapes.get(char, char) for char in expected)
            return fnmatchcase(actual, pattern)
    except (TypeError, ValueError, KeyError, OverflowError):
        return False
    return False


def metadata_matches_filters(
    metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]
) -> bool:
    """Match scalar, comparison, membership, range, LIKE and AND/OR filters.

    Sibling fields and multiple operators on one field are conjunctive. Unknown
    operators, malformed conditions and incomparable values fail closed, rather
    than silently ignoring a condition or aborting an entire result batch.
    """
    if not isinstance(metadata, dict) or (filters is not None and not isinstance(filters, dict)):
        return False
    for key, expected in (filters or {}).items():
        if key in ("$and", "$or"):
            if not isinstance(expected, list) or not expected:
                return False
            if not all(isinstance(clause, dict) and clause for clause in expected):
                return False
            matches = [metadata_matches_filters(metadata, clause) for clause in expected]
            if not (all(matches) if key == "$and" else any(matches)):
                return False
        elif not isinstance(key, str) or key.startswith("$"):
            return False
        else:
            actual = metadata.get(key)
            conditions = (
                expected if isinstance(expected, dict) and "type" not in expected
                else {"$eq": expected}
            )
            if not conditions or not all(
                _matches_operator(actual, operation, value)
                for operation, value in conditions.items()
            ):
                return False
    return True
