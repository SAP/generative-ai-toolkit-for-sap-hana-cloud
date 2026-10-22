"""
Utility functions for counting tokens in a query.
"""
import os
import re
import shutil
import tempfile
import atexit


def secure_report_dir(prefix: str) -> str:
    """V-162: create a private, randomized temp directory (0700) for generated artifacts.

    Replaces predictable, world-shared ``<tmp>/<fixed-name>`` paths created with
    ``os.makedirs(..., exist_ok=True)``, which silently accept a co-resident attacker's
    pre-planted symlink (symlink redirect / TOCTOU). ``mkdtemp`` creates a fresh directory
    owned by the current user with 0700 perms and fails if it cannot create it uniquely.
    """
    path = tempfile.mkdtemp(prefix=prefix if prefix.endswith(("-", "_")) else prefix + "-")
    try:
        os.chmod(path, 0o700)
    except OSError:
        # Windows ACLs are not represented by POSIX mode bits. mkdtemp still creates a
        # fresh directory; deployment-specific ACL policy remains the host's responsibility.
        pass
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def sanitize_artifact_name(name, fallback: str = "artifact") -> str:
    """V-162: strip path separators / traversal from a caller-controlled filename component."""
    # Treat both slash styles as separators on every platform so a Windows-style
    # attacker path is also contained when the service runs on Linux (and vice versa).
    base = str(name if name is not None else "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return cleaned or fallback


def count_tokens(query):
    """
    Counts the number of tokens in the given query.

    Parameters
    ----------
    query : str
        The input query.

    Returns
    -------
    int
        The number of tokens in the query.
    """
    # V-160: optional dependencies must never be installed as an import side effect.
    # Import nltk only for the feature that needs it, and fail with an actionable error.
    try:
        import nltk
    except ImportError as exc:  # pragma: no cover - depends on the installation extra
        raise ImportError(
            "count_tokens requires the optional 'nlp' dependencies; "
            "install hana_ai[nlp] or install nltk explicitly."
        ) from exc
    return len(nltk.word_tokenize(query))

def remove_prefix_sharp(table_name):
    """
    Removes the '#' prefix from the given table name.

    Parameters
    ----------
    table_name : str
        The input table name.

    Returns
    -------
    str
        The table name without the '#' prefix.
    """
    if table_name.startswith("#"):
        return table_name[1:]
    return table_name
