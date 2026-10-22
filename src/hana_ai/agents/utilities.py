"""
Utilities for HANA AI agents.
"""

import json
import inspect

def _check_generated_cap_for_bas(intermediate_steps):
    """
    Check if the generated CAP artifacts are valid.

    Parameters
    ----------
    intermediate_steps : str
        The intermediate steps to check.

    Returns
    -------
    bool
        True if the generated CAP artifacts are valid, False otherwise.
    """
    try:
        ss = json.loads(intermediate_steps)
    except:
        return False
    if intermediate_steps is None:
        return False
    if not isinstance(ss, list):
        return False
    for step in ss:
        for substep in step:
            if isinstance(substep, dict) and 'type' in substep and substep['type'] == 'constructor':
                if 'kwargs' in substep and 'tool' in substep['kwargs']:
                    tool_name = substep['kwargs']['tool']
                    if tool_name == "cap_artifacts_for_bas":
                        return True
    return False

def _inspect_python_code(intermediate_steps, tools):
    try:
        ss = json.loads(intermediate_steps)
    except:
        return None
    if intermediate_steps is None:
        return None
    collect_tool_call = []
    if not isinstance(ss, list):
        return None
    for step in ss:
        for substep in step:
            if isinstance(substep, dict) and 'type' in substep and substep['type'] == 'constructor':
                if 'kwargs' in substep and 'tool' in substep['kwargs']:
                    tool_name = substep['kwargs']['tool']
                    for tool in tools:
                        if tool.name == tool_name:
                            collect_tool_call.append({"tool_name": tool_name, "parameters": json.dumps(substep['kwargs']['tool_input']), "python_code": inspect.getsource(tool._run)})
                            break
    return collect_tool_call

def _get_user_info(connection_context):
    # V-174: derive identity from a stable, structured identifier rather than string-parsing
    # the DBAPI connection repr (which is not a security-grade identifier and can collide or
    # be attacker-shaped, mis-mapping the memory table across tenants).
    userkey = getattr(connection_context, "userkey", None)
    if userkey:
        return userkey
    direct_user = getattr(connection_context, "user", None)
    if direct_user:
        return str(direct_user)
    conn = getattr(connection_context, "connection", None)
    for attr in ("user", "get_user"):
        val = getattr(conn, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                val = None
        if val:
            return str(val)
    # A normal hana_ml.ConnectionContext created with user/password does not retain a public
    # ``user`` field, and hdbcli does not expose one either. Query the authenticated HANA
    # session using constant SQL rather than falling back to the connection object's repr.
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_USER FROM DUMMY")
        row = cursor.fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
    raise ValueError(
        "Unable to determine a stable HANA identity from userkey, a structured user "
        "attribute, or SELECT CURRENT_USER"
    )
