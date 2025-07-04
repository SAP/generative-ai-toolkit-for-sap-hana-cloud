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
