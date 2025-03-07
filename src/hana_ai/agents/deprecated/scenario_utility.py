"""
utility for scenario agent

This module provides utility functions for scenario agent.

The following function is available:

    * :func `find_substring_bracketed_by_tag`
    * :func `find_all_substrings_bracketed_by_tag`
    * :func `llm_invoke_wait_for_ratelimit`
    * :func `agent_invoke_wait_for_ratelimit`
    * :func `get_fields_by_llm`
    * :func `execute_code_with_fields`
"""
#pylint: disable=consider-using-in

import re
import time

from openai import RateLimitError
from termcolor import colored
from langchain_core.prompts import PromptTemplate
from hana_ml.algorithms.pal.tsa.stationarity_test import stationarity_test
from hana_ml.algorithms.pal.tsa.trend_test import trend_test
from hana_ml.algorithms.pal.tsa.seasonal_decompose import seasonal_decompose
from .scenario_prompts import EXECUTE_CODE_GENERIC, GET_FIELDS, GET_TARGET_SCHEMA, GET_TARGET_TABLE

def find_substring_bracketed_by_tag(string, start_tag='<tag>', end_tag='</tag>'):
    """
    Find substring bracketed by tag.

    Parameters
    ----------
    string : str
        String.
    start_tag : str, optional
        Start tag, by default '<tag>'
    end_tag : str, optional
        End tag, by default '</tag>'

    Returns
    -------
    str
        Substring.
    """
    pattern = re.escape(start_tag) + r'([\s\S]*?)' + re.escape(end_tag)
    match = re.search(pattern, string)
    if match:
        return match.group(1).strip()
    if start_tag.startswith('<') and start_tag.endswith('>'):
        start_tag = start_tag.replace('<', '&lt;')
        start_tag = start_tag.replace('>', '&gt;')
    if end_tag.startswith('<') and end_tag.endswith('>'):
        end_tag = end_tag.replace('<', '&lt;')
        end_tag = end_tag.replace('>', '&gt;')
    pattern = re.escape(start_tag) + r'([\s\S]*?)' + re.escape(end_tag)
    match = re.search(pattern, string)
    if match:
        return match.group(1).strip()
    return None

def find_all_substrings_bracketed_by_tag(string, start_tag='<tag>', end_tag='</tag>'):
    """
    Find all substrings bracketed by tag.

    Parameters
    ----------
    string : str
        String.
    start_tag : str, optional
        Start tag, by default '<tag>'
    end_tag : str, optional
        End tag, by default '</tag>'

    Returns
    -------
    list
        List of substrings.
    """
    pattern = re.escape(start_tag) + r'([\s\S]*?)' + re.escape(end_tag)
    result = [match.strip() for match in re.findall(pattern, string)]
    if len(result) == 0:
        if start_tag.startswith('<') and start_tag.endswith('>'):
            start_tag = start_tag.replace('<', '&lt;')
            start_tag = start_tag.replace('>', '&gt;')
        if end_tag.startswith('<') and end_tag.endswith('>'):
            end_tag = end_tag.replace('<', '&lt;')
            end_tag = end_tag.replace('>', '&gt;')
        pattern = re.escape(start_tag) + r'([\s\S]*?)' + re.escape(end_tag)
        result = [match.strip() for match in re.findall(pattern, string)]
    return result

def llm_invoke_wait_for_ratelimit(llm, prompt, is_wait_for_rate_limit=False):
    """
    Invoke LLM and wait for rate limit.

    Parameters
    ----------
    llm : BaseLLM
        LLM.
    prompt : str
        Prompt.
    is_wait_for_rate_limit : bool, optional
        Whether to wait for rate limit, by default False
    """
    if is_wait_for_rate_limit:
        try:
            return llm.invoke(prompt)
        except RateLimitError as e:
            print(f"Rate limit error: {e}")
            # Please retry after 2 seconds.
            retry_seconds = int(find_substring_bracketed_by_tag(e.message, start_tag="Please retry after ", end_tag=" seconds.").strip())
            print(f"Retry after {retry_seconds} seconds.")
            time.sleep(retry_seconds + 1)
            return llm_invoke_wait_for_ratelimit(llm, prompt, is_wait_for_rate_limit)
    return llm.invoke(prompt)

def agent_invoke_wait_for_ratelimit(agent, prompt, is_wait_for_rate_limit=False):
    """
    Agent invoke.
    """
    if is_wait_for_rate_limit:
        try:
            return agent.invoke(prompt)
        except RateLimitError as e:
            print(f"Rate limit error: {e}")
            # Please retry after 2 seconds.
            retry_seconds = int(find_substring_bracketed_by_tag(e.message, start_tag="Please retry after ", end_tag=" seconds.").strip())
            print(f"Retry after {retry_seconds} seconds.")
            time.sleep(retry_seconds + 1)
            return agent_invoke_wait_for_ratelimit(agent, prompt, is_wait_for_rate_limit)
    return agent.invoke(prompt)

def get_fields_by_llm(history_message, query, fields_description, llm, is_wait_for_rate_limit=True, verbose=False, show_prompt=False):
    """
    Get fields.

    Parameters
    ----------
    history_message : str or list
        History message.
    query : str
        Query.
    fields_description : dict
        Fields description. The key is the field name, and the value is the field description.
    llm : BaseLLM
        LLM.
    is_wait_for_rate_limit : bool, optional
        Whether to wait for rate limit, by default True
    verbose : bool, optional
        Whether to show verbose, by default False
    show_prompt : bool, optional
        Whether to show prompt, by default False

    Returns
    -------
    dict
        Fields.
    """
    def _clean_fields(fields):
        result = []
        for vv in fields:
            if len(vv) > 0:
                result.append(vv)
        if len(result) == 0:
            return None
        if len(result) == 1:
            return result[0]
        return result
    result = {}
    prompt = PromptTemplate.from_template(GET_FIELDS).format(history_message=history_message, query=query, fields_description=",".join(fields_description.values()), fields_key=",".join(fields_description.keys()))
    response = llm_invoke_wait_for_ratelimit(llm, prompt, is_wait_for_rate_limit).content
    if verbose:
        if show_prompt:
            print(colored(f"[Prompt] {prompt}", "light_blue"))
        print(colored("[AI] Finding fields...", "light_green"), colored(response, "light_red"))
    for kk in fields_description.keys():
        count = response.count(f"<{kk}>")
        found_field = None
        if count > 1:
            found_field = find_all_substrings_bracketed_by_tag(response, start_tag=f"<{kk}>", end_tag=f"</{kk}>")
            found_field = _clean_fields(found_field)
        else:
            found_field = find_substring_bracketed_by_tag(response, start_tag=f"<{kk}>", end_tag=f"</{kk}>")
        if found_field is not None:
            if found_field == "" or found_field == "None":
                found_field = None
            result[kk] = found_field
    return result

def execute_code_with_fields(agent, query, fields, code_template=None, is_wait_for_rate_limit=True, verbose=False, show_prompt=False):
    """
    Execute code with fields.

    Parameters
    ----------
    agent : AgentExecutor
        Agent.
    query : str
        Query.
    fields : dict
        Fields.
    additional_context : str, optional
        Additional context, by default ""
    is_wait_for_rate_limit : bool, optional
        Whether to wait for rate limit, by default True
    verbose : bool, optional
        Whether to show verbose, by default False
    show_prompt : bool, optional
        Whether to show prompt, by default False
    """
    def _format_dict(fields):
        result = ""
        for kk, vv in fields.items():
            result += f"{kk}: {vv}\n"
        return result
    prompt = PromptTemplate.from_template(EXECUTE_CODE_GENERIC).format(query=query, fields=_format_dict(fields), code_template=code_template)
    result = agent_invoke_wait_for_ratelimit(agent, prompt, is_wait_for_rate_limit)["output"]
    if verbose:
        if show_prompt:
            print(colored(f"[Prompt] {prompt}", "light_blue"))
        print(colored("[AI] Executing code...", "light_green"), colored(result, "light_red"))
    return result

def ts_char(df, key, endog):
    """
    This function is used to get the characteristics of time series data.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    key : str
        The key column of the DataFrame.
    endog : str
        The endogenous column of the DataFrame.
    """
    prompt = ''
    key_col_type = df.get_table_structure()[key]
    key_ = key
    df_ = df
    if 'INT' not in key_col_type.upper():
        key_ = "NEW_" + key
        df_ = df.add_id(key_, ref_col=key)

    # Intermitent Test
    zero_values = df_.filter(f'"{endog}" = 0').count()
    total_values = df_.count()
    if total_values == 0:
        zero_proportion = 1
    else:
        zero_proportion = zero_values / total_values
    prompt += f"Intermittent Test: proportion of zero values is {zero_proportion}\n"

    # Stationarity Test
    result = stationarity_test(df_, key_, endog).collect()
    prompt += "Stationarity Test: "
    for _, row in result.iterrows():
        prompt += f"The {row['STATS_NAME']} is {row['STATS_VALUE']}."
    prompt += "\n"

    # Trend Test
    result = trend_test(df_, key_, endog)[0].collect()
    for _, row in result.iterrows():
        if row['STAT_NAME'] == 'TREND':
            if row['STAT_VALUE'] == 1:
                prompt += 'Trend Test:' + " Upward trend."
            elif row['STAT_VALUE'] == -1:
                prompt += 'Trend Test:' + " Downward trend."
            else:
                prompt += 'Trend Test:' + " No trend."
    prompt += "\n"

    # Seasonality Test
    result = seasonal_decompose(df_, key_, endog)[0].collect()
    prompt += "Seasonality Test: "
    for _, row in result.iterrows():
        prompt += f"The {row['STAT_NAME']} is {row['STAT_VALUE']}."
    
    return prompt

def get_table(connection_context, llm, query, scenario, verbose=False, show_prompt=False):
        schema_list = connection_context.sql("SELECT SCHEMA_NAME FROM SCHEMAS").collect()['SCHEMA_NAME'].tolist()
        prompt = PromptTemplate.from_template(GET_TARGET_SCHEMA).format(query=query, schema_list=schema_list)
        result = llm_invoke_wait_for_ratelimit(llm, prompt, True)
        if verbose:
            if show_prompt:
                print(colored(f"[Prompt] {prompt}", "light_blue"))
            print(colored("[AI] Finding the schema...", "light_green"), colored(result.content, "light_red"))
        schema = find_substring_bracketed_by_tag(result.content, start_tag="<schema>", end_tag="</schema>")
        if schema == "" or schema == "None":
            schema = None
        table_list = connection_context.get_tables(schema=schema)['TABLE_NAME'].tolist()
        if len(table_list) == 0:
            return "No tables found in the schema."
        prompt = PromptTemplate.from_template(GET_TARGET_TABLE).format(table_list=table_list, query=query, scenario=scenario)
        result = llm_invoke_wait_for_ratelimit(llm, prompt, True)
        if verbose:
            if show_prompt:
                print(colored(f"[Prompt] {prompt}", "light_blue"))
            print(colored("[AI] Finding the working table...", "light_green"), colored(result.content, "light_red"))
        working = find_substring_bracketed_by_tag(result.content, start_tag="<table>", end_tag="</table>")
        if working == "" or working == "None":
            working = None
        return schema, working
