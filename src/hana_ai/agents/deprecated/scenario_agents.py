"""
Scenario Chat Agent

The following class is available:

    * :class:`HANAChatAgent`
"""
#pylint: disable=wildcard-import, too-many-return-statements, dangerous-default-value, consider-using-in, too-many-nested-blocks, consider-iterating-dictionary, unused-wildcard-import
import json
import os
import time
import ctypes
import logging

from openai import RateLimitError
from termcolor import colored

from langchain_core.prompts import PromptTemplate
from langchain.llms.base import BaseLLM

from hana_ml.dataframe import ConnectionContext, DataFrame, quotename
from hana_ml.artifacts.generators.hana import HANAGeneratorForCAP

from hana_ai.agents.hana_dataframe_agent import create_hana_dataframe_agent
from hana_ai.agents.hana_sql_agent import create_hana_sql_agent
from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine
from .scenario_prompts import *
from .scenario_utility import (
    find_substring_bracketed_by_tag,
    find_all_substrings_bracketed_by_tag,
    get_fields_by_llm,
    get_table,
    ts_char
)
from .scenario_scope import SCENARIO_DESCRIPTION
#to-dos:
# 1. handle multiple scenarios in one query "train & predict", "predict & analysis"
# 2. write a utility function for user to fulfill the mandatory/optional fields

logging.disable(logging.CRITICAL)

def _code_llm_invoke(code_llm, prompt, is_wait_for_rate_limit=True):
    """
    Code LLM invoke.
    """
    if is_wait_for_rate_limit:
        try:
            return code_llm.invoke(prompt)
        except RateLimitError as e:
            print(f"Rate limit error: {e}")
            # Please retry after 2 seconds.
            retry_seconds = int(find_substring_bracketed_by_tag(e.message, start_tag="Please retry after ", end_tag=" seconds.").strip())
            print(f"Retry after {retry_seconds} seconds.")
            time.sleep(retry_seconds + 1)
            return _code_llm_invoke(code_llm, prompt, is_wait_for_rate_limit=is_wait_for_rate_limit)
    return code_llm.invoke(prompt)

def _get_fields(history_message, query, fields_description, fields_key, code_llm, is_wait_for_rate_limit=True, verbose=False, show_prompt=False):
    """
    Get fields.
    """
    result = {}
    prompt = PromptTemplate.from_template(GET_FIELDS).format(history_message=history_message, query=query, fields_description=",".join(list(map(quotename, fields_description))), fields_key=",".join(fields_key))
    response = _code_llm_invoke(code_llm, prompt, is_wait_for_rate_limit).content
    if verbose:
        if show_prompt:
            print(colored(f"[Prompt] {prompt}", "light_blue"))
        print(colored("[AI] Finding fields...", "light_green"), colored(response, "light_red"))
    for kk in fields_key:
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

def _get_formatted_outputs(outputs, scenario_scope):
    if "current_model_type" not in outputs:
        return None
    if "current_scenario_type" not in outputs:
        return None
    outputs_dict = {}
    for scenario in scenario_scope:
        if scenario in outputs:
            if outputs["current_scenario_type"] in outputs[scenario]:
                if outputs["current_model_type"] in outputs[scenario][outputs["current_scenario_type"]]:
                    outputs_dict[scenario] = outputs[scenario][outputs["current_scenario_type"]][outputs["current_model_type"]]
    formated_outputs = ""
    for scenario, vv in outputs_dict.items():
        formated_outputs += f"The {scenario} table: {vv['data_table']}\n"
        formated_outputs += f"The {scenario} schema: {vv['data_schema']}\n"
        formated_outputs += f"The results/outputs after {scenario}: {vv['outputs']}\n"
    return formated_outputs

def _clean_fields(fields):
    """
    Clean fields.
    """
    result = []
    for vv in fields:
        if len(vv) > 0:
            result.append(vv)
    if len(result) == 0:
        return None
    if len(result) == 1:
        return result[0]
    return result

def _remove_quotes(text):
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text.startswith("`") and text.endswith("`"):
        return text[1:-1]
    if text.startswith("“") and text.endswith("”"):
        return text[1:-1]
    if text.startswith("‘") and text.endswith("’"):
        return text[1:-1]
    if text.startswith("‟") and text.endswith("”"):
        return text[1:-1]
    if text.startswith("„") and text.endswith("”"):
        return text[1:-1]
    if text.startswith("‹") and text.endswith("›"):
        return text[1:-1]
    if text.startswith("«") and text.endswith("»"):
        return text[1:-1]
    return text

def _get_scenario_code_template(scenario):
    """
    Get scenario code templates.
    """
    ids = []
    descriptions = []
    examples = []

    temp_directory = os.path.join(os.path.dirname(__file__), "scenario_knowledge_base", scenario)
    if not os.path.exists(temp_directory):
        return None
    for filename in os.listdir(temp_directory):
        if filename.endswith('.txt'):
            with open(os.path.join(temp_directory, filename)) as f:
                ids.append(filename.replace(".txt", ""))
                contents = f.read()
                #split contents by '------' into two parts: description and example
                contents = contents.split('------', maxsplit=1)
                #description
                descriptions.append(contents[0])
                #example
                examples.append(contents[1])
    return {"id": ids, "description": descriptions, "example": examples}

def _select_scenario(llm, query, scenario_scope, scenario_description, scenario, verbose, show_prompt):
    """
    Select scenario.
    """
    scenario_desc = ""
    for kk, vv in scenario_description.items():
        scenario_desc += f"\"{kk}\" represents that {vv}. "
    prompt = PromptTemplate.from_template(SCENARIO_ROUTER).format(query=query, scenario_scope=", ".join(list(map(quotename, scenario_scope))), scenario_description=scenario_desc, scenario=scenario)
    result = _code_llm_invoke(llm, prompt, is_wait_for_rate_limit=True)
    if verbose:
        if show_prompt:
            print(colored(f"[Prompt] {prompt}", "light_blue"))
        print(colored("[AI] Scenario router...", "light_green"), colored(result.content, "light_red"))
    scenario_result = find_substring_bracketed_by_tag(result.content, start_tag="<scenario>", end_tag="</scenario>")
    if scenario_result is not None:
        if scenario_result in scenario_description:
            return scenario_result
    return scenario

class HANAChatAgent(object):
    """
    HANA Chat Agent.

    Parameters
    ----------
    chat_llm : BaseLLM
        Chat LLM.
    code_llm : BaseLLM
        Code LLM.
    connection_context : ConnectionContext
        Connection context.
    scenario_scope : list
        Scenario scope.

        Default value is ["training", "prediction", "scoring", "result_analysis", "cap_artifacts"].
    scenario_types_scope : list
        Scenario types scope.

        Default value is ["classification", "regression", "timeseries", "clustering"].
    scenario_description : dict
        Scenario description.

        Default value is {
                            "training": "Training scenario",
                            "prediction": "Prediction scenario",
                            "scoring": "Scoring scenario",
                            "result_analysis": "The analysis of the prediction result or the trained model",
                            "cap_artifacts": "CAP artifacts generation"
                         }.
    force : bool
        Force to update the knowledge base.

        Default value is False.
    verbose : bool
        Verbose.
    show_prompt : bool
        Show prompt for visualize the prompt flow.
    """

    current_agent : any
    outputs : dict
    inputs : dict
    def __init__(self,
                 chat_llm: BaseLLM,
                 code_llm: BaseLLM,
                 connection_context: ConnectionContext,
                 scenario_scope : list = ["training", "prediction", "scoring", "result_analysis", "cap_artifacts", "ts_model_suggest", "database"],
                 scenario_types_scope : list = ["classification", "regression", "timeseries", "clustering"],
                 scenario_description = SCENARIO_DESCRIPTION,
                 force=False,
                 verbose=False,
                 show_prompt=False,
                 execute_confirm=True,
                 message_number=20,
                 is_wait_for_rate_limit=True):
        self.chat_llm = chat_llm
        self.code_llm = code_llm
        self.connection_context = connection_context
        self.scenario_scope = scenario_scope
        self.scenario_types_scope = scenario_types_scope
        self.scenario_description = scenario_description
        for ss in scenario_scope:
            if ss not in self.scenario_description:
                self.scenario_description[ss] = f"{ss} scenario"
        self.force = force
        self.verbose = verbose
        self.base_agent = {}
        for scenario in scenario_scope:
            self.base_agent[scenario] = HANABaseChatAgent(
                scenario,
                chat_llm,
                code_llm,
                connection_context,
                scenario_scope=scenario_scope,
                scenario_types_scope=scenario_types_scope,
                scenario_description=scenario_description,
                force=force,
                verbose=verbose,
                show_prompt=show_prompt,
                execute_confirm=execute_confirm)
        self.sql_agent = create_hana_sql_agent(llm=self.code_llm, connection_context=self.connection_context, verbose=self.verbose, handle_parsing_errors=True)
        self.current_agent = None
        self.outputs = {}
        self.inputs = {}
        self.show_prompt = show_prompt
        self.cap_fields = {"project_name": "MyProject", "output_dir": "cap_artifacts", "namespace": "hana.ml"}
        self.message_number = message_number
        self.is_wait_for_rate_limit = is_wait_for_rate_limit
        self.history_message = []

    def _generate_cap_artifacts(self, scenario, project_name, output_dir, namespace=None, archive=False):
        """
        Generate CAP artifacts.
        """
        self._fetch_outputs(scenario)
        if scenario in self.outputs:
            if self.outputs["current_scenario_type"] in self.outputs[scenario]:
                if self.outputs["current_model_type"] in self.outputs[scenario][self.outputs["current_scenario_type"]]:
                    python_address = self.outputs[scenario][self.outputs["current_scenario_type"]][self.outputs["current_model_type"]].get("python_object_address")
                    if python_address is not None:
                        model_scenario = ctypes.cast(int(python_address), ctypes.py_object).value
                        hanagen = HANAGeneratorForCAP(project_name=project_name,
                                                      output_dir=output_dir,
                                                      namespace=namespace)
                        hanagen.materialize_ds_data()
                        hanagen.generate_artifacts(model_scenario, model_position=True, cds_gen=False, archive=archive)

    def _generate_cap_artifacts_all(self, project_name, output_dir, namespace=None, archive=False):
        """
        Generate CAP artifacts all.
        """
        for scenario in ["training", "prediction", "scoring"]:
            self._generate_cap_artifacts(scenario, project_name, output_dir, namespace, archive)

    def _fetch_outputs(self, scenario):
        """
        Fetch outputs.
        """
        if scenario not in ["cap_artifacts", "ts_model_suggest", "database"]:
            if self.base_agent[scenario].output_tables is not None:
                self.outputs[scenario] = self.base_agent[scenario].output_tables
                self.outputs["current_model_type"] = self.base_agent[scenario].model_type
                self.outputs["current_scenario_type"] = self.base_agent[scenario].scenario_type
        return self.outputs

    def _fetch_inputs(self, scenario):
        """
        Fetch inputs.
        """
        if scenario not in ["cap_artifacts", "ts_model_suggest", "database"]:
            if self.base_agent[scenario].df is not None:
                self.inputs[scenario] = {
                        "data_table": self.base_agent[scenario].data_table,
                        "data_schema": self.base_agent[scenario].data_schema,
                        "current_model_type": self.base_agent[scenario].model_type,
                        "mandatory_fields": self.base_agent[scenario].mandatory_fields,
                        "optional_fields": self.base_agent[scenario].optional_fields
                    }
        return self.inputs

    def get_scenario_template(self):
        """
        Get scenario template.
        """
        return self.current_agent.get_scenario_template()

    def enable_wait_for_rate_limit(self):
        """
        Enable wait for rate limit.
        """
        self.is_wait_for_rate_limit = True
        for scenario in self.scenario_scope:
            self.base_agent[scenario].enable_wait_for_rate_limit()

    def disable_wait_for_rate_limit(self):
        """
        Disable wait for rate limit.
        """
        self.is_wait_for_rate_limit = False
        for scenario in self.scenario_scope:
            self.base_agent[scenario].disable_wait_for_rate_limit()

    def chat(self, query):
        """
        Chat with HANA Chat Agent.

        Parameters
        ----------
        query : str
            Query.
        """
        if len(self.history_message) > self.message_number:
            self.history_message = self.history_message[-self.message_number:]
        if query.startswith("/inputs"):
            for ss in self.scenario_scope:
                self._fetch_inputs(ss)
            return self.inputs

        if query.startswith("/outputs"):
            for ss in self.scenario_scope:
                self._fetch_outputs(ss)
            return self.outputs

        if query.startswith("/reset"):
            # user should confirm to reset
            confirm = input("Do you want to reset all the agents? (yes/no)")
            if confirm == "yes":
                for ss in self.scenario_scope:
                    self.base_agent[ss].reset()
                self.outputs = {}
                self.inputs = {}
                self.history_message = []
                return "All agents have been reset."

        if query.startswith("/sql"):
            query = query[4:].strip()
            return self.sql_agent.invoke(query)['output']

        for ss in self.scenario_scope:
            if query.startswith(f"/{ss}"):
                self.base_agent[ss]._fetch_global_outputs(self.outputs)
                chat_response = self.base_agent[ss].chat(query)
                self._fetch_inputs(ss)
                self._fetch_outputs(ss)
                self.history_message.append(f"[Question][{ss}]: " + query + "\n")
                self.history_message.append(f"[Response][{ss}]: " + chat_response + "\n")
                for sss in self.scenario_scope:
                    self.base_agent[sss].history_message = self.history_message
                return chat_response

        current_scenario = None
        if self.current_agent is not None:
            current_scenario = _select_scenario(llm=self.code_llm, query=query, scenario_scope=self.scenario_scope, scenario_description=self.scenario_description, scenario=self.current_agent.scenario, verbose=self.verbose, show_prompt=self.show_prompt)
        else:
            current_scenario = _select_scenario(llm=self.code_llm, query=query, scenario_scope=self.scenario_scope, scenario_description=self.scenario_description, scenario=None, verbose=self.verbose, show_prompt=self.show_prompt)
        if current_scenario in self.scenario_scope:
            if current_scenario == "database":
                temp_message = '\n'.join(self.history_message)
                formed_query = f"The historical conversations: `{temp_message}`. The user question is \"{query}\"."
                returned_message = self.sql_agent.invoke(formed_query)['output']
                self.history_message.append(f"[Question][{current_scenario}]: " + query + "\n")
                self.history_message.append(f"[Response][{current_scenario}]: " + returned_message + "\n")
                for ss in self.scenario_scope:
                    self.base_agent[ss].history_message = self.history_message
                return returned_message
            if current_scenario == "cap_artifacts":
                fields_description = ["project name", "output directory", "namespace"]
                fields_key = ["project_name", "output_dir", "namespace"]
                fields = _get_fields('', query, fields_description, fields_key, self.code_llm, is_wait_for_rate_limit=self.is_wait_for_rate_limit, verbose=self.verbose, show_prompt=self.show_prompt)
                if "project_name" in fields:
                    self.cap_fields["project_name"] = fields["project_name"]
                if "output_dir" in fields:
                    self.cap_fields["output_dir"] = fields["output_dir"]
                if "namespace" in fields:
                    self.cap_fields["namespace"] = fields["namespace"]
                self._generate_cap_artifacts_all(project_name=self.cap_fields["project_name"], output_dir=self.cap_fields["output_dir"], namespace=self.cap_fields["namespace"], archive=False)
                returned_message = "CAP artifacts have been generated."
                self.history_message.append(f"[Question][{current_scenario}]: " + query + "\n")
                self.history_message.append(f"[Response][{current_scenario}]: " + returned_message + "\n")
                for ss in self.scenario_scope:
                    self.base_agent[ss].history_message = self.history_message
                return returned_message
            if current_scenario == "ts_model_suggest":
                fields_description = ["table name", "id column name", "endog column name"]
                fields_key = ["table", "key", "endog"]
                fields = _get_fields('', query, fields_description, fields_key, self.code_llm, is_wait_for_rate_limit=self.is_wait_for_rate_limit, verbose=self.verbose, show_prompt=self.show_prompt)
                df = None
                if "table" in fields:
                    schema, table = get_table(connection_context=self.connection_context,
                                              llm=self.code_llm,
                                              scenario="ts_model_suggest",
                                              query=query,
                                              show_prompt=self.show_prompt,
                                              verbose=self.verbose)
                    if table:
                        df = self.connection_context.table(table, schema=schema)
                        if "key" in fields:
                            key = fields["key"]
                        else:
                            key = None
                        if key is None:
                            key = df.columns[0]
                        if "endog" in fields:
                            endog = fields["endog"]
                        else:
                            endog = None
                        if endog is None:
                            endog = df.columns[1]
                        ts_char_ = ts_char(df, key, endog)
                        prompt = PromptTemplate.from_template(MODEL_SUGGEST).format(query=query, data_char=ts_char_)
                        returned_message = _code_llm_invoke(self.code_llm, prompt, is_wait_for_rate_limit=True).content
                        if self.verbose:
                            if self.show_prompt:
                                print(colored(f"[Prompt] {prompt}", "light_blue"))
                            print(colored("[AI] Model suggest...", "light_green"), colored(returned_message, "light_red"))
                        #self.base_agent['training'].suggested_model = returned_message
                        self.history_message.append(f"[Question][{current_scenario}]: " + query + "\n")
                        self.history_message.append(f"[Response][{current_scenario}]: " + returned_message + "\n")
                        for ss in self.scenario_scope:
                            self.base_agent[ss].history_message = self.history_message
                        return returned_message
                returned_message = "Please provide the table name, id column name, and endog column name."
                self.history_message.append(f"[Question][{current_scenario}]: " + query + "\n")
                self.history_message.append(f"[Response][{current_scenario}]: " + returned_message + "\n")
                for ss in self.scenario_scope:
                    self.base_agent[ss].history_message = self.history_message
                return returned_message
            if self.current_agent is not None:
                if self.current_agent.scenario != current_scenario:
                    self.base_agent[current_scenario].agent = None
            self.current_agent = self.base_agent[current_scenario]
            self.current_agent._fetch_global_outputs(self.outputs)
            returned_message = self.current_agent.chat(query)
            self._fetch_inputs(current_scenario)
            self._fetch_outputs(current_scenario)
            self.history_message.append(f"[Question][{current_scenario}]: " + query + "\n")
            self.history_message.append(f"[Response][{current_scenario}]: " + returned_message + "\n")
            for ss in self.scenario_scope:
                self.base_agent[ss].history_message = self.history_message
            return returned_message

        return _code_llm_invoke(self.code_llm, query, is_wait_for_rate_limit=True).content

class HANABaseChatAgent(object):
    """
    HANA Base Chat Agent.

    Parameters
    ----------
    scenario : str
        Scenario.
    chat_llm : BaseLLM
        Chat LLM.
    code_llm : BaseLLM
        Code LLM.
    connection_context : ConnectionContext
        Connection context.
    scenario_scope : list
        Scenario scope.
    scenario_types_scope : list
        Scenario types scope.
    scenario_description : dict
        Scenario description.
    force : bool
        Force.
    verbose : bool
        Verbose.
    show_prompt : bool
        Show prompt.
    """
    knowledge_base: HANAMLinVectorEngine
    connection_context: ConnectionContext
    context: str
    model_type: str
    scenario : str
    scenario_type: str
    df: DataFrame
    history_message: list
    ai_response: list
    output_tables: dict
    data_table: str
    data_schema: str
    mandatory_fields: dict
    optional_fields: dict
    parameters_description: dict
    show_prompt: bool
    global_outputs: dict
    effective_query: list
    #suggested_model: str

    def __init__(self,
                 scenario : str,
                 chat_llm: BaseLLM,
                 code_llm: BaseLLM,
                 connection_context: ConnectionContext,
                 scenario_scope : list,
                 scenario_types_scope : list,
                 scenario_description: dict,
                 force=False,
                 verbose=False,
                 show_prompt=False,
                 execute_confirm=True):
        self.connection_context = connection_context
        for ss in scenario_scope:
            if scenario == ss:
                knowledge_table = f"HANA_AI_{ss.upper()}_KNOWLEDGE_BASE"
                break
        self.scenario_types_scope = scenario_types_scope
        self.scenario_scope = scenario_scope
        self.scenario = scenario
        if force:
            if self.connection_context.has_table(knowledge_table):
                self.connection_context.drop_table(knowledge_table)
        self.knowledge_base = HANAMLinVectorEngine(self.connection_context, knowledge_table)
        code_template = _get_scenario_code_template(self.scenario)
        if code_template is not None:
            if not self.connection_context.has_table(knowledge_table):
                self.knowledge_base.upsert_knowledge(code_template)
            else:
                if self.connection_context.table(knowledge_table).count() == 0:
                    self.knowledge_base.upsert_knowledge(code_template)
        self.agent = None
        self.context = None
        self.model_type = None
        self.scenario_type = None
        self.chat_llm = chat_llm
        self.code_llm = code_llm
        self.df = None
        self.history_message = []
        self.ai_response = []
        self.output_tables = {}
        self.verbose = verbose
        self.data_table = None
        self.data_schema = None
        self.mandatory_fields = {}
        self.optional_fields = {}
        parameters_desc = json.load(open(os.path.join(os.path.dirname(__file__), "hana_ml_parameters.json")))
        self.parameters_description = parameters_desc[self.scenario] if self.scenario in parameters_desc else {}
        self.show_prompt = show_prompt
        self.effective_query = []
        self.scenario_description = scenario_description
        self.is_wait_for_rate_limit = True
        self.execute_confirm = execute_confirm
        #self.suggested_model = None

    def get_scenario_template(self):
        """
        Get scenario template.
        """
        return {
            "scenario": self.scenario,
            "scenario_type": self.scenario_type,
            "model_type": self.model_type,
            "context": self.context,
            "data_table": self.data_table,
            "data_schema": self.data_schema,
            "mandatory_fields": self.mandatory_fields,
            "optional_fields": self.optional_fields
        }

    def enable_wait_for_rate_limit(self):
        """
        Enable wait for rate limit.
        """
        self.is_wait_for_rate_limit = True

    def disable_wait_for_rate_limit(self):
        """
        Disable wait for rate limit.
        """
        self.is_wait_for_rate_limit = False

    def code_llm_invoke(self, prompt):
        """
        Code LLM invoke.

        Parameters
        ----------
        prompt : str
            Prompt.
        """
        if self.is_wait_for_rate_limit:
            try:
                return self.code_llm.invoke(prompt)
            except RateLimitError as e:
                print(f"Rate limit error: {e}")
                # Please retry after 2 seconds.
                retry_seconds = find_substring_bracketed_by_tag(e.message, start_tag="Please retry after ", end_tag=" seconds.")
                if retry_seconds is None:
                    retry_seconds = 1
                    print(f"Retry after {retry_seconds} second.")
                else:
                    retry_seconds = int(retry_seconds.strip())
                    print(f"Retry after {retry_seconds} seconds.")
                time.sleep(retry_seconds + 1)
                return self.code_llm_invoke(prompt)
        return self.code_llm.invoke(prompt)

    def chat_llm_invoke(self, prompt):
        """
        Chat LLM invoke.

        Parameters
        ----------
        prompt : str
            Prompt.
        """
        if self.is_wait_for_rate_limit:
            try:
                return self.chat_llm.invoke(prompt)
            except RateLimitError as e:
                print(f"Rate limit error: {e}")
                # Please retry after 2 seconds.
                retry_seconds = int(find_substring_bracketed_by_tag(e.message, start_tag="Please retry after ", end_tag=" seconds.").strip())
                print(f"Retry after {retry_seconds} seconds.")
                time.sleep(retry_seconds + 1)
                return self.chat_llm_invoke(prompt)
        return self.chat_llm.invoke(prompt)

    def agent_invoke(self, prompt):
        """
        Agent invoke.

        Parameters
        ----------
        prompt : str
            Prompt.
        """
        if self.is_wait_for_rate_limit:
            try:
                return self.agent.invoke(prompt)
            except RateLimitError as e:
                print(f"Rate limit error: {e}")
                # Please retry after 2 seconds.
                retry_seconds = int(find_substring_bracketed_by_tag(e.message, start_tag="Please retry after ", end_tag=" seconds.").strip())
                print(f"Retry after {retry_seconds} seconds.")
                time.sleep(retry_seconds + 1)
                return self.agent_invoke(prompt)
        return self.agent.invoke(prompt)

    def _friendly_response(self, query, system_response):
        """
        Refine response.
        """
        prompt = PromptTemplate.from_template(REFINE_RESPONSE).format(history_message='\n'.join(self.history_message), query=query, system_response=system_response)
        result = self.chat_llm_invoke(prompt).content
        if self.verbose:
            if self.show_prompt:
                print(colored(f"[Prompt] {prompt}", "light_blue"))
            print(colored("[AI] Chat Response...", "light_green"), colored(result, "light_red"))
        return result

    def _fetch_global_outputs(self, outputs):
        """
        Fetch global outputs.
        """
        self.global_outputs = outputs

    def reset(self):
        """
        Reset.
        """
        self.agent = None
        self.context = None
        self.model_type = None
        self.scenario_type = None
        self.df = None
        self.history_message = []
        self.ai_response = []
        self.data_table = None
        self.data_schema = None
        self.mandatory_fields = {}
        self.optional_fields = {}
        self.effective_query = []
        self.output_tables = {}

    def refresh_knowledge_base(self):
        """
        Refresh knowledge base.
        """
        code_template = _get_scenario_code_template(self.scenario)
        if code_template is not None:
            self.knowledge_base.upsert_knowledge(code_template)

    def refresh_parameters_description(self):
        """
        Refresh parameters description.
        """
        parameters_desc = json.load(open(os.path.join(os.path.dirname(__file__), "hana_ml_parameters.json")))
        self.parameters_description = parameters_desc[self.scenario] if self.scenario in parameters_desc else {}

    #[future work] replace by a generic function to fetch the output table
    def _get_current_model_table(self):
        """
        Get current model table.
        """
        if self.model_type is not None:
            training_model_type = self.model_type
            if "training" in self.global_outputs:
                if self.scenario_type in self.global_outputs["training"]:
                    if training_model_type in self.global_outputs["training"][self.scenario_type]:
                        outputs_list = self.global_outputs.get("training", {}).get(self.scenario_type, {}).get(training_model_type, {}).get("outputs", [])
                        for output in outputs_list:
                            if "model" in output.lower():
                                return output
        return None

    def _get_mandatory_fields_template(self, model_type):
        """
        Get mandatory fields.
        """
        mandatory_fields = {}
        if "mandatory_fields" not in self.parameters_description[model_type]:
            return mandatory_fields
        for kk, _ in self.parameters_description[model_type]["mandatory_fields"].items():
            mandatory_fields[kk] = None
        return mandatory_fields

    def _get_optional_fields_template(self, model_type):
        optional_fields = {}
        if "optional_fields" not in self.parameters_description[model_type]:
            return optional_fields
        for kk, _ in self.parameters_description[model_type]["optional_fields"].items():
            optional_fields[kk] = None
        return optional_fields

    def _form_agent(self, query):
        #[future work] add a new router to decide whether fetch the data from existing table or from the results of the previous scenarios
        prompt = PromptTemplate.from_template(DATA_SOURCE_ROUTER).format(query=query, scenario=self.scenario)
        result = self.code_llm_invoke(prompt)
        if self.verbose:
            if self.show_prompt:
                print(colored(f"[Prompt] {prompt}", "light_blue"))
            print(colored("[AI] Checking the data source...", "light_green"), colored(result.content, "light_red"))
        if "<ans>yes</ans>" in result.content:
            outputs = _get_formatted_outputs(self.global_outputs, self.scenario_scope)
            prompt = PromptTemplate.from_template(GET_DATA_SOURCE_FROM_RESULT).format(query=query, calculated_result=outputs)
            result = self.code_llm_invoke(prompt)
            if self.verbose:
                if self.show_prompt:
                    print(colored(f"[Prompt] {prompt}", "light_blue"))
                print(colored("[AI] Getting the data source from the results...", "light_green"), colored(result.content, "light_red"))
            schema = find_substring_bracketed_by_tag(result.content, start_tag="<schema>", end_tag="</schema>")
            if schema == "" or schema == "None":
                schema = None
            table = find_substring_bracketed_by_tag(result.content, start_tag="<table>", end_tag="</table>")
            if table == "" or table == "None":
                table = None
            if table:
                if schema:
                    self.df = self.connection_context.table(table, schema=schema)
                else:
                    self.df = self.connection_context.table(table)
                self.data_table = table
                self.data_schema = schema
                self.agent = create_hana_dataframe_agent(llm=self.code_llm, df=self.df, verbose=self.verbose, handle_parsing_errors=True)
                return self.agent
        schema_list = self.connection_context.sql("SELECT SCHEMA_NAME FROM SCHEMAS").collect()['SCHEMA_NAME'].tolist()
        prompt = PromptTemplate.from_template(GET_TARGET_SCHEMA).format(query=query, schema_list=schema_list)
        result = self.code_llm_invoke(prompt)
        if self.verbose:
            if self.show_prompt:
                print(colored(f"[Prompt] {prompt}", "light_blue"))
            print(colored("[AI] Finding the schema...", "light_green"), colored(result.content, "light_red"))
        self.ai_response.append(result.content)
        schema = find_substring_bracketed_by_tag(result.content, start_tag="<schema>", end_tag="</schema>")
        if schema == "" or schema == "None":
            schema = None
        table_list = self.connection_context.get_tables(schema=schema)['TABLE_NAME'].tolist()
        if len(table_list) == 0:
            self.agent = create_hana_dataframe_agent(llm=self.code_llm, df=self.connection_context.sql("SELECT * FROM DUMMY"), verbose=self.verbose, handle_parsing_errors=True)
            return self.agent
        prompt = PromptTemplate.from_template(GET_TARGET_TABLE).format(table_list=table_list, query=query, scenario=self.scenario)
        result = self.code_llm_invoke(prompt)
        self.ai_response.append(result.content)
        if self.verbose:
            if self.show_prompt:
                print(colored(f"[Prompt] {prompt}", "light_blue"))
            print(colored("[AI] Finding the working table...", "light_green"), colored(result.content, "light_red"))
        working = find_substring_bracketed_by_tag(result.content, start_tag="<table>", end_tag="</table>")
        if working == "" or working == "None":
            working = None
        if working:
            if schema:
                self.df = self.connection_context.table(working, schema=schema)
            else:
                self.df = self.connection_context.table(working)
            self.data_table = working
            self.data_schema = schema
            self.agent = create_hana_dataframe_agent(llm=self.code_llm, df=self.df, verbose=self.verbose, handle_parsing_errors=True)
            return self.agent
        self.agent = create_hana_dataframe_agent(llm=self.code_llm, df=self.connection_context.sql("SELECT * FROM DUMMY"), verbose=self.verbose, handle_parsing_errors=True)
        return self.agent

    def chat(self, query):
        """
        Chat with HANA Base Chat Agent.

        Parameters
        ----------
        query : str
            Query.
        """
        try:
            model_types_scope = self.knowledge_base.get_knowledge().collect().iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error collecting knowledge base: {e}")
            model_types_scope = []
        # if self.suggested_model is not None:
        #     prompt = PromptTemplate.from_template(USE_SUGGESTED_MODEL).format(query=query, model_types_scope=", ".join(model_types_scope), suggested_model=self.suggested_model)
        #     model_suggestion = self.code_llm_invoke(prompt)
        #     if self.verbose:
        #         if self.show_prompt:
        #             print(colored(f"[Prompt] {prompt}", "light_blue"))
        #         print(colored(f"[AI] Checking if the user wants to use the suggested model...", "light_green"), colored(model_suggestion.content, "light_red"))
        #     suggested_model_type = find_substring_bracketed_by_tag(model_suggestion.content, start_tag="<model_type>", end_tag="</model_type>")
        #     if suggested_model_type in model_types_scope:
        #         if self.model_type is None:
        #             self.model_type = suggested_model_type
        #             query = "Suggested model_type: " + self.model_type + "\n" + query
        #             self.context = self.knowledge_base.query(input=f"Scenario type: {self.scenario_type}\nModel type: {self.model_type}\n User query: {query}")
        #             self.model_type = self.knowledge_base.model_type
        #             if self.model_type not in self.mandatory_fields:
        #                 self.mandatory_fields[self.model_type] = self._get_mandatory_fields_template(self.model_type)
        #             if self.model_type not in self.optional_fields:
        #                 self.optional_fields[self.model_type] = self._get_optional_fields_template(self.model_type)
        effective_response = []
        if len(self.ai_response) > 30:
            self.ai_response = self.ai_response[-30:]
        if self.agent is None:
            agent = self._form_agent(query)
            if isinstance(agent, str):
                returned_message = self._friendly_response(query, agent)
                return returned_message
        if self.data_table is not None:
            prompt = PromptTemplate.from_template(CHECK_DATA_CHANGE).format(data_table=self.data_table, query=query, scenario=self.scenario)
            change_table = self.code_llm_invoke(prompt)
            self.ai_response.append(change_table.content)
            if self.verbose:
                if self.show_prompt:
                    print(colored(f"[Prompt] {prompt}", "light_blue"))
                print(colored(f"[AI] Checking if the user wants to change the {self.scenario} data...", "light_green"), colored(change_table.content, "light_red"))
            change_table = find_substring_bracketed_by_tag(change_table.content, start_tag="<table>", end_tag="</table>")
            if change_table == "" or change_table == "None":
                change_table = None
            if change_table is not None:
                if self.execute_confirm:
                    msg = input(f"Detect the user wants to change the {self.scenario} data. Do you want to change the {self.scenario} data? It will delete all data from the previous {self.scenario} job. Do you want to continue? (yes/no)")
                    if msg == "yes":
                        self.reset()
                        return self.chat(query)
                    return "Ok, I will stop here."
                else:
                    self.reset()
                    return self.chat(query)
        is_update_count = False
        if self.scenario_type is None:
            prompt = PromptTemplate.from_template(CHECK_SCENARIO_CHANGE).format(query=query,
                                                                                scenario=self.scenario,
                                                                                scenario_type=self.scenario_type,
                                                                                scenario_types_scope=", ".join(self.scenario_types_scope))
            result = self.code_llm_invoke(prompt)
            self.ai_response.append(result.content)
            if self.verbose:
                if self.show_prompt:
                    print(colored(f"[Prompt] {prompt}", "light_blue"))
                print(colored("[AI] Checking scenario type...", "light_green"), colored(result.content, "light_red"))
            for ss in self.scenario_types_scope:
                if f"<scenario_type>{ss}</scenario_type>" in result.content:
                    self.scenario_type = ss
                    is_update_count = True
                    break
            if is_update_count:
                effective_response.append(result.content)
            if  self.scenario_type is None:
                self.effective_query.append(query)
                if not is_update_count:
                    effective_response.append(result.content)
                returned_message = self._friendly_response(query, effective_response)
                return returned_message
        is_update_count = False
        # [future work][done] improve the model type (e.g. AutomaticClassificationTraining) with more accurate response via llm
        effective_query = query
        if len(self.effective_query) > 0:
            effective_query = self.effective_query[-1] + "\n" + query
        prompt = PromptTemplate.from_template(CHECK_MODEL_TYPE_CHANGE).format(history_message='\n'.join(self.history_message), model_types_scope=", ".join(model_types_scope), model_type=self.model_type, query=effective_query, scenario=self.scenario)
        result = self.code_llm_invoke(prompt)
        self.ai_response.append(result.content)
        if self.verbose:
            if self.show_prompt:
                print(colored(f"[Prompt] {prompt}", "light_blue"))
            print(colored("[AI] Checking model type...", "light_green"), colored(result.content, "light_red"))
        previous_model_type = self.model_type
        is_update_count = False
        for model_type in model_types_scope:
            if f"<model_type>{model_type}</model_type>" in result.content:
                if self.model_type != model_type:
                    previous_model_type = self.model_type
                    self.model_type = model_type
                    # we can directly use id to fetch the knowledge, however we have to use HANA vector engine for semantic search
                    top_n = self.knowledge_base.get_knowledge().count()
                    self.context = self.knowledge_base.query(input=f"Model type: {self.model_type}\n") # [future work] remove vectore store and purely use model_type to fetch the knowledge but hana vector engine becomes useless...
                    if self.model_type != self.knowledge_base.model_type:
                        rag_err = f"RAG error! Fail to find the correct model type in the knowledge base. It should be the model_type {self.model_type} but it finds the {self.knowledge_base.model_type} in the knowledge base. Try to correct."
                        print(colored(rag_err, "red"))
                        for topk in range(1, top_n):
                            self.context = self.knowledge_base.query(input=f"Model type: {self.model_type}\n", top_n=topk)
                            if self.model_type == self.knowledge_base.model_type:
                                rag_err = f"The RAG issue has been resolved."
                                print(colored(rag_err, "blue"))
                                break
                        if self.model_type != self.knowledge_base.model_type:
                            return "Fail to find the correct model type in the knowledge base."
                        self.chat(rag_err)
                    if self.model_type not in self.mandatory_fields:
                        self.mandatory_fields[self.model_type] = self._get_mandatory_fields_template(self.model_type)
                        if previous_model_type is not None:
                            for kk, vv in self.mandatory_fields[previous_model_type].items():
                                if kk in self.mandatory_fields[self.model_type]:
                                    self.mandatory_fields[self.model_type][kk] = vv
                    if self.model_type not in self.optional_fields:
                        self.optional_fields[self.model_type] = self._get_optional_fields_template(self.model_type)
                    is_update_count = True
                    break
        if is_update_count:
            effective_response.append(result.content)
        is_update_count = False

        # fill up the optional fields
        if self.model_type in self.parameters_description:
            optional_fields_key = ", ".join(list(self.parameters_description[self.model_type]["optional_fields"].keys()))
            optional_fields_description = ", ".join(list(map(quotename, self.parameters_description[self.model_type]["optional_fields"].values())))
            effective_query = query
            if len(self.effective_query) > 0:
                effective_query = self.effective_query[-1] + "\n" + query
            if len(optional_fields_description) > 0:
                prompt = PromptTemplate.from_template(GET_FIELDS).format(history_message='\n'.join(self.history_message), query=effective_query, fields_description=optional_fields_description, fields_key=optional_fields_key)
                optional_fields_find_result = self.code_llm_invoke(prompt)
                self.ai_response.append(optional_fields_find_result.content)
                if self.verbose:
                    if self.show_prompt:
                        print(colored(f"[Prompt] {prompt}", "light_blue"))
                    print(colored(f"[AI] Checking optional fields: {optional_fields_description}...", "light_green"), colored(optional_fields_find_result.content, "light_red"))
                for kk, vv in self.optional_fields[self.model_type].items():
                    count = optional_fields_find_result.content.count(f"<{kk}>")
                    if count > 1:
                        found_field = find_all_substrings_bracketed_by_tag(optional_fields_find_result.content, start_tag=f"<{kk}>", end_tag=f"</{kk}>")
                        found_field = _clean_fields(found_field)
                    else:
                        found_field = find_substring_bracketed_by_tag(optional_fields_find_result.content, start_tag=f"<{kk}>", end_tag=f"</{kk}>")
                    if found_field is not None:
                        if found_field == "" or found_field == "None":
                            found_field = None
                        else:
                            is_update_count = True
                        self.optional_fields[self.model_type][kk] = found_field
                if is_update_count:
                    effective_response.append(optional_fields_find_result.content)
        is_update_count = False
        # if there's None in mandatory_fields, then ask the user to provide the missing fields
        # [future work][done] merge multiple prompts into one prompt

        if self.model_type in self.parameters_description:
            missing_mandatory_fields = {}
            for kk, vv in self.mandatory_fields[self.model_type].items():
                if self.scenario in ["prediction", "scoring"]:
                    if kk == "model_table":
                        self.mandatory_fields[self.model_type][kk] = self._get_current_model_table()
                        continue
                if vv is None:
                    missing_mandatory_fields[kk] = vv
            mandatory_fields_key = ", ".join(list(missing_mandatory_fields.keys()))
            mandatory_fields_description = ", ".join(list(map(quotename, [self.parameters_description[self.model_type]["mandatory_fields"][kk] for kk in missing_mandatory_fields.keys()])))
            effective_query = query
            if len(self.effective_query) > 0:
                effective_query = self.effective_query[-1] + "\n" + query
            if len(mandatory_fields_description) > 0:
                prompt = PromptTemplate.from_template(GET_FIELDS).format(history_message='\n'.join(self.history_message), query=effective_query, fields_description=mandatory_fields_description, fields_key=mandatory_fields_key)
                mandatory_fields_find_result = self.code_llm_invoke(prompt)
                self.ai_response.append(mandatory_fields_find_result.content)
                if self.verbose:
                    if self.show_prompt:
                        print(colored(f"[Prompt] {prompt}", "light_blue"))
                    print(colored(f"[AI] Checking mandatory fields: {mandatory_fields_description}...", "light_green"), colored(mandatory_fields_find_result.content, "light_red"))
                found_kk = []
                for kk, vv in missing_mandatory_fields.items():
                    count = mandatory_fields_find_result.content.count(f"<{kk}>")
                    if count > 1:
                        found_field = find_all_substrings_bracketed_by_tag(mandatory_fields_find_result.content, start_tag=f"<{kk}>", end_tag=f"</{kk}>")
                        found_field = _clean_fields(found_field)
                    else:
                        found_field = find_substring_bracketed_by_tag(mandatory_fields_find_result.content, start_tag=f"<{kk}>", end_tag=f"</{kk}>")
                    if found_field is not None:
                        if found_field == "" or found_field == "None":
                            found_field = None
                        self.mandatory_fields[self.model_type][kk] = found_field
                        if found_field is not None:
                            is_update_count = True
                            found_kk.append(kk)
                if is_update_count:
                    effective_response.append(mandatory_fields_find_result.content)
                for kk in found_kk:
                    missing_mandatory_fields.pop(kk)
                if len(missing_mandatory_fields) > 0:
                    self.effective_query.append(query)
                    if not is_update_count:
                        effective_response.append(mandatory_fields_find_result.content)
                    returned_message = self._friendly_response(query, effective_response)
                    return returned_message
        is_update_count = False
        prompt = PromptTemplate.from_template(CHECK_SCENARIO_CHANGE).format(scenario_type=self.scenario_type, query=query, scenario=self.scenario, scenario_types_scope=", ".join(self.scenario_types_scope))
        if self.model_type in self.parameters_description:
            for kk, vv in self.mandatory_fields[self.model_type].items():
                prompt += PromptTemplate.from_template(CHECK_MANDATORY_FIELD_CHANGE).format(mandatory_field_description=self.parameters_description[self.model_type]["mandatory_fields"][kk], mandatory_field_key=kk, mandatory_field_value=vv)
        result = self.code_llm_invoke(prompt)
        self.ai_response.append(result.content)
        if self.verbose:
            if self.show_prompt:
                print(colored(f"[Prompt] {prompt}", "light_blue"))
            print(colored("[AI] Checking if the user wants to change the scenario type or mandatory fields...", "light_green"), colored(result.content, "light_red"))
        new_scenario_type = find_substring_bracketed_by_tag(result.content, start_tag="<scenario_type>", end_tag="</scenario_type>")
        if new_scenario_type is not None:
            if self.scenario_type == new_scenario_type:
                new_scenario_type = None
            if  new_scenario_type == "None" or new_scenario_type == "":
                new_scenario_type = None
        new_mandatory_fields = {}
        if self.model_type in self.parameters_description:
            for kk, vv in self.mandatory_fields[self.model_type].items():
                new_field = find_substring_bracketed_by_tag(result.content, start_tag=f"<{kk}>", end_tag=f"</{kk}>")
                if new_field is not None:
                    if new_field == "" or new_field == "None":
                        new_field = None
                    if new_field != vv:
                        new_mandatory_fields[kk] = new_field

        if new_scenario_type is not None or len(new_mandatory_fields) > 0:
            if new_scenario_type is not None:
                self.scenario_type = new_scenario_type
            for kk, vv in new_mandatory_fields.items():
                self.mandatory_fields[self.model_type][kk] = vv
            self.effective_query.append(query)
            effective_response.append(result.content)
            returned_message = self._friendly_response(query, effective_response)
            return returned_message
        check_command_prompt = PromptTemplate.from_template(CHECK_COMMAND).format(query=query, scenario=self.scenario)
        check_command_result = self.code_llm_invoke(check_command_prompt)
        self.ai_response.append(check_command_result.content)
        if self.verbose:
            if self.show_prompt:
                print(colored(f"[Prompt] {check_command_prompt}", "light_blue"))
            print(colored(f"[AI] Checking if the user asks for {self.scenario}...", "light_green"), colored(check_command_result.content, "light_red"))
        if "<ans>yes</ans>" not in check_command_result.content:
            self.effective_query.append(query)
            effective_response.append(result.content)
            returned_message = self._friendly_response(query, effective_response)
            return returned_message
        if self.execute_confirm:
            should_continue = input(f"Do you want continue for {self.scenario}? (yes/no)")
            if should_continue != "yes":
                return "Ok, I will stop here."

        additional_context = PromptTemplate.from_template(ADDITIONAL_CONTEXT).format(scenario_type=self.scenario_type, mandatory_fields=self.mandatory_fields, optional_fields={k: v for k, v in self.optional_fields.items() if v is not None})

        prompt = PromptTemplate.from_template(EXECUTE_CODE).format(additional_context=additional_context, query=query, context=self.context)
        outputs = self.agent_invoke(prompt)
        if self.scenario not in ["training", "prediction", "scoring", "preprocessing"]:
            returned_message = self._friendly_response(query, outputs["output"])
            return returned_message
        prompt = PromptTemplate.from_template(GET_OUTPUTS).format(execute_result=outputs["output"])
        result = self.code_llm_invoke(prompt)
        self.ai_response.append("Find the corresponding output tables after execution: " + result.content)
        effective_response.append("The execution has been done. The corresponding output tables after execution: " + result.content)
        if self.verbose:
            if self.show_prompt:
                print(colored(f"[Prompt] {prompt}", "light_blue"))
            print(colored("[AI] Getting output tables...", "light_green"), colored(result.content, "light_red"))
        outputs_table = find_all_substrings_bracketed_by_tag(result.content, start_tag="<output>", end_tag="</output>")
        python_object_address = find_substring_bracketed_by_tag(result.content, start_tag="<python_object_address>", end_tag="</python_object_address>")
        if outputs_table is not None:
            if self.scenario_type not in self.output_tables:
                self.output_tables[self.scenario_type] = {}
            self.output_tables[self.scenario_type][self.model_type] = {
                "data_table": self.data_table,
                "data_schema": self.data_schema,
                "outputs": list(map(_remove_quotes, outputs_table)),
                "python_object_address": python_object_address
            }
            returned_message = self._friendly_response(query, effective_response) + "\n" + f"Output tables: {outputs_table}"
            return returned_message
        returned_message = self._friendly_response(query, effective_response)
        return returned_message
        