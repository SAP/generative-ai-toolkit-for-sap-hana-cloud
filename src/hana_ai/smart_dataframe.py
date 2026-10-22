"""
Smart DataFrame.

The following class is available:

    * :class `SmartDataFrame`
"""
import re
from typing import List

from hana_ml.dataframe import DataFrame

from hana_ai.langchain_compat import (
    AgentExecutor,
    BaseLLM,
    BaseTool,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessage,
    build_agent_executor,
)
from hana_ai.tools.df_tools.additive_model_forecast_tools import AdditiveModelForecastFitAndSave, AdditiveModelForecastLoadModelAndPredict
from hana_ai.tools.df_tools.automatic_timeseries_tools import AutomaticTimeSeriesFitAndSave, AutomaticTimeSeriesLoadModelAndPredict, AutomaticTimeSeriesLoadModelAndScore
from hana_ai.tools.df_tools.fetch_tools import FetchDataTool
from hana_ai.tools.df_tools.intermittent_forecast_tools import IntermittentForecast
from hana_ai.tools.df_tools.ts_outlier_detection_tools import TSOutlierDetection
from hana_ai.tools.df_tools.ts_visualizer_tools import TimeSeriesDatasetReport


_SQL_WRITE_KEYWORDS = {
    "ALTER", "CALL", "CREATE", "DELETE", "DO", "DROP", "EXEC", "EXECUTE",
    "GRANT", "INSERT", "INTO", "MERGE", "REVOKE", "TRUNCATE",
    "UPDATE", "UPSERT",
}


def _readonly_sql_tokens(sql: str) -> List[str]:
    """Return tokens outside literals/quoted identifiers and reject SQL comments."""
    visible = []
    i = 0
    state = "normal"
    while i < len(sql):
        char = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if state == "normal":
            if char == "'":
                state = "string"
                visible.append(" ")
            elif char == '"':
                state = "identifier"
                visible.append(" ")
            elif (char == "-" and nxt == "-") or (char == "/" and nxt == "*"):
                raise ValueError("SQL comments are not allowed in generated SQL")
            else:
                visible.append(char)
        elif state == "string":
            if char == "'" and nxt == "'":
                visible.extend((" ", " "))
                i += 1
            elif char == "'":
                state = "normal"
                visible.append(" ")
            else:
                visible.append(" ")
        else:
            if char == '"' and nxt == '"':
                visible.extend((" ", " "))
                i += 1
            elif char == '"':
                state = "normal"
                visible.append(" ")
            else:
                visible.append(" ")
        i += 1
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


def _extract_readonly_select(llm_output: str) -> str:
    """Accept exactly one comment-free SELECT/WITH statement from an LLM response."""
    if not isinstance(llm_output, str) or not llm_output.strip():
        raise ValueError("LLM did not return a SQL statement")
    text = llm_output.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    tokens = _readonly_sql_tokens(text)
    if not tokens or _top_level_statement_verb(tokens) != "SELECT":
        raise ValueError("generated SQL must contain only a SELECT or WITH statement")
    if ";" in tokens:
        raise ValueError("multiple or stacked SQL statements are not allowed")
    forbidden = sorted(_SQL_WRITE_KEYWORDS.intersection(tokens))
    if forbidden:
        raise ValueError(f"generated SQL contains forbidden operation(s): {', '.join(forbidden)}")
    if re.search(r"\bFOR\s+UPDATE\b", " ".join(tokens)):
        raise ValueError("SELECT FOR UPDATE is not read-only")
    return text

class SmartDataFrame(DataFrame):
    """
    Smart DataFrame.

    Parameters
    ----------
    dataframe : DataFrame
        Dataframe.

    Examples
    --------
    >>> from hana_ai.smart_dataframe import SmartDataFrame

    >>> sdf = SmartDataFrame(dataframe=hana_df)
    >>> sdf.configure(llm=llm, verbose=True)
    >>> sdf.ask(question="Show the samples of the dataset")
    >>> new_sdf = sdf.transform(question="Get last two rows")
    """
    llm: BaseLLM
    _dataframe: DataFrame
    tools: List[BaseTool]
    agent: AgentExecutor
    kwargs: dict
    def __init__(self, dataframe: DataFrame):
        super(SmartDataFrame, self).__init__(dataframe.connection_context, dataframe.select_statement)
        self._dataframe = dataframe
        self._is_configured = False
        self.prefix = f"Given the select statement of a dataframe: {self._dataframe.select_statement}, "

    def configure(self,
                  llm: BaseLLM,
                  tools: List[BaseTool] = None,
                  verbose: bool = False,
                  **kwargs):
        """
        Configure the Smart DataFrame.

        Parameters
        ----------
        llm : BaseLLM
            LLM.
        toolkit : List of BaseTool, optional
            The dataframe toos to be used.

            Defaults to df_tools.
        """
        conn = self._dataframe.connection_context
        self.llm = llm
        if tools is None:
            tools = [
                FetchDataTool(conn),
                TimeSeriesDatasetReport(conn),
                TSOutlierDetection(conn),
                AutomaticTimeSeriesFitAndSave(conn),
                AutomaticTimeSeriesLoadModelAndPredict(conn),
                AutomaticTimeSeriesLoadModelAndScore(conn),
                AdditiveModelForecastFitAndSave(conn),
                AdditiveModelForecastLoadModelAndPredict(conn),
                IntermittentForecast(conn)
            ]
        self.tools = tools
        self.ask_tools = tools
        self.transform_tools = []

        for tool in tools:
            if hasattr(tool, 'is_transform'):
                self.transform_tools.append(tool.set_transform(True))

        self.kwargs = kwargs
        system_prompt = "You are a helpful assistant with access to tools. Always use tools when appropriate."
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.ask_executor = build_agent_executor(
            self.llm,
            self.ask_tools,
            prompt=prompt,
            system_prompt=system_prompt,
            verbose=verbose,
        )
        self.transform_executor = build_agent_executor(
            self.llm,
            self.transform_tools,
            prompt=prompt,
            system_prompt=system_prompt,
            verbose=verbose,
        )
        self._is_configured = True

    def ask(self, question: str):
        """
        Ask a question.

        Parameters
        ----------
        question : str
            Question.
        verbose : bool, optional
            Verbose. Default to False.
        """
        if self._is_configured is False:
            raise Exception("The SmartDataFrame is not configured. Please call the configure method first.")
        agent_input = {
            "input": [{
                "type": "text", 
                "text": f"Context:\n{self.prefix}\n\nQuestion: {question}"
            }]
        }
        return self.ask_executor.invoke(agent_input)["output"]

    @classmethod
    def _construct(cls, dataframe: DataFrame, llm: BaseLLM, tools: List[BaseTool], **kwargs):
        sdf = cls(dataframe)
        sdf.configure(llm, tools, **kwargs)
        return sdf

    def transform(self, question: str, output_key='output'):
        """
        Transform the dataframe.

        Parameters
        ----------
        question : str
            Question.
        verbose : bool, optional
            Verbose. Default to False.
        """
        if self._is_configured is False:
            raise Exception("The SmartDataFrame is not configured. Please call the configure method first.")

        # Updated prompt to demand ONLY the SQL statement
        agent_input = {
            "input": [{
                "type": "text", 
                "text": (
                    f"Context:\n{self.prefix}\n\n"
                    f"Question: {question}\n\n"
                    "IMPORTANT: Return ONLY the SQL select statement as a string. "
                    "DO NOT include any additional text, explanations, or formatting. "
                    "ONLY the raw SQL query."
                )
            }]
        }

        result = self.transform_executor.invoke(agent_input)
        select_statement = result['output']

        select_statement = _extract_readonly_select(select_statement)

        # Create new SmartDataFrame with generated SQL
        sdf = self._construct(
            self._dataframe.connection_context.sql(select_statement),
            self.llm,
            self.tools,
            **self.kwargs
        )
        return sdf
