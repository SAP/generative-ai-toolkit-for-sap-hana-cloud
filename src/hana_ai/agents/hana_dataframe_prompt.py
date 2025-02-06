# flake8: noqa
"""
HANA Dataframe Prompt
"""
PREFIX = """
You are working with a HANA dataframe in Python that is similar to Spark dataframe. The name of the dataframe is `df`. `connection_context` is `df`'s attribute. To handle connection or to use dataframe functions, you should use python_repl_ast tool. In most cases, you should use CodeTemplatesFromVectorDB tool. You should use the tools below to answer the question posed of you. :"""

SUFFIX = """
This is the result of `print(df.head().collect())`:
{df}

Begin!
Question: {input}
{agent_scratchpad}"""
