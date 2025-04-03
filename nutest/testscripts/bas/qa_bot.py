"""
This script is designed to process a question and chat history using the HANA AI.
"""

import sys
import json
from datetime import datetime, date

from pandas import Timestamp

from numpy import int64
 
from gen_ai_hub.proxy.langchain import init_llm
from hana_ml import dataframe
import pandas as pd
from hana_ai.agents.hanaml_agent_with_memory import stateless_call
from hana_ai.tools.toolkit import HANAMLToolkit

class _CustomEncoder(json.JSONEncoder):
    """
    This class is used to encode the model attributes into JSON string.
    """
    def default(self, obj): #pylint: disable=arguments-renamed
        if isinstance(obj, Timestamp):
            # Convert Timestamp to ISO string
            return obj.isoformat()
        elif isinstance(obj, (datetime, date)):
            # Convert datetime or date to ISO string
            return obj.isoformat()
        elif isinstance(obj, (int64, int)):
            # Convert numpy int64 or Python int to Python int
            return int(obj)
        # Let other types use the default handler
        return super().default(obj)

# pylint: disable=line-too-long, broad-exception-caught

connection_context = dataframe.ConnectionContext(userkey="RaysKey") # need to discuss how to get the connection context
llm = init_llm('gpt-4-32k', temperature=0.0, max_tokens=400) # use proxy package
tools = HANAMLToolkit(connection_context, used_tools='all', return_direct={"ts_dataset_report": False}).get_tools()

def process_strings(question: str, chat_history: list[str]) -> str:
    """
    Process the input question and chat history using the HANA AI.
    """
    return stateless_call(
        question=question,
        chat_history=chat_history,
        tools=tools,
        llm=llm,
        verbose=False
    )

if __name__ == "__main__":
    try:
        # Read input
        raw_input = sys.stdin.read().strip()
        input_data = json.loads(raw_input)

        # Validate input
        if 'chat_history' not in input_data:
            raise ValueError("Missing 'chat_history' key")
        # validate input
        if 'question' not in input_data:
            raise ValueError("Missing 'question' key")
        if not isinstance(input_data['question'], str):
            raise ValueError("'question' must be a string")
        if not isinstance(input_data['chat_history'], list):
            raise ValueError("'chat_history' must be a list")
        if not all(isinstance(item, str) for item in input_data['chat_history']):
            raise ValueError("All items in 'chat_history' must be strings")

        # Process and output
        result = process_strings(input_data['question'], input_data['chat_history'])
        if isinstance(result, pd.DataFrame):
            result = result.to_dict(orient="records")
        print(json.dumps({"result": result}, ensure_ascii=False, cls=_CustomEncoder))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, cls=_CustomEncoder))
    finally:
        sys.stdout.flush()  # Ensure output is sent
