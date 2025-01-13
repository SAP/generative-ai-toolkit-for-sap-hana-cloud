from hana_ml import dataframe
from hana_ml.algorithms.pal.utility import DataSets, Settings

#url, port, user, pwd = Settings.load_config("./config/Demo_connection.ini")
url, port, user, pwd = Settings.load_config("./config/HAIDEV_connection.ini")
connection_context  = dataframe.ConnectionContext(url, port, user, pwd, encrypt=True, sslValidateCertificate=None)


from gen_ai_hub.proxy.core.proxy_clients import set_proxy_version
from hana_ml import dataframe
from hana_ml.algorithms.pal.utility import DataSets

#
set_proxy_version('gen-ai-hub') 

# %%
from gen_ai_hub.proxy.langchain import init_llm
from generative_ai_toolkit_for_sap_hana_cloud.agents.hana_dataframe_agent import create_hana_dataframe_agent
from generative_ai_toolkit_for_sap_hana_cloud.tools.toolkit import HANAMLToolkit
from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.embedding_service import GenAIHubEmbeddings
from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.hana_vector_engine import HANAMLinVectorEngine
from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.code_templates import get_code_templates
#llm = init_llm('gpt-35-turbo', temperature=0.0, max_tokens=512)
llm = init_llm('gpt-4', temperature=0.0, max_tokens=512)


embedding_func = GenAIHubEmbeddings()
toolkit = HANAMLToolkit()

#hana_vec = HANAMLinVectorEngine(connection_context, "hana_vec_hana_ml_knowledge_v2")

hana_vec = HANAMLinVectorEngine(connection_context, "second_hana_vec_hana_ml_knowledge")


toolkit.set_vectordb(hana_vec)


# Prompts:
# - "create a time series forecast model for the table SALES_REFUNDS include model statistics and save the model"
# - "calculate predictions and show results in a forecast line plot"
# - "generate the code to create the CAP artifacts and execute the code"

refunds_hdf = connection_context.table('SALES_REFUNDS')

agent = create_hana_dataframe_agent(llm=llm, toolkit=toolkit, df=refunds_hdf, verbose=True, handle_parsing_errors=True)


import sys
print('===OUTPUT BEGIN===')
agent.run(sys.argv[1])
print('===OUTPUT END===')





