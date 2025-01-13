#unittest for hana_dataframe_agent
import unittest
from gen_ai_hub.proxy.langchain import init_llm
from hana_ml import dataframe
from hana_ml.algorithms.pal.utility import DataSets
from generative_ai_toolkit_for_sap_hana_cloud.agents.hana_dataframe_agent import create_hana_dataframe_agent
from generative_ai_toolkit_for_sap_hana_cloud.tools.toolkit import HANAMLToolkit
from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.code_templates import get_code_templates
from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.embedding_service import GenAIHubEmbeddings
from generative_ai_toolkit_for_sap_hana_cloud.vectorstore.hana_vector_engine import HANAMLinVectorEngine


class TestHanaDataframeAgent(unittest.TestCase):

    def test_agent_with_hana_vector_engine(self):
        llm = init_llm('gpt-4', temperature=0.0, max_tokens=256)
        url = "8e5882ff-9a6f-4b50-b479-9cf8f0060f1d.hana.demo-hc-3-haas-hc-dev.dev-aws.hanacloud.ondemand.com"
        port = 443
        user = "PAL_TEAM"
        pwd = "2Sz7Id7yrwOF1A"
        cc = dataframe.ConnectionContext(url, port, user, pwd, encrypt=True, sslValidateCertificate=False)
        data = DataSets.load_covid_data(cc)
        data = data.to_head("Date")
        hanavec = HANAMLinVectorEngine(cc, "test_table")
        hanavec.create_knowledge()
        toolkit = HANAMLToolkit()
        toolkit.set_vectordb(hanavec)
        agent = create_hana_dataframe_agent(llm=llm, toolkit=toolkit, df=data, verbose=True)
        result = agent.invoke("show me the columns in the dataframe")
        expected = "The columns in the dataframe are 'Date', 'Confirmed', 'Recovered', 'Deaths', and 'Increase rate'."
        self.assertEqual(result, expected)

        cc.drop_table("test_table")
