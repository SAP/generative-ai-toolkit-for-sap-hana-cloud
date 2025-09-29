"""
This module includes embedding service from local embedding model or from llm_commons

The following classes are available:

    * :class `PALModelEmbeddings`
    * :class `HANAVectorEmbeddings`
"""

# pylint: disable=redefined-builtin
# pylint: disable=unnecessary-dunder-call
# pylint: disable=unused-argument

import re
from typing import List
import uuid

import subprocess
import sys

try:
    from gen_ai_hub.proxy.langchain import init_embedding_model as gen_ai_hub_embedding_model
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sap-ai-sdk-gen[all]"])
    from gen_ai_hub.proxy.langchain import init_embedding_model as gen_ai_hub_embedding_model

import pandas as pd
from langchain.embeddings.base import Embeddings
from hana_ml.dataframe import ConnectionContext, create_dataframe_from_pandas
from hana_ml.text.pal_embeddings import PALEmbeddings
from hana_ml.algorithms.pal.pal_base import try_drop

class PALModelEmbeddings(Embeddings):
    """
    PAL embedding model.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context.
    model_version : str, optional
        Model version. Default to None.
    batch_size : int, optional
        Batch size. Default to None.
    thread_number : int, optional
        Thread number. Default to None.
    is_query : bool, optional
        Use different embedding model for query purpose. Default to None.
    """
    model_version: str
    connection_context: ConnectionContext
    batch_size: int
    thread_number: int
    is_query: bool

    def __init__(self, connection_context, model_version=None, batch_size=None, thread_number=None, is_query=None):
        """
        Init PAL embedding model.
        """
        self.model_version = model_version
        self.connection_context = connection_context
        self.batch_size = batch_size
        self.thread_number = thread_number
        self.is_query = is_query

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        pe = PALEmbeddings(self.model_version)
        temporary_table = "#PAL_EMBEDDINGS_" + str(uuid.uuid4()).replace("-", "_")
        df = create_dataframe_from_pandas(self.connection_context, pandas_df=pd.DataFrame({"ID": range(len(input)), "TEXT": input}), table_name=temporary_table, disable_progressbar=True, table_type="COLUMN")
        result = pe.fit_transform(data=df, key="ID", target="TEXT", thread_number=self.thread_number, batch_size=self.batch_size, is_query=self.is_query)
        self.model_version = pe.stat_.collect().iat[1, 1]
        result = list(map(lambda x: list(x[0]), result[result.columns[-2]].collect().to_numpy()))
        try_drop(self.connection_context, temporary_table)
        try_drop(self.connection_context, pe._fit_output_table_names)
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Parameters
        ----------
        texts : List[str]
            List of texts.

        Returns
        -------
        List[List[float]]
            List of embeddings.
        """
        return self.__call__(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Parameters
        ----------
        text : str
            Text.

        Returns
        -------
        List[float]
            Embedding.
        """
        return self.__call__(text)[0]

    def get_text_embedding_batch(self, texts: List[str], show_progress=False, **kwargs):
        """
        Get text embedding batch.

        Parameters
        ----------
        texts : List[str]
            List of texts.

        Returns
        -------
        List[List[float]]
            List of embeddings.
        """
        return self.embed_documents(texts)

class HANAVectorEmbeddings(Embeddings):
    """
    PAL embedding model.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context.
    model_version : str, optional
        Model version.  Default to 'SAP_NEB.20240715'
    """
    model_version: str
    connection_context: ConnectionContext

    def __init__(self, connection_context, model_version='SAP_NEB.20240715'):
        """
        Init PAL embedding model.
        """
        self.model_version = model_version
        self.connection_context = connection_context

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return _cc_embed_query(self.connection_context, input, model_version=self.model_version)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Parameters
        ----------
        texts : List[str]
            List of texts.

        Returns
        -------
        List[List[float]]
            List of embeddings.
        """
        return self.__call__(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Parameters
        ----------
        text : str
            Text.

        Returns
        -------
        List[float]
            Embedding.
        """
        return self.__call__(text)[0]

    def get_text_embedding_batch(self, texts: List[str], show_progress=False, **kwargs):
        """
        Get text embedding batch.

        Parameters
        ----------
        texts : List[str]
            List of texts.

        Returns
        -------
        List[List[float]]
            List of embeddings.
        """
        return self.embed_documents(texts)

class GenAIHubEmbeddings(Embeddings):
    """
    A class representing the embedding service for GenAIHub.

    Parameters
    ----------
    model_id: str
        Model ID. Defaults to 'text-embedding-ada-002'.
    """
    model: Embeddings
    def __init__(self, model_id='text-embedding-ada-002', **kwargs):
        """
        Init embedding service from llm_commons.
        """
        self.model = gen_ai_hub_embedding_model(model_id, **kwargs)

    def __call__(self, input):
        result = []
        if isinstance(input, list):
            result = self.model.embed_documents(input)
        else:
            result.append(self.model.embed_query(input))
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Parameters
        ----------
        texts : List[str]
            List of texts.

        Returns
        -------
        List[List[float]]
            List of embeddings.
        """
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Parameters
        ----------
        text : str
            Text.

        Returns
        -------
        List[float]
            Embedding.
        """
        return self.model.embed_query(text)

    def get_text_embedding_batch(self, texts: List[str], show_progress=False, **kwargs):
        """
        Get text embedding batch.

        Parameters
        ----------
        texts : List[str]
            List of texts.

        Returns
        -------
        List[List[float]]
            List of embeddings.
        """
        return self.embed_documents(texts)

def _cc_embed_query(connection_context, query, model_version='SAP_NEB.20240715'):
    """
    Create a query embedding and return a vector.

    Parameters
    ----------
    connection_context : ConnectionContext
        The HANA connection context.
    query : str or list of str
        The query to embed.
    model_version : str, optional
        Text Embedding Model version. Options are 'SAP_NEB.20240715' and 'SAP_GXY.20250407'.

        Defaults to 'SAP_NEB.20240715'.

    Returns
    -------
    list of float when query is str, list of list of float when query is list of str
    """
    def _safe_escape_single_quotes(text):
        # 在需要时应用转义
        if "'" in text:
            # 检查是否已经包含转义序列
            if "''" not in text:
                escaped_prompt = re.sub(r"(?<!')'", "''", text)
            else:
                # 如果已经包含转义序列，直接使用原始prompt
                escaped_prompt = text
        else:
            escaped_prompt = text
        return escaped_prompt


    if isinstance(query, (list, tuple)):
        sql = ''
        for i, q in enumerate(query):
            if i > 0:
                sql += ' UNION ALL '
            escaped_query = _safe_escape_single_quotes(q)
            sql += f"SELECT '{escaped_query}' AS TEXT FROM DUMMY"
        return connection_context.sql(sql).add_vector("TEXT", text_type='QUERY', embed_col="EMBEDDING").select(["EMBEDDING"]).collect()["EMBEDDING"].tolist()
    escaped_query = _safe_escape_single_quotes(query)
    return connection_context.sql(f"SELECT '{escaped_query}' AS TEXT FROM DUMMY").add_vector("TEXT", text_type='QUERY', embed_col="EMBEDDING", model_version=model_version).select(["EMBEDDING"]).collect()["EMBEDDING"].iat[0]
