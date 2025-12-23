"""
Cross Encoder inside HANA

    * :class:`PALCrossEncoder`
"""

import uuid
import logging
from hdbcli import dbapi
import pandas as pd
from hana_ml.dataframe import create_dataframe_from_pandas, DataFrame
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    try_drop
)
# pylint: disable=line-too-long, super-with-arguments, too-many-arguments, too-many-positional-arguments, consider-using-f-string
logger = logging.getLogger(__name__)

class PALCrossEncoder(PALBase):
    """
    PAL embeddings base class.
    """
    def __init__(self, connection_context):
        super(PALCrossEncoder, self).__init__()
        self.connection_context = connection_context
        self.stats_ = None

    def _predict(self, data, thread_number=None, batch_size=None, max_token_num=None, model_version=None):
        """
        Reranking with Cross Encoder
        """
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = '#PAL_CROSSENCODER_RESULT_TBL_{}_{}'.format(0, unique_id)
        stats_tbl = '#PAL_CROSSENCODER_STATS_TBL_{}_{}'.format(0, unique_id)
        outputs = [result_tbl, stats_tbl]
        param_rows = [("BATCH_SIZE", batch_size, None, None),
                      ("MODEL_VERSION", None, None, model_version),
                      ("THREAD_NUMBER", thread_number, None, None),
                      ("MAX_TOKEN_NUM", max_token_num, None, None)]
        try:
            self._call_pal_auto(self.connection_context,
                                'PAL_CROSSENCODER',
                                data,
                                ParameterTable().with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            msg = str(self.connection_context.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            try_drop(self.connection_context, outputs)
            raise
        except Exception as db_err:
            logger.exception(str(db_err))
            try_drop(self.connection_context, outputs)
            raise
        self.stats_ = self.connection_context.table(stats_tbl)
        return self.connection_context.table(result_tbl)

    def predict(self, data, thread_number=None, batch_size=None, max_token_num=None, model_version=None, return_table=False):
        """
        Reranking with Cross Encoder
        """
        if isinstance(data, list):
            # data is list of (query, content)
            # have to upload data to hana table with columns id, query, content

            df = pd.DataFrame(data, columns=['QUERY', 'CONTENT'])
            # add id column to df
            df.insert(0, 'ID', range(len(df)))
            # create temporary table name with uuid
            temporary_table = "#PAL_CROSSENCODER_INPUT_TBL_" + str(uuid.uuid4()).replace("-", "_")
            data_ = create_dataframe_from_pandas(self.connection_context, df, table_name=temporary_table, disable_progressbar=True)
        elif isinstance(data, DataFrame):
            data_ = data
        else:
            raise ValueError("data should be list of (query, content) or hana_ml.dataframe.DataFrame with (id, query, content) columns")
        result = self._predict(data_,
                               thread_number=thread_number,
                               batch_size=batch_size,
                               max_token_num=max_token_num,
                               model_version=model_version)
        if return_table:
            return result
        # fetch result to pandas dataframe by collect with first column as id and second column as score
        result_df = result.collect()
        try_drop(self.connection_context, temporary_table)
        # sort result_df by id
        result_df = result_df.sort_values(by=result_df.columns[0])
        # return score column as ndarray
        return result_df[result_df.columns[1]].to_numpy()
