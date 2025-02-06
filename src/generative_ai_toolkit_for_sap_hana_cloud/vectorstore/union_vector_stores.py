"""
Aggregate different vector stores.

The following class is available:

    * :class `UnionVectorStores`
"""

# pylint: disable=redefined-builtin

import uuid
import numpy as np
from hana_ai.vectorstore.hana_vector_engine import HANAMLinVectorEngine

def _is_all_hana_vector_stores(vector_stores):
    """
    Check if all vector stores are HANA vector stores.

    Parameters:
    -----------
    vector_stores: list
        List of vector stores.
    """
    return all(isinstance(store, HANAMLinVectorEngine) for store in vector_stores)

def _hana_vector_stores_query(vector_stores, input, top_n):
    ranking_results = []
    for item in range(0, top_n):
        for store in vector_stores:
            distance = 1 if store.current_query_distance is None else store.current_query_distance
            ranking_results.append((float(distance), store.query(input, item + 1)))
            if item >= store.current_query_rows - 1:
                continue
    sorted_result = np.sort(ranking_results, axis=0)
    return sorted_result[top_n-1][1]

class UnionVectorStores(object):
    """
    Aggregate different vector stores.

    Parameters:
    -----------
    vector_stores: list
        List of vector stores.
    """
    def __init__(self, vector_stores):
        self.vector_stores = vector_stores
        if _is_all_hana_vector_stores(vector_stores):
            self.is_hana = True
        else:
            self.is_hana = False

    def query(self, input, top_n=1):
        """
        Query the vector stores.

        Parameters:
        -----------
        input: str
            Input.
        top_n: int
            Top N.
        """
        if self.is_hana:
            return _hana_vector_stores_query(self.vector_stores, input, top_n)
        else:
            num_bins = len(self.vector_stores)
            cur_bin = top_n % num_bins - 1
            if cur_bin < 0:
                cur_bin = 0
            cur_pos = top_n // num_bins
            return self.vector_stores[cur_bin].query(input, cur_pos + 1)

def merge_hana_vector_store(vector_stores, table_name=None, schema=None, **kwargs):
    """
    Merge the HANA vector stores.

    Parameters:
    -----------
    vector_stores: list
        List of vector stores.
    table_name: str
        Table name.
    schema: str, optional
        Schema name.
    """
    list_of_tables = list(map(lambda x: x.get_knowledge(), vector_stores))
    if schema is not None and table_name is not None:
        merged_df = list_of_tables[0].union(list_of_tables[1:]).save((schema, table_name))
    else:
        if table_name is None:
            table_name = "#merged_hana_vector_store_{}".format(str(uuid.uuid1()).replace('-', '_').upper())
        merged_df = list_of_tables[0].union(list_of_tables[1:]).save(table_name)
    new_hana_vec = HANAMLinVectorEngine(connection_context=merged_df.connection_context,
                                        table_name=table_name,
                                        schema=schema,
                                        **kwargs)
    return new_hana_vec
