"""
Utility functions for counting tokens in a query.
"""

import nltk

def count_tokens(query):
    """
    Counts the number of tokens in the given query.

    Parameters:
    query (str): The input query.

    Returns:
    int: The number of tokens in the query.
    """
    return len(nltk.word_tokenize(query))

