import unittest
from hana_ai.vectorstore.embedding_service import GenAIHubEmbeddings

"create unittest for GenAIHubEmbeddings"
class TestGenAIHubEmbeddings(unittest.TestCase):
    def test_GenAIHubEmbeddings(self):
        model = GenAIHubEmbeddings()
        embeddings = model('hello')
        self.assertEqual(len(embeddings[0]), 1536)

