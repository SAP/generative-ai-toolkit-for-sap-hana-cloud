#this is unittest to test the code template for prediction
#
import unittest
from hana_ai.vectorstore.code_templates import get_code_templates
#test get_code_templates_for_prediction

class TestCodeTemplateForPrediction(unittest.TestCase):
    def test_get_code_templates_for_prediction(self):
        result = get_code_templates()
        self.assertEqual(len(result["id"]), 5)
        self.assertEqual(len(result["description"]), 5)
        self.assertEqual(len(result["example"]), 5)
        self.assertEqual(result["id"][0], '0')
        self.assertEqual(result["id"][1], '1')
