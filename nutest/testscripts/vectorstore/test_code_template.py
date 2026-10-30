#this is unittest to test the code template for prediction
#
import unittest
from hana_ai.vectorstore.code_templates import get_code_templates
#test get_code_templates_for_prediction

class TestCodeTemplateForPrediction(unittest.TestCase):
    def test_get_code_templates_for_prediction(self):
        result = get_code_templates()
        template_count = len(result["id"])
        self.assertGreater(template_count, 0)
        self.assertEqual(len(result["description"]), template_count)
        self.assertEqual(len(result["example"]), template_count)
        self.assertEqual(len(set(result["id"])), template_count)
        self.assertTrue(all(result["id"]))
        self.assertTrue(all(result["description"]))
        self.assertTrue(all(result["example"]))
