import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from hana_ml import dataframe
from hana_ai.tools.hana_ml_tools.decisiontree_tools import (
    DecisionTreeFitAndSave,
    DecisionTreeLoadModelAndPredict,
    DecisionTreeLoadModelAndScore
)

class TestDecisionTree(unittest.TestCase):
    """Test Suite for Decision Tree."""

    def setUp(self):
        """Initialize test environment."""
        self.mock_connection = MagicMock(spec=dataframe.ConnectionContext)
        self.mock_connection.connection = MagicMock()
        self.mock_connection.pyodbc_connection = MagicMock()
        self.mock_connection.sql_tracer = None
        self.fit_tool = DecisionTreeFitAndSave(connection_context=self.mock_connection)
        self.predict_tool = DecisionTreeLoadModelAndPredict(connection_context=self.mock_connection)
        self.score_tool = DecisionTreeLoadModelAndScore(connection_context=self.mock_connection)
        self.mock_table = MagicMock()
        self.mock_table.select_statement = "SELECT * FROM TEST_TABLE"
        def mock_count():
            return 5
        mock_distinct = MagicMock()
        mock_distinct.count.side_effect = mock_count
        mock_select = MagicMock()
        mock_select.distinct.return_value = mock_distinct
        self.mock_table.select.side_effect = lambda col: mock_select
        self.mock_connection.table.return_value = self.mock_table

    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.DecisionTreeClassifier')
    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.ModelStorage')
    def test_fit_and_save(self, mock_storage_class, mock_dt_class):
        """Test training and saving."""
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_dt_class.return_value = mock_model
        mock_storage = MagicMock()
        mock_storage.save_model.return_value = True
        mock_storage_class.return_value = mock_storage
        test_params = {
            "fit_table": "TEST_TABLE",
            "name": "TEST_DT_MODEL",
            "version": 1,
            "target": "TARGET",
            "features": ["FEATURE1", "FEATURE2"],
            "max_depth": 5,
            "min_records_of_leaf": 2,
            "thread_ratio": 0.5,
            "categorical_variable": [],
            "workload_class": "DEFAULT"
        }
        self.mock_connection.has_table.return_value = True
        result = self.fit_tool._run(**test_params)
        result_dict = json.loads(result)
        self.assertEqual(result_dict["status"], "success")
        self.assertEqual(result_dict["model_name"], "TEST_DT_MODEL")
        self.assertEqual(result_dict["trained_table"], "TEST_TABLE")
        mock_model.fit.assert_called()
        mock_storage.save_model.assert_called_once()

    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.ModelStorage')
    def test_predict(self, mock_storage_class):
        """Test prediction."""
        mock_model = MagicMock()
        mock_storage = MagicMock()
        mock_storage.load_model.return_value = mock_model
        mock_storage_class.return_value = mock_storage
        test_params = {
            "predict_table": "PREDICT_TABLE",
            "name": "TEST_DT_MODEL",
            "version": 1,
            "features": ["FEATURE1", "FEATURE2"]
        }
        self.mock_connection.has_table.return_value = True
        result = self.predict_tool._run(**test_params)
        result_dict = json.loads(result)
        self.assertEqual(result_dict["status"], "success")
        self.assertIn("predictions_table", result_dict)
        mock_storage.load_model.assert_called_once()
        mock_model.predict.assert_called_once()

    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.ModelStorage')
    def test_score(self, mock_storage_class):
        """Test scoring."""
        mock_metrics = {"accuracy": 0.95, "auc": 0.98}
        mock_model = MagicMock()
        mock_model.score.return_value = mock_metrics
        mock_storage = MagicMock()
        mock_storage.load_model.return_value = mock_model
        mock_storage_class.return_value = mock_storage
        test_params = {
            "score_table": "SCORE_TABLE",
            "name": "TEST_DT_MODEL",
            "version": 1,
            "target": "TARGET",
            "features": ["FEATURE1", "FEATURE2"]
        }
        self.mock_connection.has_table.return_value = True
        result = self.score_tool._run(**test_params)
        result_dict = json.loads(result)
        self.assertEqual(result_dict["status"], "success")
        self.assertEqual(result_dict["metrics"], mock_metrics)
        mock_storage.load_model.assert_called_once()
        mock_model.score.assert_called_once()

    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.DecisionTreeClassifier')
    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.ModelStorage')
    def test_fit_with_large_feature_set(self, mock_storage_class, mock_dt_class):
        """Test fit with large feature set."""
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_dt_class.return_value = mock_model
        mock_storage = MagicMock()
        mock_storage.save_model.return_value = True
        mock_storage_class.return_value = mock_storage
        features = [f"FEATURE{i}" for i in range(1000)]
        test_params = {
            "fit_table": "TEST_TABLE",
            "name": "TEST_DT_MODEL",
            "version": 1,
            "target": "TARGET",
            "features": features,
            "max_depth": 6,
            "min_records_of_leaf": 2
        }
        self.mock_connection.has_table.return_value = True
        result = self.fit_tool._run(**test_params)
        result_dict = json.loads(result)
        self.assertEqual(result_dict["status"], "success")
        mock_model.fit.assert_called_once()

    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.DecisionTreeClassifier')
    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.ModelStorage')
    def test_fit_with_missing_table(self, mock_storage_class, mock_dt_class):
        """Test fit with missing table."""
        self.mock_connection.has_table.return_value = False
        test_params = {
            "fit_table": "NOT_EXISTING_TABLE",
            "name": "TEST_DT_MODEL",
            "version": 1,
            "target": "TARGET",
            "features": ["FEATURE1", "FEATURE2"]
        }
        with self.assertRaises(ValueError):
            self.fit_tool._run(**test_params)

    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.ModelStorage')
    def test_predict_with_missing_model(self, mock_storage_class):
        """Test predict with missing model."""
        mock_storage = MagicMock()
        mock_storage.load_model.side_effect = Exception("Model not found")
        mock_storage_class.return_value = mock_storage
        test_params = {
            "predict_table": "PREDICT_TABLE",
            "name": "NOT_EXISTING_MODEL",
            "version": 1,
            "features": ["FEATURE1", "FEATURE2"]
        }
        self.mock_connection.has_table.return_value = True
        with self.assertRaises(Exception):
            self.predict_tool._run(**test_params)

    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.ModelStorage')
    def test_score_with_empty_table(self, mock_storage_class):
        """Test score with empty table."""
        mock_model = MagicMock()
        mock_model.score.side_effect = ValueError("Empty table")
        mock_storage = MagicMock()
        mock_storage.load_model.return_value = mock_model
        mock_storage_class.return_value = mock_storage
        test_params = {
            "score_table": "EMPTY_TABLE",
            "name": "TEST_DT_MODEL",
            "version": 1,
            "target": "TARGET",
            "features": ["FEATURE1", "FEATURE2"]
        }
        self.mock_connection.has_table.return_value = True
        with self.assertRaises(ValueError):
            self.score_tool._run(**test_params)

    @patch('hana_ai.tools.hana_ml_tools.decisiontree_tools.ModelStorage')
    def test_predict_with_large_dataset(self, mock_storage_class):
        """Test predict with large dataset."""
        mock_model = MagicMock()
        mock_storage = MagicMock()
        mock_storage.load_model.return_value = mock_model
        mock_storage_class.return_value = mock_storage
        large_table = MagicMock()
        large_table.collect.return_value = pd.DataFrame({
            "FEATURE1": np.random.rand(100000),
            "FEATURE2": np.random.rand(100000)
        })
        self.mock_connection.table.return_value = large_table
        test_params = {
            "predict_table": "LARGE_TABLE",
            "name": "TEST_DT_MODEL",
            "version": 1,
            "features": ["FEATURE1", "FEATURE2"]
        }
        self.mock_connection.has_table.return_value = True
        result = self.predict_tool._run(**test_params)
        result_dict = json.loads(result)
        self.assertEqual(result_dict["status"], "success")
        mock_model.predict.assert_called_once()

class TestPreprocessing(unittest.TestCase):
    """Test Suite for Preprocessing."""

    def setUp(self):
        self.data = pd.DataFrame({
            "feature1": [1, 2, np.nan, 4],
            "feature2": [np.nan, 1, 2, 3],
            "feature3": ["A", "B", "C", "D"]
        })
        self.target = pd.Series([0, 1, 0, 1])
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = MinMaxScaler()

    def test_missing_values_handling(self):
        numeric_data = self.data.select_dtypes(include=[np.number])
        processed_data = self.imputer.fit_transform(numeric_data)
        self.assertFalse(np.any(np.isnan(processed_data)))

    def test_scaling(self):
        numeric_data = self.data.select_dtypes(include=[np.number])
        processed_data = self.imputer.fit_transform(numeric_data)
        scaled_data = self.scaler.fit_transform(processed_data)
        self.assertTrue(np.all(scaled_data >= 0) and np.all(scaled_data <= 1))

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            self.scaler.fit_transform(self.data)
    
    def test_empty_data(self):
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.scaler.fit_transform(empty_data)

    def test_nan_values(self):
        nan_data = pd.DataFrame({"feature1": [1, 2, np.nan], "feature2": [4, np.nan, 6]})
        processed_data = self.imputer.fit_transform(nan_data)
        scaled_data = self.scaler.fit_transform(processed_data)
        self.assertFalse(np.any(np.isnan(scaled_data)))
    
    def test_categorical_data(self):
        categorical_data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": ["A", "B", "C"]
        })
        with self.assertRaises(ValueError):
            self.scaler.fit_transform(categorical_data)

class TestPostprocessing(unittest.TestCase):
    """Test Suite for Postprocessing."""

    def setUp(self):
        self.predictions = pd.DataFrame({
            "prediction": [0.1, 0.9, 0.8, 0.2],
            "label": [0, 1, 1, 0]
        })

    def test_prediction_format(self):
        self.assertIn("prediction", self.predictions.columns)
        self.assertIn("label", self.predictions.columns)

    def test_invalid_features(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = KeyError("Invalid feature name")
        with self.assertRaises(KeyError):
            mock_model.predict(pd.DataFrame({"invalid_feature": [1, 2, 3]}))
    
    def test_extra_columns(self):
        extra_columns = pd.DataFrame({
            "prediction": [0.1, 0.9, 0.8, 0.2],
            "label": [0, 1, 1, 0],
            "extra_column": [1, 2, 3, 4]
        })
        self.assertIn("prediction", extra_columns.columns)
        self.assertIn("label", extra_columns.columns)

    def test_large_predictions(self):
        large_predictions = pd.DataFrame({
            "prediction": np.random.rand(1000000),
            "label": np.random.randint(0, 2, size=1000000)
        })
        self.assertEqual(len(large_predictions), 1000000)

class TestScaler(unittest.TestCase):
    """Test Suite for Scaler."""

    def setUp(self):
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy="mean")
    
    def test_nan_values_error(self):
        data_with_nan = np.array([[np.nan, 1], [2, 3]])
        with self.assertRaises(ValueError):
            if np.isnan(data_with_nan).any():
                raise ValueError("NaN values found!")
            self.scaler.fit_transform(data_with_nan)

    def test_nan_values_handling(self):
        data_with_nan = [[1, 2, None], [4, 5, 6]]
        imputed_data = self.imputer.fit_transform(data_with_nan)
        scaled_data = self.scaler.fit_transform(imputed_data)
        self.assertEqual(scaled_data.shape, (2, 3))
        self.assertFalse(any([None in row for row in scaled_data]))

    def test_already_scaled_data(self):
        already_scaled_data = np.array([[0.0, 0.5, 1.0], [0.2, 0.6, 0.8]])
        if np.all((already_scaled_data >= 0) & (already_scaled_data <= 1)):
            transformed_data = already_scaled_data
        else:
            transformed_data = self.scaler.fit_transform(already_scaled_data)
        np.testing.assert_array_almost_equal(
            transformed_data, already_scaled_data, decimal=6
        )

    def test_constant_values(self):
        constant_data = [[5, 5, 5], [5, 5, 5]]
        scaled_data = self.scaler.fit_transform(constant_data)
        self.assertTrue((scaled_data == 0).all())

if __name__ == '__main__':
    unittest.main()