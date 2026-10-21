"""
Random Decision Tree Tools for SAP HANA ML.

Classes:
    - RDTTreeFitAndSave: Train and save a Random Decision Tree model.
    - RDTTreeLoadModelAndPredict: Load a trained model and make predictions.
    - RDTTreeLoadModelAndScore: Evaluate model performance on test data.

All parameters are strictly defined and documented for reliable LangChain integration.
"""
import logging
import json
from typing import Optional, Type, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from hana_ml import ConnectionContext
from hana_ml.model_storage import ModelStorage
from hana_ml.algorithms.pal.trees import RDTClassifier as RandomDecisionTree

logger = logging.getLogger(__name__)

class ModelFitInput(BaseModel):
    """
    Input schema for training a Random Decision Tree model.

    Parameters:
        fit_table (str): Name of the HANA table containing training data.
        name (str): Model name for storage.
        version (int, optional): Model version. Default: 1.
        target (str): Target variable for classification.
        features (List[str]): List of feature column names.
        n_estimators (int, optional): Number of trees. Default: 100.
        max_features (int, optional): Maximum number of features to consider.
        max_depth (int, optional): Maximum tree depth.
        min_samples_leaf (int, optional): Minimum samples per leaf. Default: 1.
        random_state (int, optional): Random seed for reproducibility.
        thread_ratio (float, optional): Ratio of threads to use.
        categorical_variable (List[str], optional): List of categorical feature names.
        key (str, optional): Key column for prediction/scoring.
        workload_class (str, optional): HANA workload class.
    """
    fit_table: str = Field(..., description="Training data table name")
    name: str = Field(..., description="Model name in storage")
    version: Optional[int] = Field(1, description="Model version")
    target: str = Field(..., description="Target variable for classification")
    features: List[str] = Field(..., description="List of feature columns")
    n_estimators: Optional[int] = Field(100, description="Number of trees")
    max_features: Optional[int] = Field(None, description="Maximum number of features to consider")
    max_depth: Optional[int] = Field(None, description="Maximum tree depth")
    min_samples_leaf: Optional[int] = Field(1, description="Minimum samples per leaf")
    random_state: Optional[int] = Field(None, description="Random seed for reproducibility")
    thread_ratio: Optional[float] = Field(None, description="Ratio of threads to use")
    categorical_variable: Optional[List[str]] = Field(None, description="List of categorical feature names")
    key: Optional[str] = Field(None, description="Key column for prediction/scoring")
    workload_class: Optional[str] = Field(None, description="HANA workload class")

class ModelPredictInput(BaseModel):
    """
    Input schema for model prediction.

    Parameters:
        predict_table (str): Table containing data to predict.
        name (str): Model name to load.
        version (int, optional): Model version. Default: 1.
        features (List[str]): Feature columns for prediction.
        key (str, optional): Key column for prediction.
    """
    predict_table: str = Field(..., description="Table containing data to predict")
    name: str = Field(..., description="Model name to load")
    version: Optional[int] = Field(1, description="Model version")
    features: List[str] = Field(..., description="Feature columns for prediction")
    key: Optional[str] = Field(None, description="Key column for prediction")

class ModelScoreInput(BaseModel):
    """
    Input schema for model scoring.

    Parameters:
        score_table (str): Table containing test data.
        name (str): Model name to evaluate.
        version (int, optional): Model version. Default: 1.
        target (str): Target variable for evaluation.
        features (List[str]): Feature columns for scoring.
    """
    score_table: str = Field(..., description="Table containing test data")
    name: str = Field(..., description="Model name to evaluate")
    version: Optional[int] = Field(1, description="Model version")
    target: str = Field(..., description="Target variable for evaluation")
    features: List[str] = Field(..., description="Feature columns for scoring")

class RDTTreeFitAndSave(BaseTool):
    """
    Train and save a Random Decision Tree model.

    Parameters: see ModelFitInput

    Returns:
        JSON string with status, model name, version, and training table.
    """
    name: str = "rdttree_fit_and_save"
    description: str = "Train a Random Decision Tree and save it to model storage"
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = ModelFitInput

    def __init__(self, connection_context: ConnectionContext, return_direct: bool = False) -> None:
        super().__init__(
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(self, **kwargs) -> str:
        fit_table = kwargs.get("fit_table")
        name = kwargs.get("name")
        target = kwargs.get("target")
        features = kwargs.get("features")

        if not self.connection_context.has_table(fit_table):
            raise ValueError(f"Table {fit_table} does not exist.")

        if not features or not isinstance(features, list):
            raise ValueError("Features must be provided as a non-empty list.")

        train_df = self.connection_context.table(fit_table)

        # Optional: Automatic categorical detection if not provided
        if not kwargs.get("categorical_variable"):
            cat_vars = []
            for col in features:
                try:
                    unique_vals = train_df.select(col).distinct().count()
                    if unique_vals < 10:
                        cat_vars.append(col)
                except Exception:
                    pass
            kwargs["categorical_variable"] = cat_vars

        rdt = RandomDecisionTree(
            n_estimators=kwargs.get("n_estimators", 100),
            max_features=kwargs.get("max_features"),
            max_depth=kwargs.get("max_depth"),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=kwargs.get("random_state"),
            thread_ratio=kwargs.get("thread_ratio"),
        )
        rdt.fit(
            train_df,
            label=target,
            features=features,
            categorical_variable=kwargs.get("categorical_variable")
        )

        if kwargs.get("workload_class"):
            rdt.enable_workload_class(kwargs["workload_class"])

        model_storage = ModelStorage(self.connection_context)
        rdt.name = name
        rdt.version = kwargs.get("version", 1)
        model_storage.save_model(model=rdt, if_exists='replace')

        return json.dumps({
            "status": "success",
            "model_name": name,
            "model_version": kwargs.get("version", 1),
            "trained_table": fit_table
        })

class RDTTreeLoadModelAndPredict(BaseTool):
    """
    Load a trained Random Decision Tree model and make predictions.

    Parameters: see ModelPredictInput

    Returns:
        JSON string with prediction table name and status.
    """
    name: str = "rdttree_load_and_predict"
    description: str = "Load a Random Decision Tree model and make predictions"
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = ModelPredictInput

    def __init__(self, connection_context: ConnectionContext, return_direct: bool = False) -> None:
        super().__init__(
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(self, **kwargs) -> str:
        predict_table = kwargs.get("predict_table")
        name = kwargs.get("name")
        version = kwargs.get("version", 1)
        features = kwargs.get("features")
        key = kwargs.get("key")

        if not self.connection_context.has_table(predict_table):
            raise ValueError(f"Table {predict_table} does not exist.")

        model_storage = ModelStorage(self.connection_context)
        model = model_storage.load_model(name=name, version=version)
        predict_df = self.connection_context.table(predict_table)

        pred_result = model.predict(
            data=predict_df,
            key=key,
            features=features
        )

        predictions = pred_result.collect().to_dict(orient="records")

        return json.dumps({
            "status": "success",
            "predictions": predictions
        })

class RDTTreeLoadModelAndScore(BaseTool):
    """
    Load a trained Random Decision Tree model and evaluate its performance.

    Parameters: see ModelScoreInput

    Returns:
        JSON string with evaluation metrics and status.
    """
    name: str = "rdttree_load_and_score"
    description: str = "Load a Random Decision Tree model and evaluate its performance"
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = ModelScoreInput

    def __init__(self, connection_context: ConnectionContext, return_direct: bool = False) -> None:
        super().__init__(
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(self, **kwargs) -> str:
        score_table = kwargs.get("score_table")
        name = kwargs.get("name")
        version = kwargs.get("version", 1)
        target = kwargs.get("target")
        features = kwargs.get("features")

        if not self.connection_context.has_table(score_table):
            raise ValueError(f"Table {score_table} does not exist.")

        model_storage = ModelStorage(self.connection_context)
        model = model_storage.load_model(name=name, version=version)
        score_df = self.connection_context.table(score_table)

        metrics = model.score(
            data=score_df,
            label=target,
            features=features
        )

        return json.dumps({
            "status": "success",
            "metrics": metrics
        })
