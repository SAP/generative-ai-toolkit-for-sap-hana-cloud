"""Tools for grouped massive automatic time-series workflows."""

import json
import logging
from typing import Optional, Type, Union

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.massive_auto_ml import MassiveAutomaticTimeSeries
from hana_ml.model_storage import ModelStorage

from hana_ai.tools.hana_ml_tools.utility import (
    _CustomEncoder,
    _hana_safe_identifier,
    build_repaired_predict_dataframe,
    format_predict_mismatch_diagnostic,
    generate_model_storage_version,
    is_predict_feature_mismatch_error,
)

logger = logging.getLogger(__name__)


def _prefer_massive_model_table_for_inference(model):
    """Prefer the core model table over pipeline metadata during inference."""
    model_tables = getattr(model, "model_", None)
    if isinstance(model_tables, (list, tuple)) and len(model_tables) >= 2 and model_tables[1] is not None:
        normalized = list(model_tables)
        normalized[1] = None
        model.model_ = normalized
    return model


class ModelFitInput(BaseModel):
    """Schema for fitting a grouped massive automatic time-series model."""

    fit_table: str = Field(description="the table to fit the model. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model in model storage. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model in model storage, it is optional", default=None)
    fit_schema: Optional[str] = Field(description="the schema of the fit_table, it is optional", default=None)
    scorings: Optional[dict] = Field(description="the scorings for the model, e.g. {'MAE':-1.0, 'EVAR':1.0} and it supports EVAR, MAE, MAPE, MAX_ERROR, MSE, R2, RMSE, WMAPE, LAYERS, SPEC, TIME, and it is optional", default=None)
    generations: Optional[int] = Field(description="the number of iterations of the pipeline optimization., it is optional", default=None)
    population_size: Optional[int] = Field(description="the number of individuals in the population., it is optional", default=None)
    offspring_size: Optional[int] = Field(description="the number of children to produce at each generation., it is optional", default=None)
    elite_number: Optional[int] = Field(description="the number of the best individuals to select for the next generation., it is optional", default=None)
    min_layer: Optional[int] = Field(description="the minimum number of layers in the pipeline., it is optional", default=None)
    max_layer: Optional[int] = Field(description="the maximum number of layers in the pipeline., it is optional", default=None)
    mutation_rate: Optional[float] = Field(description="the mutation rate., it is optional", default=None)
    crossover_rate: Optional[float] = Field(description="the crossover rate., it is optional", default=None)
    random_seed: Optional[int] = Field(description="the random seed., it is optional", default=None)
    config_dict: Optional[dict] = Field(description="the configuration dictionary for the searching space, it is optional", default=None)
    progress_indicator_id: Optional[str] = Field(description="the progress indicator id, it is optional", default=None)
    fold_num: Optional[int] = Field(description="the number of folds for cross validation, it is optional", default=None)
    resampling_method: Optional[str] = Field(description="the resampling method for cross validation from {'rocv', 'block'}, it is optional", default=None)
    max_eval_time_mins: Optional[float] = Field(description="the maximum evaluation time in minutes, it is optional", default=None)
    early_stop: Optional[int] = Field(description="stop optimization progress when best pipeline is not updated for the give consecutive generations and 0 means there is no early stop, and it is optional", default=None)
    percentage: Optional[float] = Field(description="the percentage of the data to be used for training, it is optional", default=None)
    gap_num: Optional[int] = Field(description="the number of samples to exclude from the end of each train set before the test set, it is optional", default=None)
    connections: Union[Optional[dict], Optional[str]] = Field(description="the connections for the model, it is optional", default=None)
    alpha: Optional[float] = Field(description="the rejection probability in connection optimization, it is optional", default=None)
    delta: Optional[float] = Field(description="the minimum improvement in connection optimization, it is optional", default=None)
    top_k_connections: Optional[int] = Field(description="the number of top connections to keep in connection optimization, it is optional", default=None)
    top_k_pipelines: Optional[int] = Field(description="the number of top pipelines to keep in pipeline optimization, it is optional", default=None)
    fine_tune_pipline: Optional[bool] = Field(description="whether to fine tune the pipeline, it is optional", default=None)
    fine_tune_resource: Optional[int] = Field(description="the resource for fine tuning, it is optional", default=None)
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    group_key: str = Field(description="the name of the group column for multiple time series. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)
    categorical_variable: Union[Optional[str], Optional[list]] = Field(description="the categorical variable of the dataset, it is optional", default=None)
    background_size: Optional[int] = Field(description="the amount of background data in Kernel SHAP. Its value should not exceed the number of rows in the training data, it is optional", default=None)
    background_sampling_seed: Optional[int] = Field(description="the seed for sampling the background data in Kernel SHAP, it is optional", default=None)
    use_explain: Optional[bool] = Field(description="whether to use explain, it is optional", default=None)
    workload_class: Optional[str] = Field(description="the workload class for fitting the model, it is optional", default=None)


class ModelPredictInput(BaseModel):
    """Schema for predicting with a grouped massive automatic time-series model."""

    predict_table: str = Field(description="the table to predict. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model, it is optional", default=None)
    predict_schema: Optional[str] = Field(description="the schema of the predict_table, it is optional", default=None)
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    group_key: str = Field(description="the name of the group column for multiple time series. If not provided, ask the user. Do not guess.")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)
    show_explainer: Optional[bool] = Field(description="whether to show explainer, it is optional", default=None)


class ModelScoreInput(BaseModel):
    """Schema for scoring with a grouped massive automatic time-series model."""

    score_table: str = Field(description="the table to score. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model, it is optional", default=None)
    score_schema: Optional[str] = Field(description="the schema of the score_table, it is optional", default=None)
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    group_key: str = Field(description="the name of the group column for multiple time series. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)


class MassiveAutomaticTimeSeriesFitAndSave(BaseTool):
    """Fit a grouped massive automatic time-series model and save it to model storage."""

    name: str = "massive_automatic_timeseries_fit_and_save"
    description: str = "To fit an MassiveAutomaticTimeseries model or a timeseries per group(group_key Column) and save it in the model storage."
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = ModelFitInput
    return_direct: bool = False

    def __init__(self, connection_context: ConnectionContext, return_direct: bool = False) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        fit_table = kwargs.get("fit_table", None)
        if fit_table is None:
            return "Fit table is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        group_key = kwargs.get("group_key", None)
        if group_key is None:
            return "Group key is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "Endog is required"
        name = kwargs.get("name", None)
        if name is None:
            return "Model name is required"

        fit_schema = kwargs.get("fit_schema", None)
        version = kwargs.get("version", None)
        exog = kwargs.get("exog", None)
        categorical_variable = kwargs.get("categorical_variable", None)
        background_size = kwargs.get("background_size", None)
        background_sampling_seed = kwargs.get("background_sampling_seed", None)
        use_explain = kwargs.get("use_explain", None)
        workload_class = kwargs.get("workload_class", None)

        if not self.connection_context.has_table(fit_table, schema=fit_schema):
            return f"Table {fit_table} does not exist in the database."
        fit_df = self.connection_context.table(fit_table, schema=fit_schema)
        if key not in fit_df.columns:
            return f"Key {key} does not exist in the table {fit_table}."
        if group_key not in fit_df.columns:
            return f"Group key {group_key} does not exist in the table {fit_table}."

        auto_ts = MassiveAutomaticTimeSeries(
            scorings=kwargs.get("scorings", None),
            generations=kwargs.get("generations", None),
            population_size=kwargs.get("population_size", None),
            offspring_size=kwargs.get("offspring_size", None),
            elite_number=kwargs.get("elite_number", None),
            min_layer=kwargs.get("min_layer", None),
            max_layer=kwargs.get("max_layer", None),
            mutation_rate=kwargs.get("mutation_rate", None),
            crossover_rate=kwargs.get("crossover_rate", None),
            random_seed=kwargs.get("random_seed", None),
            config_dict=kwargs.get("config_dict", None),
            progress_indicator_id=kwargs.get("progress_indicator_id", None),
            fold_num=kwargs.get("fold_num", None),
            resampling_method=kwargs.get("resampling_method", None),
            max_eval_time_mins=kwargs.get("max_eval_time_mins", None),
            early_stop=kwargs.get("early_stop", None),
            percentage=kwargs.get("percentage", None),
            gap_num=kwargs.get("gap_num", None),
            connections=kwargs.get("connections", None),
            alpha=kwargs.get("alpha", None),
            delta=kwargs.get("delta", None),
            top_k_connections=kwargs.get("top_k_connections", None),
            top_k_pipelines=kwargs.get("top_k_pipelines", None),
            fine_tune_pipeline=kwargs.get("fine_tune_pipeline", kwargs.get("fine_tune_pipline", None)),
            fine_tune_resource=kwargs.get("fine_tune_resource", None),
        )
        if workload_class is not None:
            auto_ts.enable_workload_class(workload_class)
        else:
            auto_ts.disable_workload_class_check()
        auto_ts.fit(
            fit_df,
            key=key,
            group_key=group_key,
            endog=endog,
            exog=exog,
            categorical_variable=categorical_variable,
            background_size=background_size,
            background_sampling_seed=background_sampling_seed,
            use_explain=use_explain,
        )
        auto_ts.name = name
        ms = ModelStorage(connection_context=self.connection_context)
        auto_ts.version = generate_model_storage_version(ms, version, name)
        ms.save_model(model=auto_ts, if_exists="replace")
        return json.dumps(
            {
                "trained_table": fit_table,
                "model_storage_name": name,
                "model_storage_version": auto_ts.version,
            },
            cls=_CustomEncoder,
        )

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)

# End of module.




class MassiveAutomaticTimeSeriesLoadModelAndPredict(BaseTool):
    """Load a grouped massive automatic time-series model and predict."""

    name: str = "massive_automatic_timeseries_load_model_and_predict"
    description: str = "To load a model and do the prediction using massive automatic timeseries model."
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = ModelPredictInput
    return_direct: bool = False

    def __init__(self, connection_context: ConnectionContext, return_direct: bool = False) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        predict_table = kwargs.get("predict_table", None)
        if predict_table is None:
            return "Prediction table is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        group_key = kwargs.get("group_key", None)
        if group_key is None:
            return "Group key is required"
        name = kwargs.get("name", None)
        if name is None:
            return "Model name is required"

        version = kwargs.get("version", None)
        exog = kwargs.get("exog", None)
        show_explainer = kwargs.get("show_explainer", None)
        predict_schema = kwargs.get("predict_schema", None)
        if not self.connection_context.has_table(predict_table, schema=predict_schema):
            return json.dumps({"error": f"Table {predict_table} does not exist."}, cls=_CustomEncoder)

        predict_df = self.connection_context.table(predict_table, schema=predict_schema)
        if key not in predict_df.columns:
            return json.dumps({"error": f"Key {key} does not exist in table {predict_table}."}, cls=_CustomEncoder)
        if group_key not in predict_df.columns:
            return json.dumps({"error": f"Group key {group_key} does not exist in table {predict_table}."}, cls=_CustomEncoder)

        ms = ModelStorage(connection_context=self.connection_context)
        model = _prefer_massive_model_table_for_inference(ms.load_model(name, version))
        if getattr(model, "version", None) is not None:
            version = model.version

        original_columns = list(predict_df.columns)
        auto_repair_details = None
        prepared_df, kept_columns, missing_columns = build_repaired_predict_dataframe(
            predict_df,
            key=key,
            exog=exog,
            group_key=group_key,
        )
        if missing_columns:
            return format_predict_mismatch_diagnostic(
                predict_table=predict_table,
                predict_schema=predict_schema,
                original_columns=original_columns,
                kept_columns=kept_columns,
                missing_columns=missing_columns,
                key=key,
                exog=exog,
                group_key=group_key,
                original_error="Required inference columns are missing before prediction starts.",
            )
        if kept_columns != original_columns:
            auto_repair_details = {
                "auto_repaired_predict_input": True,
                "predict_table_columns_before_repair": original_columns,
                "predict_table_columns_used_for_prediction": kept_columns,
            }

        try:
            model.predict(
                data=prepared_df,
                key=key,
                group_key=group_key,
                exog=exog,
                show_explainer=show_explainer,
            )
        except Exception as exc:  # pylint: disable=broad-except
            if not is_predict_feature_mismatch_error(exc):
                return json.dumps({"error": f"Prediction failed: {exc}"}, cls=_CustomEncoder)

            repaired_df, kept_columns, missing_columns = build_repaired_predict_dataframe(
                predict_df,
                key=key,
                exog=exog,
                group_key=group_key,
            )
            if missing_columns or kept_columns == original_columns:
                return format_predict_mismatch_diagnostic(
                    predict_table=predict_table,
                    predict_schema=predict_schema,
                    original_columns=original_columns,
                    kept_columns=kept_columns,
                    missing_columns=missing_columns,
                    key=key,
                    exog=exog,
                    group_key=group_key,
                    original_error=str(exc),
                )
            try:
                model.predict(
                    data=repaired_df,
                    key=key,
                    group_key=group_key,
                    exog=exog,
                    show_explainer=show_explainer,
                )
            except Exception as retry_exc:  # pylint: disable=broad-except
                return format_predict_mismatch_diagnostic(
                    predict_table=predict_table,
                    predict_schema=predict_schema,
                    original_columns=original_columns,
                    kept_columns=kept_columns,
                    missing_columns=missing_columns,
                    key=key,
                    exog=exog,
                    group_key=group_key,
                    original_error=str(retry_exc),
                )
            auto_repair_details = {
                "auto_repaired_predict_input": True,
                "predict_table_columns_before_repair": original_columns,
                "predict_table_columns_used_for_prediction": kept_columns,
            }

        ms.save_model(model=model, if_exists="replace_meta")
        # Uppercase user-supplied name fragments so the created table aligns with
        # HANA's default identifier folding on unquoted references.
        safe_name = _hana_safe_identifier(name)
        safe_predict_table = _hana_safe_identifier(predict_table)
        safe_predict_schema = _hana_safe_identifier(predict_schema) if predict_schema else None
        predicted_results = (
            f"PREDICT_RESULT_{safe_predict_table}_{safe_name}_{version}"
            if safe_predict_schema is None
            else f"PREDICT_RESULT_{safe_predict_schema}_{safe_predict_table}_{safe_name}_{version}"
        )
        self.connection_context.table(model._predict_output_table_names[0]).smart_save(predicted_results, force=True)
        stats = self.connection_context.table(model._predict_output_table_names[1]).collect()
        outputs = {
            "input_predict_table": predict_table,
            "input_predict_schema": predict_schema,
            "model_storage_name": name,
            "model_storage_version": version,
            "predicted_results_table": predicted_results,
        }
        for _, row in stats.iterrows():
            outputs[row[stats.columns[0]]] = row[stats.columns[1]]
        if auto_repair_details:
            outputs.update(auto_repair_details)
        return json.dumps(outputs, cls=_CustomEncoder)

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class MassiveAutomaticTimeSeriesLoadModelAndScore(BaseTool):
    """Load a grouped massive automatic time-series model and score."""

    name: str = "massive_automatic_timeseries_load_model_and_score"
    description: str = "To load a model and do the scoring for massive automatic timeseries."
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = ModelScoreInput
    return_direct: bool = False

    def __init__(self, connection_context: ConnectionContext, return_direct: bool = False) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        score_table = kwargs.get("score_table", None)
        if score_table is None:
            return "Score table is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        group_key = kwargs.get("group_key", None)
        if group_key is None:
            return "Group key is required"
        name = kwargs.get("name", None)
        if name is None:
            return "Model name is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "Endog is required"

        version = kwargs.get("version", None)
        exog = kwargs.get("exog", None)
        score_schema = kwargs.get("score_schema", None)
        if not self.connection_context.has_table(score_table, schema=score_schema):
            return json.dumps({"error": f"Table {score_table} does not exist."}, cls=_CustomEncoder)

        score_df = self.connection_context.table(score_table, schema=score_schema)
        if key not in score_df.columns:
            return json.dumps({"error": f"Key {key} does not exist in table {score_table}."}, cls=_CustomEncoder)
        if group_key not in score_df.columns:
            return json.dumps({"error": f"Group key {group_key} does not exist in table {score_table}."}, cls=_CustomEncoder)

        ms = ModelStorage(connection_context=self.connection_context)
        model = _prefer_massive_model_table_for_inference(ms.load_model(name, version))
        if getattr(model, "version", None) is not None:
            version = model.version
        model.score(data=score_df, key=key, group_key=group_key, endog=endog, exog=exog)
        ms.save_model(model=model, if_exists="replace_meta")
        scored_results = (
            f"SCORE_RESULT_{score_table}_{name}_{version}"
            if score_schema is None
            else f"SCORE_RESULT_{score_schema}_{score_table}_{name}_{version}"
        )
        self.connection_context.table(model._score_output_table_names[0]).smart_save(scored_results, force=True)
        stats = self.connection_context.table(model._score_output_table_names[1]).collect()
        outputs = {"scored_results_table": scored_results}
        for _, row in stats.iterrows():
            outputs[row[stats.columns[0]]] = row[stats.columns[1]]
        return json.dumps(outputs, cls=_CustomEncoder)

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)
