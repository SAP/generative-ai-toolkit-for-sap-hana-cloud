"""
This module contains the tools for automatic timeseries.
"""

import json
import logging
from typing import Optional, Type, Union
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.model_storage import ModelStorage
from hana_ml.algorithms.pal.auto_ml import AutomaticTimeSeries

logger = logging.getLogger(__name__)

class ModelFitInput(BaseModel):
    """
    The schema of the inputs for fitting the model.
    """
    fit_table: str = Field(description="the table to fit the model. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model in model storage. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model in model storage, it is optional", default=None)
    # init args
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
    # fit args
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)
    categorical_variable: Union[Optional[str], Optional[list]] = Field(description="the categorical variable of the dataset, it is optional", default=None)
    background_size: Optional[int] = Field(description="the amount of background data in Kernel SHAP. Its value should not exceed the number of rows in the training data, it is optional", default=None)
    background_sampling_seed: Optional[int] = Field(description="the seed for sampling the background data in Kernel SHAP, it is optional", default=None)
    use_explain: Optional[bool] = Field(description="whether to use explain, it is optional", default=None)
    workload_class: Optional[str] = Field(description="the workload class for fitting the model, it is optional", default=None)

class ModelPredictInput(BaseModel):
    """
    The schema of the inputs for predicting the model.
    """
    # init args
    predict_table: str = Field(description="the table to predict. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model, it is optional", default=None)
    # fit args
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)
    show_explainer: Optional[bool] = Field(description="whether to show explainer, it is optional", default=None)
    predict_args: Optional[dict] = Field(description="the arguments for prediction, it is optional", default=None)

class ModelScoreInput(BaseModel):
    """
    The schema of the inputs for scoring the model.
    """
    score_table: str = Field(description="the table to score. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model, it is optional", default=None)
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)
    predict_args: Optional[dict] = Field(description="the arguments for prediction, it is optional", default=None)

class AutomaticTimeSeriesFitAndSave(BaseTool):
    """
    This tool fits a time series model and saves it in the model storage.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The JSON string of the trained table, model storage name, and model storage version.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - fit_table
                  - The table to fit the model. If not provided, ask the user. Do not guess.
                * - name
                  - The name of the model in model storage. If not provided, ask the user. Do not guess.
                * - version
                  - The version of the model in model storage, it is optional
                * - scorings
                  - The scorings for the model, e.g. {'MAE':-1.0, 'EVAR':1.0} and it supports EVAR, MAE, MAPE, MAX_ERROR, MSE, R2, RMSE, WMAPE, LAYERS, SPEC, TIME, and it is optional
                * - generations
                  - The number of iterations of the pipeline optimization., it is optional
                * - population_size
                  - The number of individuals in the population., it is optional
                * - offspring_size
                  - The number of children to produce at each generation., it is optional
                * - elite_number
                  - The number of the best individuals to select for the next generation., it is optional
                * - min_layer
                  - The minimum number of layers in the pipeline., it is optional
                * - max_layer
                  - The maximum number of layers in the pipeline., it is optional
                * - mutation_rate
                  - The mutation rate., it is optional
                * - crossover_rate
                  - The crossover rate., it is optional
                * - random_seed
                  - The random seed., it is optional
                * - config_dict
                  - The configuration dictionary for the searching space, it is optional
                * - progress_indicator_id
                  - The progress indicator id, it is optional
                * - fold_num
                  - The number of folds for cross validation, it is optional
                * - resampling_method
                  - The resampling method for cross validation from {'rocv', 'block'}, it is optional
                * - max_eval_time_mins
                  - The maximum evaluation time in minutes, it is optional
                * - early_stop
                  - Stop optimization progress when the best pipeline is not updated for the give consecutive generations and 0 means there is no early stop, and it is optional
                * - percentage
                  - The percentage of the data to be used for training, it is optional
                * - gap_num
                  - The number of samples to exclude from the end of each train set before the test set, it is optional
                * - connections
                  - The connections for the model, it is optional
                * - alpha
                  - The rejection probability in connection optimization, it is optional
                * - delta
                  - The minimum improvement in connection optimization, it is optional
                * - top_k_connections
                  - The number of top connections to keep in connection optimization, it is optional
                * - top_k_pipelines
                  - The number of top pipelines to keep in pipeline optimization, it is optional
                * - fine_tune_pipline
                  - Whether to fine tune the pipeline, it is optional
                * - fine_tune_resource
                  - The resource for fine tuning, it is optional
                * - key
                  - The key of the dataset. If not provided, ask the user. Do not guess.
                * - endog
                  - The endog of the dataset. If not provided, ask the user. Do not guess.
                * - exog
                  - The exog of the dataset, it is optional
                * - categorical_variable
                  - The categorical variable of the dataset, it is optional
                * - background_size
                  - The amount of background data in Kernel SHAP. Its value should not exceed the number of rows in the training data, it is optional
                * - background_sampling_seed
                  - The seed for sampling the background data in Kernel SHAP, it is optional
                * - use_explain
                  - Whether to use explain, it is optional
                * - workload_class
                  - The workload class for fitting the model, it is optional
    """
    name: str = "automatic_timeseries_fit_and_save"
    """Name of the tool."""
    description: str = "To fit an AutomaticTimeseries model and save it in the model storage."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ModelFitInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(
        self,
        fit_table: str,
        key: str,
        endog: str,
        name: str,
        version: Optional[str]=None,
        scorings: Optional[dict]=None,
        generations: Optional[int]=None,
        population_size: Optional[int]=None,
        offspring_size: Optional[int]=None,
        elite_number: Optional[int]=None,
        min_layer: Optional[int]=None,
        max_layer: Optional[int]=None,
        mutation_rate: Optional[float]=None,
        crossover_rate: Optional[float]=None,
        random_seed: Optional[int]=None,
        config_dict: Optional[dict]=None,
        progress_indicator_id: Optional[str]=None,
        fold_num: Optional[int]=None,
        resampling_method: Optional[str]=None,
        max_eval_time_mins: Optional[float]=None,
        early_stop: Optional[int]=None,
        percentage: Optional[float]=None,
        gap_num: Optional[int]=None,
        connections: Union[Optional[dict], Optional[str]]=None,
        alpha: Optional[float]=None,
        delta: Optional[float]=None,
        top_k_connections: Optional[int]=None,
        top_k_pipelines: Optional[int]=None,
        fine_tune_pipeline: Optional[bool]=None,
        fine_tune_resource: Optional[int]=None,
        exog: Union[Optional[str], Optional[list]]=None,
        categorical_variable: Union[Optional[str], Optional[list]]=None,
        background_size: Optional[int]=None,
        background_sampling_seed: Optional[int]=None,
        use_explain: Optional[bool]=None,
        workload_class: Optional[str]=None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        auto_ts = AutomaticTimeSeries(
            scorings=scorings,
            generations=generations,
            population_size=population_size,
            offspring_size=offspring_size,
            elite_number=elite_number,
            min_layer=min_layer,
            max_layer=max_layer,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            random_seed=random_seed,
            config_dict=config_dict,
            progress_indicator_id=progress_indicator_id,
            fold_num=fold_num,
            resampling_method=resampling_method,
            max_eval_time_mins=max_eval_time_mins,
            early_stop=early_stop,
            percentage=percentage,
            gap_num=gap_num,
            connections=connections,
            alpha=alpha,
            delta=delta,
            top_k_connections=top_k_connections,
            top_k_pipelines=top_k_pipelines,
            fine_tune_pipeline=fine_tune_pipeline,
            fine_tune_resource=fine_tune_resource,
        )
        if workload_class is not None:
            auto_ts.enable_workload_class(workload_class)
        else:
            auto_ts.disable_workload_class_check()
        auto_ts.fit(self.connection_context.table(fit_table),
                    key=key,
                    endog=endog,
                    exog=exog,
                    categorical_variable=categorical_variable,
                    background_size=background_size,
                    background_sampling_seed=background_sampling_seed,
                    use_explain=use_explain)
        auto_ts.name = name
        ms = ModelStorage(connection_context=self.connection_context)
        if version is None:
            version = ms._get_new_version_no(name)
            if version is None:
                version = 1
            else:
                version = int(version)
        auto_ts.version = version
        ms.save_model(model=auto_ts, if_exists='replace')
        return json.dumps({"trained_table": fit_table, "model_storage_name": name, "model_storage_version": version})

    async def _arun(
        self,
        fit_table: str,
        key: str,
        endog: str,
        name: str,
        version: Optional[str]=None,
        scorings: Optional[dict]=None,
        generations: Optional[int]=None,
        population_size: Optional[int]=None,
        offspring_size: Optional[int]=None,
        elite_number: Optional[int]=None,
        min_layer: Optional[int]=None,
        max_layer: Optional[int]=None,
        mutation_rate: Optional[float]=None,
        crossover_rate: Optional[float]=None,
        random_seed: Optional[int]=None,
        config_dict: Optional[dict]=None,
        progress_indicator_id: Optional[str]=None,
        fold_num: Optional[int]=None,
        resampling_method: Optional[str]=None,
        max_eval_time_mins: Optional[float]=None,
        early_stop: Optional[int]=None,
        percentage: Optional[float]=None,
        gap_num: Optional[int]=None,
        connections: Union[Optional[dict], Optional[str]]=None,
        alpha: Optional[float]=None,
        delta: Optional[float]=None,
        top_k_connections: Optional[int]=None,
        top_k_pipelines: Optional[int]=None,
        fine_tune_pipeline: Optional[bool]=None,
        fine_tune_resource: Optional[int]=None,
        exog: Union[Optional[str], Optional[list]]=None,
        categorical_variable: Union[Optional[str], Optional[list]]=None,
        background_size: Optional[int]=None,
        background_sampling_seed: Optional[int]=None,
        use_explain: Optional[bool]=None,
        workload_class: Optional[str]=None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        self._run(
            fit_table=fit_table,
            key=key,
            endog=endog,
            name=name,
            version=version,
            scorings=scorings,
            generations=generations,
            population_size=population_size,
            offspring_size=offspring_size,
            elite_number=elite_number,
            min_layer=min_layer,
            max_layer=max_layer,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            random_seed=random_seed,
            config_dict=config_dict,
            progress_indicator_id=progress_indicator_id,
            fold_num=fold_num,
            resampling_method=resampling_method,
            max_eval_time_mins=max_eval_time_mins,
            early_stop=early_stop,
            percentage=percentage,
            gap_num=gap_num,
            connections=connections,
            alpha=alpha,
            delta=delta,
            top_k_connections=top_k_connections,
            top_k_pipelines=top_k_pipelines,
            fine_tune_pipeline=fine_tune_pipeline,
            fine_tune_resource=fine_tune_resource,
            exog=exog,
            categorical_variable=categorical_variable,
            background_size=background_size,
            background_sampling_seed=background_sampling_seed,
            use_explain=use_explain,
            workload_class=workload_class,
            run_manager=run_manager
        )

class AutomaticTimeseriesLoadModelAndPredict(BaseTool):
    """
    This tool load model from model storage and do the prediction.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The JSON string of the predicted results table and the statistics.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - predict_table
                  - The table to predict. If not provided, ask the user. Do not guess.
                * - name
                  - The name of the model. If not provided, ask the user. Do not guess.
                * - version
                  - The version of the model, it is optional
                * - key
                  - The key of the dataset. If not provided, ask the user. Do not guess.
                * - exog
                  - The exog of the dataset, it is optional
                * - show_explainer
                  - Whether to show explainer, it is optional
                * - predict_args
                  - The arguments for prediction, it is optional
    """
    name: str = "automatic_timeseries_load_model_and_predict"
    """Name of the tool."""
    description: str = "To load a model and do the prediction using automatic timeseries model."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ModelPredictInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(
        self,
        predict_table: str,
        key: str,
        name: str,
        version: str=None,
        exog: Union[str, list]=None,
        show_explainer: bool=None,
        predict_args: dict=None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        ms = ModelStorage(connection_context=self.connection_context)
        model = ms.load_model(name, version)
        if hasattr(model, 'version'):
            if model.version is not None:
                version = model.version
        model.predict(data=self.connection_context.table(predict_table),
                      key=key,
                      exog=exog,
                      show_explainer=show_explainer,
                      predict_args=predict_args)
        ms.save_model(model=model, if_exists='replace_meta')
        predicted_results = f"{name}_{version}_PREDICTED_RESULTS"
        self.connection_context.table(model._predict_output_table_names[0]).save(predicted_results, force=True)
        stats = self.connection_context.table(model._predict_output_table_names[1]).collect()
        outputs = {"predicted_results_table": predicted_results}
        for _, row in stats.iterrows():
            outputs[row[stats.columns[0]]] = row[stats.columns[1]]
        return json.dumps(outputs)

    async def _arun(
        self,
        predict_table: str,
        key: str,
        name: str,
        version: str=None,
        exog: Union[str, list]=None,
        show_explainer: bool=None,
        predict_args: dict=None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        self._run(
            predict_table=predict_table,
            key=key,
            name=name,
            version=version,
            exog=exog,
            show_explainer=show_explainer,
            predict_args=predict_args,
            run_manager=run_manager
        )

class AutomaticTimeseriesLoadModelAndScore(BaseTool):
    """
    This tool load model from model storage and do the scoring.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The JSON string of the scored results table and the statistics.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - score_table
                  - The table to score. If not provided, ask the user. Do not guess.
                * - name
                  - The name of the model. If not provided, ask the user. Do not guess.
                * - version
                  - The version of the model, it is optional
                * - key
                  - The key of the dataset. If not provided, ask the user. Do not guess.
                * - endog
                  - The endog of the dataset. If not provided, ask the user. Do not guess.
                * - exog
                  - The exog of the dataset, it is optional
                * - predict_args
                  - The arguments for prediction, it is optional
    """
    name: str = "automatic_timeseries_load_model_and_score"
    """Name of the tool."""
    description: str = "To load a model and do the scoring for automatic timeseries."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ModelScoreInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(
        self,
        score_table: str,
        key: str,
        name: str,
        version: str=None,
        endog: str=None,
        exog: Union[str, list]=None,
        predict_args: dict=None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        ms = ModelStorage(connection_context=self.connection_context)
        model = ms.load_model(name, version)
        if hasattr(model, 'version'):
            if model.version is not None:
                version = model.version
        model.score(data=self.connection_context.table(score_table),
                    key=key,
                    endog=endog,
                    exog=exog,
                    predict_args=predict_args)
        ms.save_model(model=model, if_exists='replace_meta')
        scored_results = f"{name}_{version}_SCORED_RESULTS"
        self.connection_context.table(model._score_output_table_names[0]).save(scored_results, force=True)
        stats = self.connection_context.table(model._score_output_table_names[1]).collect()
        outputs = {"scored_results_table": scored_results}
        for _, row in stats.iterrows():
            outputs[row[stats.columns[0]]] = row[stats.columns[1]]
        return json.dumps(outputs)

    async def _arun(
        self,
        score_table: str,
        key: str,
        name: str,
        version: str=None,
        endog: str=None,
        exog: Union[str, list]=None,
        predict_args: dict=None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        self._run(
            score_table=score_table,
            key=key,
            name=name,
            version=version,
            endog=endog,
            exog=exog,
            predict_args=predict_args,
            run_manager=run_manager
        )
