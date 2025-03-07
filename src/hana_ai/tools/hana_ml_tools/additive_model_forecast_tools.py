"""
This module contains the tools for additive model forecast.
"""


import json
import logging
from typing import Optional, Type, Union
from langchain.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.model_storage import ModelStorage
from hana_ml.algorithms.pal.tsa.additive_model_forecast import AdditiveModelForecast

logger = logging.getLogger(__name__)

class ModelFitInput(BaseModel):
    fit_table: str = Field(description="the table to fit the model. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model in model storage. If not provided, ask the user. Do not guess.")
    version: Optional[str] = Field(description="the version of the model in model storage, it is optional")
    # init args
    growth: Optional[str] = Field(description="the growth of the model chosen from {'linear', 'logistic'}, it is optional")
    logistic_growth_capacity: Optional[float] = Field(description="the logistic growth capacity of the model only valid when growth is 'logistic', it is optional")
    seasonality_mode: Optional[str] = Field(description="the seasonality mode of the model chosen from {'additive', 'multiplicative'}, it is optional")
    seasonality: Optional[str] = Field(description="adds seasonality to the model in a json format such that each str is in json format '{\"NAME\": \"MONTHLY\", \"PERIOD\":30, \"FOURIER_ORDER\":5 }', it is optional")
    num_changepoints: Optional[int] = Field(description="the number of changepoints in the model, it is optional")
    changepoint_range: Optional[float] = Field(description="the proportion of history in which trend changepoints will be estimated, it is optional")
    regressor: Optional[list] = Field(description="specifies the regressor in a list of json such that ['{\"NAME\": \"X1\", \"PRIOR_SCALE\":4, \"MODE\": \"additive\" }'], it is optional")
    changepoints: Optional[list] = Field(description="specifies a list of changepoints in the format of timestamp, such as ['2019-01-01 00:00:00, '2019-02-04 00:00:00'], it is optional")
    yearly_seasonality: Optional[str] = Field(description="Specifies whether or not to fit yearly seasonality chosen from {'auto', 'false', 'true'}, it is optional")
    weekly_seasonality: Optional[str] = Field(description="Specifies whether or not to fit weekly seasonality chosen from {'auto', 'false', 'true'}, it is optional")
    daily_seasonality: Optional[str] = Field(description="Specifies whether or not to fit daily seasonality chosen from {'auto', 'false', 'true'}, it is optional")
    seasonality_prior_scale: Optional[float] = Field(description="the parameter modulating the strength of the seasonality model, it is optional")
    holiday_prior_scale: Optional[float] = Field(description="the parameter modulating the strength of the holiday model, it is optional")
    changepoint_prior_scale: Optional[float] = Field(description="the parameter modulating the flexibility of the automatic changepoint selection, it is optional")
    # fit args
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: Optional[str] = Field(description="the endog of the dataset, it is optional")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional")
    holiday_table: Optional[str] = Field(description="the table of the holiday, it is optional")
    categorical_variable: Union[Optional[str], Optional[list]] = Field(description="the categorical variable of the dataset, it is optional")

class ModelPredictInput(BaseModel):
    predict_table: str = Field(description="the table to predict. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model. If not provided, ask the user. Do not guess.")
    version: Optional[str] = Field(description="the version of the model, it is optional")
    # predict args
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional")
    logistic_growth_capacity: Optional[float] = Field(description="the logistic growth capacity of the model only valid when growth is 'logistic', it is optional")
    interval_width: Optional[float] = Field(description="the width of the uncertainty interval, it is optional")
    uncertainty_samples: Optional[int] = Field(description="the number of simulated draws used to estimate uncertainty intervals, it is optional")
    show_explainer: Optional[bool] = Field(description="whether to show explainer, it is optional")
    decompose_seasonality: Optional[bool] = Field(description="whether to decompose seasonality only valid when show_explainer is True, it is optional")
    decompose_holiday: Optional[bool] = Field(description="whether to decompose holiday only valid when show_explainer is True, it is optional")

class AdditiveModelForecastFitAndSave(BaseTool):
    """
    This tool is used to fit and predict the additive model forecast.
    """
    name: str = "additive_model_forecast_fit_and_save"
    """Name of the tool."""
    description: str = "To fit the additive model forecast and save the model in model storage."
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
        name: str,
        version: Optional[str] = None,
        growth: Optional[str] = None,
        logistic_growth_capacity: Optional[float] = None,
        seasonality_mode: Optional[str] = None,
        seasonality: Optional[str] = None,
        num_changepoints: Optional[int] = None,
        changepoint_range: Optional[float] = None,
        regressor: Optional[list] = None,
        changepoints: Optional[list] = None,
        yearly_seasonality: Optional[str] = None,
        weekly_seasonality: Optional[str] = None,
        daily_seasonality: Optional[str] = None,
        seasonality_prior_scale: Optional[float] = None,
        holiday_prior_scale: Optional[float] = None,
        changepoint_prior_scale: Optional[float] = None,
        endog: Optional[str] = None,
        exog: Union[Optional[str], Optional[list]] = None,
        holiday_table: Optional[str] = None,
        categorical_variable: Union[Optional[str], Optional[list]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        ms = ModelStorage(connection_context=self.connection_context)
        amf = AdditiveModelForecast(
            growth=growth,
            logistic_growth_capacity=logistic_growth_capacity,
            seasonality_mode=seasonality_mode,
            seasonality=seasonality,
            num_changepoints=num_changepoints,
            changepoint_range=changepoint_range,
            regressor=regressor,
            changepoints=changepoints,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            holiday_prior_scale=holiday_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale)
        holiday_df = None
        if holiday_table is not None:
            holiday_df = self.connection_context.table(holiday_table)
        amf.fit(data=self.connection_context.table(fit_table),
                key=key,
                endog=endog,
                exog=exog,
                holiday=holiday_df,
                categorical_variable=categorical_variable)
        amf.name = name
        amf.version = version
        if version is not None:
            ms.save_model(model=amf, if_exists='replace')
        else:
            ms.save_model(model=amf)
        if version is None:
            version = int(ms._get_last_version_no(name))
        return json.dumps({"trained_table": fit_table, "model_storage_name": name, "model_storage_version": version})

    async def _arun(
        self,
        fit_table: str,
        key: str,
        name: str,
        version: Optional[str] = None,
        growth: Optional[str] = None,
        logistic_growth_capacity: Optional[float] = None,
        seasonality_mode: Optional[str] = None,
        seasonality: Optional[str] = None,
        num_changepoints: Optional[int] = None,
        changepoint_range: Optional[float] = None,
        regressor: Optional[list] = None,
        changepoints: Optional[list] = None,
        yearly_seasonality: Optional[str] = None,
        weekly_seasonality: Optional[str] = None,
        daily_seasonality: Optional[str] = None,
        seasonality_prior_scale: Optional[float] = None,
        holiday_prior_scale: Optional[float] = None,
        changepoint_prior_scale: Optional[float] = None,
        endog: Optional[str] = None,
        exog: Union[Optional[str], Optional[list]] = None,
        holiday_table: Optional[str] = None,
        categorical_variable: Union[Optional[str], Optional[list]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(
            fit_table=fit_table,
            key=key,
            name=name,
            version=version,
            growth=growth,
            logistic_growth_capacity=logistic_growth_capacity,
            seasonality_mode=seasonality_mode,
            seasonality=seasonality,
            num_changepoints=num_changepoints,
            changepoint_range=changepoint_range,
            regressor=regressor,
            changepoints=changepoints,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            holiday_prior_scale=holiday_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            endog=endog,
            exog=exog,
            holiday_table=holiday_table,
            categorical_variable=categorical_variable,
            run_manager=run_manager
        )

class AdditiveModelForecastLoadModelAndPredict(BaseTool):
    """
    This tool is used to load the additive model forecast from model storage and predict.
    """
    name: str = "additive_model_forecast_load_model_and_predict"
    """Name of the tool."""
    description: str = "To load the additive model forecast from model storage and predict."
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
        version: Optional[str] = None,
        exog: Union[Optional[str], Optional[list]] = None,
        logistic_growth_capacity: Optional[float] = None,
        interval_width: Optional[float] = None,
        uncertainty_samples: Optional[int] = None,
        show_explainer: Optional[bool] = None,
        decompose_seasonality: Optional[bool] = None,
        decompose_holiday: Optional[bool] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        ms = ModelStorage(connection_context=self.connection_context)
        model = ms.load_model(name=name, version=version)
        model.predict(data=self.connection_context.table(predict_table),
                      key=key,
                      exog=exog,
                      logistic_growth_capacity=logistic_growth_capacity,
                      interval_width=interval_width,
                      uncertainty_samples=uncertainty_samples,
                      show_explainer=show_explainer,
                      decompose_seasonality=decompose_seasonality,
                      decompose_holiday=decompose_holiday)
        ms.save_model(model=model, if_exists='replace_meta')
        predicted_results = [f"{name}_{version}_PREDICTED_RESULT"]
        self.connection_context.table(model._predict_output_table_names[0]).save(predicted_results[0])
        if show_explainer is True:
            predicted_results.append(
                f"{name}_{version}_DECOMPOSED_AND_REASON_CODE"
            )
            self.connection_context.table(model._predict_output_table_names[1]).save(predicted_results[1])
            return json.dumps({"predicted_results_table": predicted_results[0], "decomposed_and_reason_code_table": predicted_results[1]})
        return json.dumps({"predicted_results_table": predicted_results[0]})

    async def _arun(
        self,
        predict_table: str,
        key: str,
        name: str,
        version: Optional[str] = None,
        exog: Union[Optional[str], Optional[list]] = None,
        logistic_growth_capacity: Optional[float] = None,
        interval_width: Optional[float] = None,
        uncertainty_samples: Optional[int] = None,
        show_explainer: Optional[bool] = None,
        decompose_seasonality: Optional[bool] = None,
        decompose_holiday: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(
            predict_table=predict_table,
            key=key,
            name=name,
            version=version,
            exog=exog,
            logistic_growth_capacity=logistic_growth_capacity,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples,
            show_explainer=show_explainer,
            decompose_seasonality=decompose_seasonality,
            decompose_holiday=decompose_holiday,
            run_manager=run_manager
        )
