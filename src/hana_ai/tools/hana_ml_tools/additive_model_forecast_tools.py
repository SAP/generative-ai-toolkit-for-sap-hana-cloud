"""
This module contains the tools for additive model forecast.

The following class are available:

    * :class `AdditiveModelForecastFitAndSave`
    * :class `AdditiveModelForecastLoadModelAndPredict`
  * :class `MassiveAdditiveModelForecastFitAndSave`
  * :class `MassiveAdditiveModelForecastLoadModelAndPredict`
"""
#pylint: disable=too-many-return-statements

import json
import logging
from typing import Optional, Type, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.model_storage import ModelStorage
from hana_ml.algorithms.pal.tsa.additive_model_forecast import AdditiveModelForecast

from hana_ai.tools.hana_ml_tools.utility import (
  _CustomEncoder,
  _hana_safe_identifier,
  build_repaired_predict_dataframe,
  format_predict_mismatch_diagnostic,
  generate_model_storage_version,
  is_predict_feature_mismatch_error,
)

logger = logging.getLogger(__name__)

def _guess_fourier_order(period: int) -> int:
    # Calculate base value and round to nearest integer
    base_value = round(float(period) / 36.5)

    # Apply bounds: minimum 3, maximum 10
    return max(3, min(10, base_value))

class ModelFitInput(BaseModel):
    """
    This class is used to define the schema of the inputs for fitting the model
    """
    fit_table: str = Field(description="the table to fit the model. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model in model storage. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model in model storage, it is optional", default=None)
    fit_schema: Optional[str] = Field(description="the schema of the fit_table, it is optional", default=None)
    # init args
    growth: Optional[str] = Field(description="the growth of the model chosen from {'linear', 'logistic'}, it is optional", default=None)
    logistic_growth_capacity: Optional[float] = Field(description="the logistic growth capacity of the model only valid when growth is 'logistic', it is optional", default=None)
    seasonality_mode: Optional[str] = Field(description="the seasonality mode of the model chosen from {'additive', 'multiplicative'}, it is optional", default=None)
    #seasonality: Optional[str] = Field(description="adds seasonality to the model in a json format such that each str is in json format '{\"NAME\": \"MONTHLY\", \"PERIOD\":30, \"FOURIER_ORDER\":5 }', it is optional", default=None)
    period: Union[Optional[int], Optional[list]] = Field(description="the period of the seasonality and it could also be a list of periods, it is optional", default=None)
    num_changepoints: Optional[int] = Field(description="the number of changepoints in the model, it is optional", default=None)
    changepoint_range: Optional[float] = Field(description="the proportion of history in which trend changepoints will be estimated, it is optional", default=None)
    regressor: Optional[list] = Field(description="specifies the regressor in a list of json such that ['{\"NAME\": \"X1\", \"PRIOR_SCALE\":4, \"MODE\": \"additive\" }'], it is optional", default=None)
    changepoints: Optional[list] = Field(description="specifies a list of changepoints in the format of timestamp, such as ['2019-01-01 00:00:00, '2019-02-04 00:00:00'], it is optional", default=None)
    yearly_seasonality: Optional[str] = Field(description="Specifies whether or not to fit yearly seasonality chosen from {'auto', 'false', 'true'}, it is optional", default=None)
    weekly_seasonality: Optional[str] = Field(description="Specifies whether or not to fit weekly seasonality chosen from {'auto', 'false', 'true'}, it is optional", default=None)
    daily_seasonality: Optional[str] = Field(description="Specifies whether or not to fit daily seasonality chosen from {'auto', 'false', 'true'}, it is optional", default=None)
    seasonality_prior_scale: Optional[float] = Field(description="the parameter modulating the strength of the seasonality model, it is optional", default=None)
    holiday_prior_scale: Optional[float] = Field(description="the parameter modulating the strength of the holiday model, it is optional", default=None)
    changepoint_prior_scale: Optional[float] = Field(description="the parameter modulating the flexibility of the automatic changepoint selection, it is optional", default=None)
    # fit args
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: Optional[str] = Field(description="the endog of the dataset, it is optional", default=None)
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)
    holiday_table: Optional[str] = Field(description="the table of the holiday, it is optional", default=None)
    holiday_schema: Optional[str] = Field(description="the schema of the holiday_table, it is optional", default=None)
    categorical_variable: Union[Optional[str], Optional[list]] = Field(description="the categorical variable of the dataset, it is optional", default=None)

class ModelPredictInput(BaseModel):
    """
    This class is used to define the schema of the inputs for predicting the model
    """
    predict_table: str = Field(description="the table to predict. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model, it is optional", default=None)
    predict_schema: Optional[str] = Field(description="the schema of the predict_table, it is optional", default=None)
    # predict args
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)
    logistic_growth_capacity: Optional[float] = Field(description="the logistic growth capacity of the model only valid when growth is 'logistic', it is optional", default=None)
    interval_width: Optional[float] = Field(description="the width of the uncertainty interval, it is optional", default=None)
    uncertainty_samples: Optional[int] = Field(description="the number of simulated draws used to estimate uncertainty intervals, it is optional", default=None)
    show_explainer: Optional[bool] = Field(description="whether to show explainer, it is optional", default=None)
    decompose_seasonality: Optional[bool] = Field(description="whether to decompose seasonality only valid when show_explainer is True, it is optional", default=None)
    decompose_holiday: Optional[bool] = Field(description="whether to decompose holiday only valid when show_explainer is True, it is optional", default=None)
    add_placeholder: Optional[bool] = Field(description="whether to add placeholder for the endog column and it is set True by default, it is optional", default=True)


class MassiveModelFitInput(BaseModel):
    """This class is used to define the schema of the inputs for fitting a grouped additive model."""

    fit_table: str = Field(description="the table to fit the model. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model in model storage. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model in model storage, it is optional", default=None)
    fit_schema: Optional[str] = Field(description="the schema of the fit_table, it is optional", default=None)
    growth: Optional[str] = Field(description="the growth of the model chosen from {'linear', 'logistic'}, it is optional", default=None)
    logistic_growth_capacity: Optional[float] = Field(description="the logistic growth capacity of the model only valid when growth is 'logistic', it is optional", default=None)
    seasonality_mode: Optional[str] = Field(description="the seasonality mode of the model chosen from {'additive', 'multiplicative'}, it is optional", default=None)
    period: Union[Optional[int], Optional[list]] = Field(description="the period of the seasonality and it could also be a list of periods, it is optional", default=None)
    num_changepoints: Optional[int] = Field(description="the number of changepoints in the model, it is optional", default=None)
    changepoint_range: Optional[float] = Field(description="the proportion of history in which trend changepoints will be estimated, it is optional", default=None)
    regressor: Optional[list] = Field(description="specifies the regressor in a list of json such that ['{\"NAME\": \"X1\", \"PRIOR_SCALE\":4, \"MODE\": \"additive\" }'], it is optional", default=None)
    changepoints: Optional[list] = Field(description="specifies a list of changepoints in the format of timestamp, such as ['2019-01-01 00:00:00, '2019-02-04 00:00:00'], it is optional", default=None)
    yearly_seasonality: Optional[str] = Field(description="Specifies whether or not to fit yearly seasonality chosen from {'auto', 'false', 'true'}, it is optional", default=None)
    weekly_seasonality: Optional[str] = Field(description="Specifies whether or not to fit weekly seasonality chosen from {'auto', 'false', 'true'}, it is optional", default=None)
    daily_seasonality: Optional[str] = Field(description="Specifies whether or not to fit daily seasonality chosen from {'auto', 'false', 'true'}, it is optional", default=None)
    seasonality_prior_scale: Optional[float] = Field(description="the parameter modulating the strength of the seasonality model, it is optional", default=None)
    holiday_prior_scale: Optional[float] = Field(description="the parameter modulating the strength of the holiday model, it is optional", default=None)
    changepoint_prior_scale: Optional[float] = Field(description="the parameter modulating the flexibility of the automatic changepoint selection, it is optional", default=None)
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    group_key: str = Field(description="the name of the group column for multiple time series. If not provided, ask the user. Do not guess.")
    endog: Optional[str] = Field(description="the endog of the dataset, it is optional", default=None)
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)
    group_params: Optional[dict] = Field(description="group-specific parameter overrides for massive additive forecasting, it is optional", default=None)
    holiday_table: Optional[str] = Field(description="the table of the holiday, it is optional", default=None)
    holiday_schema: Optional[str] = Field(description="the schema of the holiday_table, it is optional", default=None)
    categorical_variable: Union[Optional[str], Optional[list]] = Field(description="the categorical variable of the dataset, it is optional", default=None)


class MassiveModelPredictInput(BaseModel):
    """This class is used to define the schema of the inputs for predicting a grouped additive model."""

    predict_table: str = Field(description="the table to predict. If not provided, ask the user. Do not guess.")
    name: str = Field(description="the name of the model. If not provided, ask the user. Do not guess.")
    version: Optional[int] = Field(description="the version of the model, it is optional", default=None)
    predict_schema: Optional[str] = Field(description="the schema of the predict_table, it is optional", default=None)
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    group_key: str = Field(description="the name of the group column for multiple time series. If not provided, ask the user. Do not guess.")
    exog: Union[Optional[str], Optional[list]] = Field(description="the exog of the dataset, it is optional", default=None)
    group_params: Optional[dict] = Field(description="group-specific parameter overrides for grouped prediction, it is optional", default=None)
    logistic_growth_capacity: Optional[float] = Field(description="the logistic growth capacity of the model only valid when growth is 'logistic', it is optional", default=None)
    interval_width: Optional[float] = Field(description="the width of the uncertainty interval, it is optional", default=None)
    uncertainty_samples: Optional[int] = Field(description="the number of simulated draws used to estimate uncertainty intervals, it is optional", default=None)
    show_explainer: Optional[bool] = Field(description="whether to show explainer, it is optional", default=None)
    decompose_seasonality: Optional[bool] = Field(description="whether to decompose seasonality only valid when show_explainer is True, it is optional", default=None)
    decompose_holiday: Optional[bool] = Field(description="whether to decompose holiday only valid when show_explainer is True, it is optional", default=None)
    add_placeholder: Optional[bool] = Field(description="whether to add placeholder for the endog column and it is set True by default, it is optional", default=True)

class AdditiveModelForecastFitAndSave(BaseTool):
    r"""
    This tool is used to fit and predict the additive model forecast.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The result string containing the training table name, model storage name, and model storage version.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - fit_table
                  - The name of the table containing the training data.
                * - fit_schema
                  - The schema of the fit_table, it is optional.
                * - key
                  - The key column in the training table.
                * - endog
                  - The endogenous variable column in the training table.
                * - exog
                  - External regressors to include in the model.
                * - holiday_table
                  - The name of the table containing holiday data, it is optional.
                * - holiday_schema
                  - The schema of the holiday_table, it is optional.
                * - name
                  - The name of the model to save.
                * - version
                  - The version of the model to save.
                * - categorical_variable
                  - The categorical variable columns in the training table.
                * - growth
                  - The growth of the model chosen from {'linear', 'logistic'}.
                * - logistic_growth_capacity
                  - The logistic growth capacity of the model only valid when growth is 'logistic'.
                * - seasonality_mode
                  - The seasonality mode of the model chosen from {'additive', 'multiplicative'}.
                * - period
                  - The period of the seasonality or a list of periods.
                * - num_changepoints
                  - The number of changepoints in the model.
                * - changepoint_range
                  - The proportion of history in which trend changepoints will be estimated.
                * - regressor
                  - Specifies the regressor in a list of json such that ['{\"NAME\": \"X1\", \"PRIOR_SCALE\":4, \"MODE\": \"additive\" }'].
                * - changepoints
                  - Specifies a list of changepoints in the format of timestamp, such as ['2019-01-01 00:00:00, '2019-02-04 00:00:00'].
                * - yearly_seasonality
                  - Specifies whether or not to fit yearly seasonality chosen from {'auto', 'false', 'true'}.
                * - weekly_seasonality
                  - Specifies whether or not to fit weekly seasonality chosen from {'auto', 'false', 'true'}.
                * - daily_seasonality
                  - Specifies whether or not to fit daily seasonality chosen from {'auto', 'false', 'true'}.
                * - seasonality_prior_scale
                  - The parameter modulating the strength of the seasonality model.
                * - holiday_prior_scale
                  - The parameter modulating the strength of the holiday model.
                * - changepoint_prior_scale
                  - The parameter modulating the flexibility of the automatic changepoint selection.

    """
    name: str = "additive_model_forecast_fit_and_save"
    """Name of the tool."""
    description: str = "To fit the additive model forecast and save the model in model storage."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ModelFitInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(
        self,
        **kwargs
    ) -> str:
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        fit_table = kwargs.get("fit_table", None)
        if fit_table is None:
            return "Training table is required"
        key = kwargs.get("key", None)
        if key is None:
            return "key is required"
        name = kwargs.get("name", None)
        if name is None:
            return "Model name is required"
        fit_schema = kwargs.get("fit_schema", None)
        version = kwargs.get("version", None)
        growth = kwargs.get("growth", None)
        logistic_growth_capacity = kwargs.get("logistic_growth_capacity", None)
        seasonality_mode = kwargs.get("seasonality_mode", None)
        period = kwargs.get("period", None)
        num_changepoints = kwargs.get("num_changepoints", None)
        changepoint_range = kwargs.get("changepoint_range", None)
        regressor = kwargs.get("regressor", None)
        changepoints = kwargs.get("changepoints", None)
        yearly_seasonality = kwargs.get("yearly_seasonality", None)
        weekly_seasonality = kwargs.get("weekly_seasonality", None)
        daily_seasonality = kwargs.get("daily_seasonality", None)
        seasonality_prior_scale = kwargs.get("seasonality_prior_scale", None)
        holiday_prior_scale = kwargs.get("holiday_prior_scale", None)
        changepoint_prior_scale = kwargs.get("changepoint_prior_scale", None)
        endog = kwargs.get("endog", None)
        exog = kwargs.get("exog", None)
        holiday_table = kwargs.get("holiday_table", None)
        holiday_schema = kwargs.get("holiday_schema", None)
        categorical_variable = kwargs.get("categorical_variable", None)

        # check fit_table exists
        if not self.connection_context.has_table(fit_table, schema=fit_schema):
            return f"Table {fit_table} does not exist in the database."
        # check key exists in fit_table
        if key not in self.connection_context.table(fit_table, schema=fit_schema).columns:
            return f"Key {key} does not exist in the table {fit_table}."
        ms = ModelStorage(connection_context=self.connection_context)
        seasonality = None
        if period:
            if isinstance(period, list):
                seasonality = []
                for idx, p in enumerate(period):
                    fo = _guess_fourier_order(p)
                    seasonality.append(f'{{"NAME": "SEASONALITY_{idx}", "PERIOD":{p}, "FOURIER_ORDER":{fo} }}')
            else:
                fo = _guess_fourier_order(period)
                seasonality = f'{{"NAME": "SEASONALITY", "PERIOD":{period}, "FOURIER_ORDER":{fo} }}'
        try:
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
                holiday_df = self.connection_context.table(holiday_table, schema=holiday_schema)
            amf.fit(data=self.connection_context.table(fit_table, schema=fit_schema),
                  key=key,
                  endog=endog,
                  exog=exog,
                  holiday=holiday_df,
                  categorical_variable=categorical_variable)
        except ValueError as ve:
            # Handles invalid parameter values (e.g., alpha not in [0,1])
            return f"ValueError occurred: {str(ve)}"

        except KeyError as ke:
            # Handles missing columns in the DataFrame
            return f"KeyError occurred: {str(ke)}"

        except TypeError as te:
            # Handles type mismatches (e.g., non-numeric input where number expected)
            return f"TypeError occurred: {str(te)}"

        amf.name = name
        amf.version = generate_model_storage_version(ms, version, name)
        ms.save_model(model=amf, if_exists='replace')
        return json.dumps({"trained_table": fit_table, "model_storage_name": name, "model_storage_version": amf.version}, cls=_CustomEncoder)

    async def _arun(
        self,
        **kwargs
    ) -> str:
        return self._run(**kwargs
        )

class AdditiveModelForecastLoadModelAndPredict(BaseTool):
    r"""
    This tool is used to load the additive model forecast from model storage and predict.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The name of the predicted results table.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - predict_table
                  - The name of the table containing the prediction data.
                * - predict_schema
                  - The schema of the predict_table, it is optional.
                * - key
                  - The key column in the prediction table.
                * - name
                  - The name of the model to load.
                * - version
                  - The version of the model to load.
                * - exog
                  - External regressors to include in the prediction.
                * - logistic_growth_capacity
                  - Capacity for logistic growth.
                * - interval_width
                  - Width of the prediction intervals.
                * - uncertainty_samples
                  - Number of uncertainty samples to draw.
                * - show_explainer
                  - Whether to show the explainer.
                * - decompose_seasonality
                  - Whether to decompose seasonality.
                * - decompose_holiday
                  - Whether to decompose holiday effects.
    """
    name: str = "additive_model_forecast_load_model_and_predict"
    """Name of the tool."""
    description: str = "To load the additive model forecast from model storage and predict."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ModelPredictInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(
        self,
        **kwargs
    ) -> str:
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        predict_table = kwargs.get("predict_table", None)
        if predict_table is None:
            return "Prediction table is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        name = kwargs.get("name", None)
        if name is None:
            return "Model name is required"
        predict_schema = kwargs.get("predict_schema", None)
        version = kwargs.get("version", None)
        exog = kwargs.get("exog", None)
        logistic_growth_capacity = kwargs.get("logistic_growth_capacity", None)
        interval_width = kwargs.get("interval_width", None)
        uncertainty_samples = kwargs.get("uncertainty_samples", None)
        show_explainer = kwargs.get("show_explainer", None)
        decompose_seasonality = kwargs.get("decompose_seasonality", None)
        decompose_holiday = kwargs.get("decompose_holiday", None)
        add_placeholder = kwargs.get("add_placeholder", True)

        if not self.connection_context.has_table(predict_table, schema=predict_schema):
            return f"Table {predict_table} does not exist in the database."
        predict_df = self.connection_context.table(predict_table, schema=predict_schema)
        original_columns = list(predict_df.columns)
        if key not in self.connection_context.table(predict_table, schema=predict_schema).columns:
            return f"Key {key} does not exist in the table {predict_table}."
        ms = ModelStorage(connection_context=self.connection_context)
        model = ms.load_model(name=name, version=version)
        if hasattr(model, 'version'):
            if model.version is not None:
                version = model.version
        auto_repair_details = None
        prepared_df, kept_columns, missing_columns = build_repaired_predict_dataframe(
            predict_df,
            key=key,
            exog=exog,
            add_placeholder=add_placeholder,
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
                original_error="Required inference columns are missing before prediction starts.",
            )
        if kept_columns != original_columns:
            auto_repair_details = {
                "auto_repaired_predict_input": True,
                "predict_table_columns_before_repair": original_columns,
                "predict_table_columns_used_for_prediction": kept_columns,
            }
        try:
            model.predict(data=prepared_df,
                          key=key,
                          exog=exog,
                          logistic_growth_capacity=logistic_growth_capacity,
                          interval_width=interval_width,
                          uncertainty_samples=uncertainty_samples,
                          show_explainer=show_explainer,
                          decompose_seasonality=decompose_seasonality,
                          decompose_holiday=decompose_holiday,
                          add_placeholder=add_placeholder)
        except ValueError as ve:
            # Handles invalid parameter values (e.g., alpha not in [0,1])
            return f"ValueError occurred: {str(ve)}"

        except KeyError as ke:
            # Handles missing columns in the DataFrame
            return f"KeyError occurred: {str(ke)}"

        except TypeError as te:
            # Handles type mismatches (e.g., non-numeric input where number expected)
            return f"TypeError occurred: {str(te)}"
        except Exception as exc:  # pylint: disable=broad-except
            if not is_predict_feature_mismatch_error(exc):
                return json.dumps({"error": f"Prediction failed: {exc}"}, cls=_CustomEncoder)

            repaired_df, kept_columns, missing_columns = build_repaired_predict_dataframe(
                self.connection_context.table(predict_table, schema=predict_schema),
                key=key,
                exog=exog,
                add_placeholder=add_placeholder,
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
                    original_error=str(exc),
                )

            try:
                model.predict(data=repaired_df,
                              key=key,
                              exog=exog,
                              logistic_growth_capacity=logistic_growth_capacity,
                              interval_width=interval_width,
                              uncertainty_samples=uncertainty_samples,
                              show_explainer=show_explainer,
                              decompose_seasonality=decompose_seasonality,
                              decompose_holiday=decompose_holiday,
                              add_placeholder=add_placeholder)
            except Exception as retry_exc:  # pylint: disable=broad-except
                return format_predict_mismatch_diagnostic(
                    predict_table=predict_table,
                    predict_schema=predict_schema,
                    original_columns=original_columns,
                    kept_columns=kept_columns,
                    missing_columns=missing_columns,
                    key=key,
                    exog=exog,
                    original_error=str(retry_exc),
                )

            auto_repair_details = {
                "auto_repaired_predict_input": True,
                "predict_table_columns_before_repair": original_columns,
                "predict_table_columns_used_for_prediction": kept_columns,
            }

        ms.save_model(model=model, if_exists='replace_meta')
        # Uppercase user-supplied name fragments so the created table aligns with
        # HANA's default identifier folding on unquoted references.
        safe_name = _hana_safe_identifier(name)
        safe_predict_table = _hana_safe_identifier(predict_table)
        safe_predict_schema = _hana_safe_identifier(predict_schema) if predict_schema else None
        if safe_predict_schema:
            predicted_results = [f"PREDICT_RESULT_{safe_predict_schema}_{safe_predict_table}_{safe_name}_{version}"]
        else:
            predicted_results = [f"PREDICT_RESULT_{safe_predict_table}_{safe_name}_{version}"]
        self.connection_context.table(model._predict_output_table_names[0]).smart_save(predicted_results[0], force=True)
        if show_explainer is True:
            predicted_results.append(
                f"REASON_CODE_{safe_predict_schema}_{safe_predict_table}_{safe_name}_{version}" if safe_predict_schema else
                f"REASON_CODE_{safe_predict_table}_{safe_name}_{version}"
            )
            self.connection_context.table(model._predict_output_table_names[1]).smart_save(predicted_results[1], force=True)
            outputs = {
                "input_predict_table": predict_table,
                "input_predict_schema": predict_schema,
                "model_storage_name": name,
                "model_storage_version": version,
                "predicted_results_table": predicted_results[0],
                "decomposed_and_reason_code_table": predicted_results[1],
            }
            if auto_repair_details:
                outputs.update(auto_repair_details)
            return json.dumps(outputs, cls=_CustomEncoder)
        outputs = {
            "input_predict_table": predict_table,
            "input_predict_schema": predict_schema,
            "model_storage_name": name,
            "model_storage_version": version,
            "predicted_results_table": predicted_results[0],
        }
        if auto_repair_details:
            outputs.update(auto_repair_details)
        return json.dumps(outputs, cls=_CustomEncoder)

    async def _arun(
        self,
        **kwargs
    ) -> str:
        return self._run(
            **kwargs
        )


class MassiveAdditiveModelForecastFitAndSave(BaseTool):
    """This tool fits a grouped additive model forecast and saves it in model storage."""

    name: str = "massive_additive_model_forecast_fit_and_save"
    description: str = "To fit an additive model forecast per group(group_key column) and save it in model storage."
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = MassiveModelFitInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        """Use the tool."""

        if "kwargs" in kwargs:
            kwargs = kwargs["kwargs"]
        fit_table = kwargs.get("fit_table", None)
        if fit_table is None:
            return "Training table is required"
        key = kwargs.get("key", None)
        if key is None:
            return "key is required"
        group_key = kwargs.get("group_key", None)
        if group_key is None:
            return "Group key is required"
        name = kwargs.get("name", None)
        if name is None:
            return "Model name is required"
        fit_schema = kwargs.get("fit_schema", None)
        version = kwargs.get("version", None)
        growth = kwargs.get("growth", None)
        logistic_growth_capacity = kwargs.get("logistic_growth_capacity", None)
        seasonality_mode = kwargs.get("seasonality_mode", None)
        period = kwargs.get("period", None)
        num_changepoints = kwargs.get("num_changepoints", None)
        changepoint_range = kwargs.get("changepoint_range", None)
        regressor = kwargs.get("regressor", None)
        changepoints = kwargs.get("changepoints", None)
        yearly_seasonality = kwargs.get("yearly_seasonality", None)
        weekly_seasonality = kwargs.get("weekly_seasonality", None)
        daily_seasonality = kwargs.get("daily_seasonality", None)
        seasonality_prior_scale = kwargs.get("seasonality_prior_scale", None)
        holiday_prior_scale = kwargs.get("holiday_prior_scale", None)
        changepoint_prior_scale = kwargs.get("changepoint_prior_scale", None)
        endog = kwargs.get("endog", None)
        exog = kwargs.get("exog", None)
        group_params = kwargs.get("group_params", None)
        holiday_table = kwargs.get("holiday_table", None)
        holiday_schema = kwargs.get("holiday_schema", None)
        categorical_variable = kwargs.get("categorical_variable", None)

        if not self.connection_context.has_table(fit_table, schema=fit_schema):
            return f"Table {fit_table} does not exist in the database."
        fit_df = self.connection_context.table(fit_table, schema=fit_schema)
        if key not in fit_df.columns:
            return f"Key {key} does not exist in the table {fit_table}."
        if group_key not in fit_df.columns:
            return f"Group key {group_key} does not exist in the table {fit_table}."

        ms = ModelStorage(connection_context=self.connection_context)
        seasonality = None
        if period:
            if isinstance(period, list):
                seasonality = []
                for idx, seasonal_period in enumerate(period):
                    fourier_order = _guess_fourier_order(seasonal_period)
                    seasonality.append(
                        f'{{"NAME": "SEASONALITY_{idx}", "PERIOD":{seasonal_period}, "FOURIER_ORDER":{fourier_order} }}'
                    )
            else:
                fourier_order = _guess_fourier_order(period)
                seasonality = f'{{"NAME": "SEASONALITY", "PERIOD":{period}, "FOURIER_ORDER":{fourier_order} }}'
        try:
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
                changepoint_prior_scale=changepoint_prior_scale,
                massive=True,
            )
            holiday_df = None
            if holiday_table is not None:
                holiday_df = self.connection_context.table(holiday_table, schema=holiday_schema)
            amf.fit(
                data=fit_df,
                key=key,
                group_key=group_key,
                endog=endog,
                exog=exog,
                group_params=group_params,
                holiday=holiday_df,
                categorical_variable=categorical_variable,
            )
        except ValueError as ve:
            return f"ValueError occurred: {str(ve)}"
        except KeyError as ke:
            return f"KeyError occurred: {str(ke)}"
        except TypeError as te:
            return f"TypeError occurred: {str(te)}"

        amf.name = name
        amf.version = generate_model_storage_version(ms, version, name)
        ms.save_model(model=amf, if_exists='replace')
        return json.dumps({"trained_table": fit_table, "model_storage_name": name, "model_storage_version": amf.version}, cls=_CustomEncoder)

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class MassiveAdditiveModelForecastLoadModelAndPredict(BaseTool):
    """This tool loads a grouped additive model forecast from model storage and predicts."""

    name: str = "massive_additive_model_forecast_load_model_and_predict"
    description: str = "To load a grouped additive model forecast from model storage and predict."
    connection_context: ConnectionContext = None
    args_schema: Type[BaseModel] = MassiveModelPredictInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct,
        )

    def _run(self, **kwargs) -> str:
        """Use the tool."""

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
        predict_schema = kwargs.get("predict_schema", None)
        version = kwargs.get("version", None)
        exog = kwargs.get("exog", None)
        group_params = kwargs.get("group_params", None)
        logistic_growth_capacity = kwargs.get("logistic_growth_capacity", None)
        interval_width = kwargs.get("interval_width", None)
        uncertainty_samples = kwargs.get("uncertainty_samples", None)
        show_explainer = kwargs.get("show_explainer", None)
        decompose_seasonality = kwargs.get("decompose_seasonality", None)
        decompose_holiday = kwargs.get("decompose_holiday", None)
        add_placeholder = kwargs.get("add_placeholder", True)

        if not self.connection_context.has_table(predict_table, schema=predict_schema):
            return f"Table {predict_table} does not exist in the database."
        predict_df = self.connection_context.table(predict_table, schema=predict_schema)
        original_columns = list(predict_df.columns)
        if key not in predict_df.columns:
            return f"Key {key} does not exist in the table {predict_table}."
        if group_key not in predict_df.columns:
            return f"Group key {group_key} does not exist in the table {predict_table}."

        ms = ModelStorage(connection_context=self.connection_context)
        model = ms.load_model(name=name, version=version)
        if hasattr(model, 'version') and model.version is not None:
            version = model.version

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
                group_params=group_params,
                logistic_growth_capacity=logistic_growth_capacity,
                interval_width=interval_width,
                uncertainty_samples=uncertainty_samples,
                show_explainer=show_explainer,
                decompose_seasonality=decompose_seasonality,
                decompose_holiday=decompose_holiday,
                add_placeholder=add_placeholder,
            )
        except ValueError as ve:
            return f"ValueError occurred: {str(ve)}"
        except KeyError as ke:
            return f"KeyError occurred: {str(ke)}"
        except TypeError as te:
            return f"TypeError occurred: {str(te)}"
        except Exception as exc:  # pylint: disable=broad-except
            if not is_predict_feature_mismatch_error(exc):
                return json.dumps({"error": f"Prediction failed: {exc}"}, cls=_CustomEncoder)

            repaired_df, kept_columns, missing_columns = build_repaired_predict_dataframe(
                self.connection_context.table(predict_table, schema=predict_schema),
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
                    group_params=group_params,
                    logistic_growth_capacity=logistic_growth_capacity,
                    interval_width=interval_width,
                    uncertainty_samples=uncertainty_samples,
                    show_explainer=show_explainer,
                    decompose_seasonality=decompose_seasonality,
                    decompose_holiday=decompose_holiday,
                    add_placeholder=add_placeholder,
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

        ms.save_model(model=model, if_exists='replace_meta')
        # Uppercase user-supplied name fragments so the created table aligns with
        # HANA's default identifier folding on unquoted references.
        safe_name = _hana_safe_identifier(name)
        safe_predict_table = _hana_safe_identifier(predict_table)
        safe_predict_schema = _hana_safe_identifier(predict_schema) if predict_schema else None
        if safe_predict_schema:
            predicted_results = f"PREDICT_RESULT_{safe_predict_schema}_{safe_predict_table}_{safe_name}_{version}"
            explainer_table = f"REASON_CODE_{safe_predict_schema}_{safe_predict_table}_{safe_name}_{version}"
            error_table = f"PREDICT_ERROR_{safe_predict_schema}_{safe_predict_table}_{safe_name}_{version}"
        else:
            predicted_results = f"PREDICT_RESULT_{safe_predict_table}_{safe_name}_{version}"
            explainer_table = f"REASON_CODE_{safe_predict_table}_{safe_name}_{version}"
            error_table = f"PREDICT_ERROR_{safe_predict_table}_{safe_name}_{version}"

        output_table_names = list(getattr(model, "_predict_output_table_names", []))
        self.connection_context.table(output_table_names[0]).smart_save(predicted_results, force=True)

        if len(output_table_names) > 2:
            self.connection_context.table(output_table_names[2]).smart_save(error_table, force=True)
        elif len(output_table_names) > 1 and not show_explainer:
            self.connection_context.table(output_table_names[1]).smart_save(error_table, force=True)
        else:
            error_table = None

        outputs = {
            "input_predict_table": predict_table,
            "input_predict_schema": predict_schema,
            "model_storage_name": name,
            "model_storage_version": version,
            "predicted_results_table": predicted_results,
        }
        if show_explainer and len(output_table_names) > 1:
            self.connection_context.table(output_table_names[1]).smart_save(explainer_table, force=True)
            outputs["decomposed_and_reason_code_table"] = explainer_table
        if error_table:
            outputs["prediction_error_table"] = error_table
        if auto_repair_details:
            outputs.update(auto_repair_details)
        return json.dumps(outputs, cls=_CustomEncoder)

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)
