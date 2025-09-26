"""
This module contains functions for massive time series outlier detection.

The following classes are available:

    * :class `MassiveTSOutlierDetection`
"""
import json
import logging
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.outlier_detection import OutlierDetectionTS

from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder, _create_temp_table
logger = logging.getLogger(__name__)

class MassiveTSOutlierDetectionInput(BaseModel):
    """
    The input schema for the TSOutlierDetection tool.
    """
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    group_key: str = Field(description="The name of the group column for multiple time series. If not provided, ask the user. Do no guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    auto: Optional[bool] = Field(description="whether to use auto outlier detection, it is optional", default=None)
    schema_name: Optional[str] = Field(description="the schema name of the table, it is optional", default=None)
    detect_intermittent_ts: Optional[bool] = Field(description="whether to detect intermittent time series, it is optional", default=None)
    smooth_method: Optional[str] = Field(description="the smoothing method for the time series chosen from {'no', 'median', 'loess'}, it is optional", default=None)
    window_size: Optional[int] = Field(description="odd number, the window size for median filter, not less than 3, it is optional", default=None)
    loess_lag: Optional[int] = Field(description="odd number, the lag for LOESS, not less than 3, it is optional", default=None)
    current_value_flag: Optional[bool] = Field(description="whether to take the current data point when using LOESS smoothing method, it is optional", default=None)
    outlier_method: Optional[str] = Field(description="the outlier detection method chosen from {'z1', 'z2', 'mad', 'iqr', 'isolationforest', 'dbscan'}, it is optional", default=None)
    threshold: Optional[float] = Field(description="the threshold for outlier detection, it is optional", default=None)
    detect_seasonality: Optional[bool] = Field(description="whether to detect seasonality, it is optional", default=None)
    alpha: Optional[float] = Field(description="the criterion for the autocorrelation coefficient, it is optional", default=None)
    extrapolation: Optional[bool] = Field(description="whether to extrapolate the endpoints, it is optional", default=None)
    periods: Optional[int] = Field(description="the number of periods for seasonality, it is optional", default=None)
    random_state: Optional[int] = Field(description="specifies the seed for random number generator only valid for isolationforest, it is optional", default=None)
    n_estimators: Optional[int] = Field(description="the number of trees in the forest only valid for isolationforest, it is optional", default=None)
    max_samples: Optional[int] = Field(description="specifies the number of samples to draw from input to train each tree only valid for isolationforest, it is optional", default=None)
    bootstrap: Optional[bool] = Field(description="whether to use bootstrap samples when building trees only valid for isolationforest, it is optional", default=None)
    contamination: Optional[float] = Field(description="the proportion of outliers in the data set only valid for isolationforest, it is optional", default=None)
    minpts: Optional[int] = Field(description="the number of points in a neighborhood for a point to be considered as a core point only valid for dbscan, it is optional", default=None)
    eps: Optional[float] = Field(description="the maximum distance between two samples for one to be considered as in the neighborhood of the other only valid for dbscan, it is optional", default=None)
    distance_method: Optional[str] = Field(description="the distance method for dbscan chosen from {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'standardized_euclidean', 'cosine'}, it is optional", default=None)
    dbscan_normalization: Optional[bool] = Field(description="whether to normalize the data before dbscan, it is optional", default=None)
    dbscan_outlier_from_cluster: Optional[bool] = Field(description="specifies how to take outliers from DBSCAN result, it is optional", default=None)
    thread_ratio: Optional[float] = Field(description="the ratio of threads to use for parallel processing, it is optional", default=None)
    residual_usage: Optional[str] = Field(description="specifies which residual to output chosen from {'outlier_detection', 'outlier_correction'}, it is optional", default=None)
    voting_config: Optional[dict] = Field(description="the configuration for voting, it is optional", default=None)
    voting_outlier_method_criterion: Optional[float] = Field(description="the criterion for voting outlier method, it is optional", default=None)

class MassiveTSOutlierDetection(BaseTool):
    """
    This tool detects outliers in massive time series data.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The outliers in the time series data and the statistics of the detection.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - table_name
                  - the name of the table. If not provided, ask the user. Do not guess.
                * - key
                  - the key of the dataset. If not provided, ask the user. Do not guess.
                * - group_key
                  - The name of the group column for multiple time series. If not provided, ask the user. Do no guess.
                * - endog
                  - the endog of the dataset. If not provided, ask the user. Do not guess.
                * - schema_name
                  - the schema_name of the table, it is optional
                * - auto
                  - whether to use auto outlier detection, it is optional
                * - detect_intermittent_ts
                  - whether to detect intermittent time series, it is optional
                * - smooth_method
                  - the smoothing method for the time series chosen from {'no', 'median', 'loess'}, it is optional
                * - window_size
                  - odd number, the window size for median filter, not less than 3, it is optional
                * - loess_lag
                  - odd number, the lag for LOESS, not less than 3, it is optional
                * - current_value_flag
                  - whether to take the current data point when using LOESS smoothing method, it is optional
                * - outlier_method
                  - the outlier detection method chosen from {'z1', 'z2', 'mad', 'iqr', 'isolationforest', 'dbscan'}, it is optional
                * - threshold
                  - the threshold for outlier detection, it is optional
                * - detect_seasonality
                  - whether to detect seasonality, it is optional
                * - alpha
                  - the criterion for the autocorrelation coefficient, it is optional
                * - extrapolation
                  - whether to extrapolate the endpoints, it is optional
                * - periods
                  - the number of periods for seasonality, it is optional
                * - random_state
                  - specifies the seed for random number generator only valid for isolationforest, it is optional
                * - n_estimators
                  - the number of trees in the forest only valid for isolationforest, it is optional
                * - max_samples
                  - specifies the number of samples to draw from input to train each tree only valid for isolationforest, it is optional
                * - bootstrap
                  - whether to use bootstrap samples when building trees only valid for isolationforest, it is optional
                * - contamination
                  - the proportion of outliers in the data set only valid for isolationforest, it is optional
                * - minpts
                  - the number of points in a neighborhood for a point to be considered as a core point only valid for dbscan, it is optional
                * - eps
                  - the maximum distance between two samples for one to be considered as in the neighborhood of the other only valid for dbscan, it is optional
                * - distance_method
                  - the distance method for dbscan chosen from {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'standardized_euclidean', 'cosine'}, it is optional
                * - dbscan_normalization
                  - whether to normalize the data before dbscan, it is optional
                * - dbscan_outlier_from_cluster
                  - specifies how to take outliers from DBSCAN result, it is optional
                * - thread_ratio
                  - the ratio of threads to use for parallel processing, it is optional
                * - residual_usage
                  - specifies which residual to output chosen from {'outlier_detection', 'outlier_correction'}, it is optional
                * - voting_config
                  - the configuration for voting, it is optional
                * - voting_outlier_method_criterion
                  - the criterion for voting outlier method, it is optional
    """
    name: str = "massive_ts_outlier_detection"
    """Name of the tool."""
    description: str = "To detect outliers in massive time series data or outliers per group(group_key Column). "
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = MassiveTSOutlierDetectionInput
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
        table_name = kwargs.get("table_name", None)
        if table_name is None:
            return "Table name is required"
        key = kwargs.get("key", None)
        if key is None:
            return "Key is required"
        group_key = kwargs.get("group_key", None)
        if group_key is None:
            return "Group key is required"
        endog = kwargs.get("endog", None)
        if endog is None:
            return "Endog is required"
        schema_name = kwargs.get("schema_name", None)
        auto = kwargs.get("auto", None)
        detect_intermittent_ts = kwargs.get("detect_intermittent_ts", None)
        smooth_method = kwargs.get("smooth_method", None)
        window_size = kwargs.get("window_size", None)
        loess_lag = kwargs.get("loess_lag", None)
        current_value_flag = kwargs.get("current_value_flag", None)
        outlier_method = kwargs.get("outlier_method", None)
        threshold = kwargs.get("threshold", None)
        detect_seasonality = kwargs.get("detect_seasonality", None)
        alpha = kwargs.get("alpha", None)
        extrapolation = kwargs.get("extrapolation", None)
        periods = kwargs.get("periods", None)
        random_state = kwargs.get("random_state", None)
        n_estimators = kwargs.get("n_estimators", None)
        max_samples = kwargs.get("max_samples", None)
        bootstrap = kwargs.get("bootstrap", None)
        contamination = kwargs.get("contamination", None)
        minpts = kwargs.get("minpts", None)
        eps = kwargs.get("eps", None)
        distance_method = kwargs.get("distance_method", None)
        dbscan_normalization = kwargs.get("dbscan_normalization", None)
        dbscan_outlier_from_cluster = kwargs.get("dbscan_outlier_from_cluster", None)
        thread_ratio = kwargs.get("thread_ratio", None)
        residual_usage = kwargs.get("residual_usage", None)
        voting_config = kwargs.get("voting_config", None)
        voting_outlier_method_criterion = kwargs.get("voting_outlier_method_criterion", None)

        # Check if the table exists
        if not self.connection_context.has_table(table_name, schema=schema_name):
            return json.dumps({
                "error": f"Table {table_name} does not exist in the database."
            }, cls=_CustomEncoder)
        if key not in self.connection_context.table(table_name, schema=schema_name).columns:
            return json.dumps({
                "error": f"Key {key} does not exist in the table {table_name}."
            }, cls=_CustomEncoder)
        if endog not in self.connection_context.table(table_name, schema=schema_name).columns:
            return json.dumps({
                "error": f"Endog {endog} does not exist in the table {table_name}."
            }, cls=_CustomEncoder)
        odt = OutlierDetectionTS(
                auto=auto,
                detect_intermittent_ts=detect_intermittent_ts,
                smooth_method=smooth_method,
                window_size=window_size,
                loess_lag=loess_lag,
                current_value_flag=current_value_flag,
                outlier_method=outlier_method,
                threshold=threshold,
                detect_seasonality=detect_seasonality,
                alpha=alpha,
                extrapolation=extrapolation,
                periods=periods,
                random_state=random_state,
                n_estimators=n_estimators,
                max_samples=max_samples,
                bootstrap=bootstrap,
                contamination=contamination,
                minpts=minpts,
                eps=eps,
                distance_method=distance_method,
                dbscan_normalization=dbscan_normalization,
                dbscan_outlier_from_cluster=dbscan_outlier_from_cluster,
                residual_usage=residual_usage,
                voting_config=voting_config,
                voting_outlier_method_criterion=voting_outlier_method_criterion,
                thread_ratio=thread_ratio,
                massive=True)
        df = self.connection_context.table(table_name, schema_name).select(group_key, key, endog) #pylint: disable=invalid-name
        result = odt.fit_predict(df,
                                 key=key,
                                 group_key=group_key,
                                 endog=endog)
        # Collect outliers grouped by group_key
        outlier_rows = result.filter("IS_OUTLIER = 1").collect()
        group_col = group_key if group_key in outlier_rows.columns else result.columns[0]
        id_col = key if key in outlier_rows.columns else result.columns[1]
        outlier_dict = {}
        for _, row in outlier_rows.iterrows():
            group_id = row[group_col]
            outlier_id = row[id_col]
            outlier_dict.setdefault(group_id, []).append(outlier_id)
        outliers = [
            {"group_id": group_id, "outlier_ids": outlier_ids}
            for group_id, outlier_ids in outlier_dict.items()
        ]
        results = {
            "outliers": outliers,
            "result_select_statement": _create_temp_table(self.connection_context, result.select_statement, self.name)
        }
        for _, row in odt.stats_.collect().iterrows():
            results[row[odt.stats_.columns[0]]] = row[odt.stats_.columns[1]]

        return json.dumps(results, cls=_CustomEncoder)

    async def _arun(
        self, **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            **kwargs
        )
