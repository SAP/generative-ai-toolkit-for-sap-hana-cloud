"""
This module contains functions for time series outlier detection.
"""

import json
import logging
from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.outlier_detection import OutlierDetectionTS

logger = logging.getLogger(__name__)

class TSOutlierDetectionInput(BaseModel):
    table_name: str = Field(description="the name of the table. If not provided, ask the user. Do not guess.")
    key: str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess.")
    endog: str = Field(description="the endog of the dataset. If not provided, ask the user. Do not guess.")
    auto: Optional[bool] = Field(description="whether to use auto outlier detection, it is optional")
    detect_intermittent_ts: Optional[bool] = Field(description="whether to detect intermittent time series, it is optional")
    smooth_method: Optional[str] = Field(description="the smoothing method for the time series chosen from {'no', 'median', 'loess'}, it is optional")
    window_size: Optional[int] = Field(description="odd number, the window size for median filter, not less than 3, it is optional")
    loess_lag: Optional[int] = Field(description="odd number, the lag for LOESS, not less than 3, it is optional")
    current_value_flag: Optional[bool] = Field(description="whether to take the current data point when using LOESS smoothing method, it is optional")
    outlier_method: Optional[str] = Field(description="the outlier detection method chosen from {'z1', 'z2', 'mad', 'iqr', 'isolationforest', 'dbscan'}, it is optional")
    threshold: Optional[float] = Field(description="the threshold for outlier detection, it is optional")
    detect_seasonality: Optional[bool] = Field(description="whether to detect seasonality, it is optional")
    alpha: Optional[float] = Field(description="the criterion for the autocorrelation coefficient, it is optional")
    extrapolation: Optional[bool] = Field(description="whether to extrapolate the endpoints, it is optional")
    periods: Optional[int] = Field(description="the number of periods for seasonality, it is optional")
    random_state: Optional[int] = Field(description="specifies the seed for random number generator only valid for isolationforest, it is optional")
    n_estimators: Optional[int] = Field(description="the number of trees in the forest only valid for isolationforest, it is optional")
    max_samples: Optional[int] = Field(description="specifies the number of samples to draw from input to train each tree only valid for isolationforest, it is optional")
    bootstrap: Optional[bool] = Field(description="whether to use bootstrap samples when building trees only valid for isolationforest, it is optional")
    contamination: Optional[float] = Field(description="the proportion of outliers in the data set only valid for isolationforest, it is optional")
    minpts: Optional[int] = Field(description="the number of points in a neighborhood for a point to be considered as a core point only valid for dbscan, it is optional")
    eps: Optional[float] = Field(description="the maximum distance between two samples for one to be considered as in the neighborhood of the other only valid for dbscan, it is optional")
    distiance_method: Optional[str] = Field(description="the distance method for dbscan chosen from {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'standardized_euclidean', 'cosine'}, it is optional")
    dbscan_normalization: Optional[bool] = Field(description="whether to normalize the data before dbscan, it is optional")
    dbscan_outlier_from_cluster: Optional[bool] = Field(description="specifies how to take outliers from DBSCAN result, it is optional")
    thread_ratio: Optional[float] = Field(description="the ratio of threads to use for parallel processing, it is optional")
    residual_usage: Optional[str] = Field(description="specifies which residual to output chosen from {'outlier_detection', 'outlier_correction'}, it is optional")
    voting_config: Optional[dict] = Field(description="the configuration for voting, it is optional")
    voting_outlier_method_criterion: Optional[float] = Field(description="the criterion for voting outlier method, it is optional")

class TSOutlierDetection(BaseTool):
    """
    This tool detects outliers in time series data.
    """
    name: str = "ts_outlier_detection"
    """Name of the tool."""
    description: str = "To detect outliers in time series data. "
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = TSOutlierDetectionInput
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(
        self, table_name: str, key: str, endog: str, auto: Optional[bool] = None,
        detect_intermittent_ts: Optional[bool] = None, smooth_method: Optional[str] = None,
        window_size: Optional[int] = None, loess_lag: Optional[int] = None,
        current_value_flag: Optional[bool] = None, outlier_method: Optional[str] = None,
        threshold: Optional[float] = None, detect_seasonality: Optional[bool] = None,
        alpha: Optional[float] = None, extrapolation: Optional[bool] = None,
        periods: Optional[int] = None, random_state: Optional[int] = None,
        n_estimators: Optional[int] = None, max_samples: Optional[int] = None,
        bootstrap: Optional[bool] = None, contamination: Optional[float] = None,
        minpts: Optional[int] = None, eps: Optional[float] = None,
        distiance_method: Optional[str] = None, dbscan_normalization: Optional[bool] = None,
        dbscan_outlier_from_cluster: Optional[bool] = None, thread_ratio: Optional[float] = None,
        residual_usage: Optional[str] = None, voting_config: Optional[dict] = None,
        voting_outlier_method_criterion: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
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
                distiance_method=distiance_method,
                dbscan_normalization=dbscan_normalization,
                dbscan_outlier_from_cluster=dbscan_outlier_from_cluster,
                residual_usage=residual_usage,
                voting_config=voting_config,
                voting_outlier_method_criterion=voting_outlier_method_criterion,
                thread_ratio=thread_ratio)
        df = self.connection_context.table(table_name).select(key, endog)
        result = odt.fit_predict(df,
                                 key=key,
                                 endog=endog)
        outliers = result.filter("IS_OUTLIER = 1").collect()[result.columns[0]].tolist()
        results = {
            "outliers": outliers
        }
        for _, row in odt.stats_.collect().iterrows():
            results[row[odt.stats_.columns[0]]] = row[odt.stats_.columns[1]]
        return json.dumps(results)

    async def _arun(
        self, table_name: str, key: str, endog: str, auto: Optional[bool] = None,
        detect_intermittent_ts: Optional[bool] = None, smooth_method: Optional[str] = None,
        window_size: Optional[int] = None, loess_lag: Optional[int] = None,
        current_value_flag: Optional[bool] = None, outlier_method: Optional[str] = None,
        threshold: Optional[float] = None, detect_seasonality: Optional[bool] = None,
        alpha: Optional[float] = None, extrapolation: Optional[bool] = None,
        periods: Optional[int] = None, random_state: Optional[int] = None,
        n_estimators: Optional[int] = None, max_samples: Optional[int] = None,
        bootstrap: Optional[bool] = None, contamination: Optional[float] = None,
        minpts: Optional[int] = None, eps: Optional[float] = None,
        distiance_method: Optional[str] = None, dbscan_normalization: Optional[bool] = None,
        dbscan_outlier_from_cluster: Optional[bool] = None, thread_ratio: Optional[float] = None,
        residual_usage: Optional[str] = None, voting_config: Optional[dict] = None,
        voting_outlier_method_criterion: Optional[float] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(
            table_name, key, endog, auto=auto, detect_intermittent_ts=detect_intermittent_ts, smooth_method=smooth_method,
            window_size=window_size, loess_lag=loess_lag, current_value_flag=current_value_flag, outlier_method=outlier_method,
            threshold=threshold, detect_seasonality=detect_seasonality, alpha=alpha, extrapolation=extrapolation,
            periods=periods, random_state=random_state, n_estimators=n_estimators, max_samples=max_samples,
            bootstrap=bootstrap, contamination=contamination, minpts=minpts, eps=eps, distiance_method=distiance_method,
            dbscan_normalization=dbscan_normalization, dbscan_outlier_from_cluster=dbscan_outlier_from_cluster,
            thread_ratio=thread_ratio, residual_usage=residual_usage, voting_config=voting_config,
            voting_outlier_method_criterion=voting_outlier_method_criterion, run_manager=run_manager
        )
