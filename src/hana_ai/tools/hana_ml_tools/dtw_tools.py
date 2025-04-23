"""
This module define the agent tools for `dtw()` function in hana-ml.
"""
import json
import logging
from typing import Type, List, Tuple, Union, Annotated
from annotated_types import Interval
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.dtw import dtw
from hana_ai.utility import remove_prefix_sharp

logger = logging.getLogger(__name__)
step_pt = Tuple[int, int, Union[int, float]]

class DTWInput(BaseModel):
    """
    Class for DTW input arguments.
    """
    query_table : str = Field(description="Table (or view) that contains the query data for computing dynamic time warping (DTW) distance." +\
    " If not provided, ask the user, do not guess")
    query_ts_id : str = Field(description="Specifies the column that contains the IDs of different time-series in the query data. "+\
    "If not provided, ask the user, do not guess")
    query_ts_order : str = Field(description="Specifies the column that contains the sequential-order of time-series in query data. "+\
    "If not provided, ask the user, do not guess")
    query_ts_cols : Union[str, List[str]]  = Field(description="Specifies the columns that contain the time-series' values in query data. " +\
    "If not provided, the remaining columns of the query data excluding query_ts_id and query_ts_order shall be used.", default=None)
    ref_table : str = Field(description="Table (or view) that contains the reference data for computing dynamic time warping (dtw) distance." +\
    " If not provided, ask the user, do not guess")
    ref_ts_id : str = Field(description="Specifies the column the contains of the IDs of different time-series in the reference data." +\
    " If not provide, ask the user, do not guess")
    ref_ts_order : str = Field(description="Specifies the column that contains the sequential-order of time-series in reference data." +\
    " If not provided, ask the user, do not guess")
    ref_ts_cols : str = Field(description="Specifies the columns that contain the time-series' values in reference data. " +\
    "If not provided, the remaining columns of the reference data excluding ref_ts_id and ref_ts_order shall be used.", default=None)
    radius : int = Field(description="To restrict match curve in an area near diagonal, so that no each pair of" +\
    " subscripts in the match curve is no greater than the specified value Note that inappropriate setting of this parameter" +
    " may lead to no alignment result at all.")
    thread_ratio : float = Field(description="Specifies the ratio of available threads to be used for computation", default=None)
    distance_method : str = Field(description="Specifies the distance metric used, with valid options 'manhattan', 'euclidean', 'minkowski'," +\
    " 'chebyshev' and 'cosine'", default=None)
    minkowski_power : float = Field(description="Specifies the power of Minkowski metric", default=None)
    alignment_method : str = Field(description="Specifies the alignment constraint w.r.t. beginning and end points in reference time-series," +\
    " with valid options 'closed', 'open_begin', 'open_end' and 'open'", default=None)
    step_pattern : Union[Annotated[int, Interval(ge=1, le=5)], List[Union[step_pt, Tuple[step_pt,...]]]] = Field(description="Specifies the " +\
    "step pattern with 1) predefined patterns ranging from 1 to 5, or 2) custom patterns in the form of list of tuples", default=None)
    save_alignment : bool = Field(description="If set as True, save the alignment information, otherwise do not save", default=None)

class DTW(BaseTool):
    r"""
    This tool calculates the dynamic time warping (DTW) distances between the query and the reference time-series.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The result string containing the DTW results and optionally the alignment table name and statistics.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - query_table
                  - Table (or view) that contains the query data for computing dynamic time warping (DTW) distance. If not provided, ask the user, do not guess.
                * - query_ts_id
                  - Specifies the column that contains the IDs of different time-series in the query data. If not provided, ask the user, do not guess.
                * - query_ts_order
                  - Specifies the column that identifies the sequential-orders of time-series in query data. If not provided, ask the user, do not guess.
                * - query_ts_cols
                  - Specifies the columns that contains the time-series' values in query data.
                * - ref_table
                  - Table (or view) that contains the reference data for computing dynamic time warping (DTW) distance. If not provided, ask the user, do not guess.
                * - ref_ts_id
                  - Specifies the column that contains the IDs of different time-series in the query data. If not provided, ask the user, do not guess.
                * - ref_ts_order
                  - Specifies the column that identifies the sequential-orders of time-series in reference data. If not provided, ask the user, do not guess.
                * - ref_ts_cols
                  - Specifies the columns that contains the time-series' values in reference data.
                * - radius
                  - To restrict match curve in an area near diagonal, so that no each pair of subscripts in the match curve is no greater than the specified value. Note that inappropriate setting of this parameter may lead to no alignment result at all.
                * - thread_ratio
                  - Specifies the ratio of available threads to be used for computation.
                * - distance_method
                  - Specifies the distance metric used, with valid options 'manhattan', 'euclidean', 'minkowski', 'chebyshev' and 'cosine'.
                * - minkowski_power
                  - Specifies the power of Minkowski metric.
                * - alignment_method
                  - Specifies the alignment constraint w.r.t. beginning and end points in reference time-series, with valid options 'closed', 'open_begin', 'open_end' and 'open'.
                * - step_pattern
                  - Specifies the type of step patterns, with predefined patterns ranging from 1 to 5, and custom patterns in the form of list of tuples.
                * - save_alignment
                  - If set as True, save the alignment information, otherwise do not save.
    """
    name: str = "DTW"
    """Name of the tool."""
    description: str = "To compute dynamic time warping (DTW) distances between the query and the reference time-series."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = DTWInput
    #return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context
        )

    def _run(#pylint:disable=too-many-positional-arguments
        self,
        query_table : str,
        query_ts_id : str,
        query_ts_order : str,
        ref_table : str,
        ref_ts_id : str,
        ref_ts_order : str,
        query_ts_cols : Union[str, List[str]]=None,
        ref_ts_cols : Union[str, List[str]]=None,
        radius : int=None,
        thread_ratio : float=None,
        distance_method : str=None,
        minkowski_power : float=None,
        alignment_method : str=None,
        step_pattern : Union[Annotated[int, Interval(ge=1, le=5)], List[Union[step_pt, Tuple[step_pt,...]]]]=None,
        save_alignment : bool=None,
        run_manager: CallbackManagerForToolRun = None#pylint:disable=unused-argument
        )-> str:
        query_data=self.connection_context.table(query_table)
        query_data_cols = query_data.columns
        if query_ts_cols is None:
            query_ts_cols = query_data_cols.copy()
            for col in [query_ts_id, query_ts_order]:
                try:
                    query_ts_cols.remove(col)
                except Exception:
                    msg = f'Column `{col}` not found in query data.'
                    return f'ValueError: {msg}'
        elif isinstance(query_ts_cols, str):
            query_ts_cols = [query_ts_cols]
        ref_data=self.connection_context.table(ref_table)
        ref_data_cols = ref_data.columns
        if ref_ts_cols is None:
            ref_ts_cols = ref_data_cols.copy()
            for col in [ref_ts_id, ref_ts_order]:
                try:
                    ref_ts_cols.remove(col)
                except Exception:
                    msg = f'Column `{col}` not found in reference data.'
                    return f'ValueError: {msg}'
        elif isinstance(ref_ts_cols, str):
            ref_ts_cols = [ref_ts_cols]
        query_data = query_data[[query_ts_id, query_ts_order] + query_ts_cols]
        ref_data = ref_data[[ref_ts_id, ref_ts_order] + ref_ts_cols]
        try:
            dtw_out = dtw(query_data=query_data,
                          ref_data=ref_data,
                          radius=radius,
                          thread_ratio=thread_ratio,
                          distance_method=distance_method,
                          minkowski_power=minkowski_power,
                          alignment_method=alignment_method,
                          step_pattern=step_pattern,
                          save_alignment=save_alignment)
        except ValueError as verr:
            # Handles invalid parameter values (e.g., alpha not in [0,1])
            return f'ValueError occurred: {str(verr)}'
        except KeyError as kerr:
            # Handles missing columns in the DataFrame
            return f'KeyError occurred: {str(kerr)}'
        except TypeError as terr:
            # Handles type mismatches (e.g., non-numeric input where number expected)
            return f'TypeError occurred: {str(terr)}'

        res_key = "DTW_results_in_tuple" + "(" + ", ".join(dtw_out[0].columns) + ")"
        res_list = []
        for row in dtw_out[0].collect().itertuples():
            res_list.append(str(row[1:]))
        res_content = "[" + ", ".join(res_list) + "]"
        out_dict = {}
        out_dict[res_key] = res_content
        if save_alignment:
            align_tab = "_".join([remove_prefix_sharp(query_table),
                                  remove_prefix_sharp(ref_table),
                                  "DTW_ALIGNMENT"])
            dtw_out[1].save(align_tab, force=True)
            out_dict['dtw_alignment_table'] = align_tab
        if dtw_out[2].count() > 0:
            for _, row in dtw_out[2].collect().iterrows():
                out_dict[row['STATS_NAME']] = row['STATS_VALUE']
        return json.dumps(out_dict)

    async def _arun(self,#pylint:disable=too-many-positional-arguments
                    query_table : str,
                    query_ts_id : str,
                    query_ts_order : str,
                    ref_table : str,
                    ref_ts_id : str,
                    ref_ts_order : str,
                    query_ts_cols : Union[str, List[str]]=None,
                    ref_ts_cols : Union[str, List[str]]=None,
                    radius : int=None,
                    thread_ratio : float=None,
                    distance_method : str=None,
                    minkowski_power : float=None,
                    alignment_method : str=None,
                    step_pattern : Union[Annotated[int, Interval(ge=1, le=5)], List[Union[step_pt, Tuple[step_pt,...]]]]=None,
                    save_alignment : bool=None,
                    run_manager: AsyncCallbackManagerForToolRun = None#pylint:disable=unused-argument
                    )-> str:
        """Use the tool asynchronously."""
        return self._run(query_table=query_table,
                         query_ts_id=query_ts_id,
                         query_ts_order=query_ts_order,
                         ref_table=ref_table,
                         ref_ts_id=ref_ts_id,
                         ref_ts_order=ref_ts_order,
                         radius=radius,
                         thread_ratio=thread_ratio,
                         distance_method=distance_method,
                         minkowski_power=minkowski_power,
                         alignment_method=alignment_method,
                         step_pattern=step_pattern,
                         save_alignment=save_alignment,
                         run_manager=run_manager)
