"""
This module define the agent tools for `dtw()` function in hana-ml.
"""
import json
import logging
from typing import Type, List, Tuple, Union
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

class DTWInput(BaseModel):
    """
    Class for DTW input arguments.
    """
    query_table : str = Field(description="Table (or view) that contains the query data for computing dynamic time warping (DTW) distance." +\
    " If not provided, ask the user, do not guess")
    ref_table : str = Field(description="Table (or view) that contains the reference data for computing dynamic time warping (dtw) distance." +\
    " If not provided, ask the user, do not guess")
    radius : int = Field(description="To restrict match curve in an area near diagonal, so that no each pair of" +\
    " subscripts in the match curve is no greater than the specified value Note that inappropriate setting of this parameter" +
    " may lead to no alignment result at all.")
    thread_ratio : float = Field(description="Specifies the ratio of available threads to be used for computation", default=None)
    distance_method : str = Field(description="Specifies the distance metric used, with valid options 'manhattan', 'euclidean', 'minkowski'," +\
    " 'chebyshev' and 'cosine'", default=None)
    minkowski_power : float = Field(description="Specifies the power of Minkowski metric", default=None)
    alignment_method : str = Field(description="Specifies the alignment constraint w.r.t. beginning and end points in reference time-series," +\
    " with valid options 'closed', 'open_begin', 'open_end' and 'open'", default=None)
    step_pattern : Union[int, List[Tuple[int, int]]] = Field(description="Specifies the type of step patterns, with predefined patterns ranging from 1 to 5," +\
    " and custom patterns in the form of list of tuples", default=None)
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
                * - ref_table
                  - Table (or view) that contains the reference data for computing dynamic time warping (DTW) distance. If not provided, ask the user, do not guess.
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
        ref_table : str,
        radius : int=None,
        thread_ratio : float=None,
        distance_method : str=None,
        minkowski_power : float=None,
        alignment_method : str=None,
        step_pattern : Union[int, List[Tuple[int, int]]]=None,
        save_alignment : bool=None,
        run_manager: CallbackManagerForToolRun = None#pylint:disable=unused-argument
        )-> str:
        dtw_out = dtw(query_data=self.connection_context.table(query_table),
                      ref_data=self.connection_context.table(ref_table),
                      radius=radius,
                      thread_ratio=thread_ratio,
                      distance_method=distance_method,
                      minkowski_power=minkowski_power,
                      alignment_method=alignment_method,
                      step_pattern=step_pattern,
                      save_alignment=save_alignment)
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
                    ref_table : str,
                    radius : int=None,
                    thread_ratio : float=None,
                    distance_method : str=None,
                    minkowski_power : float=None,
                    alignment_method : str=None,
                    step_pattern : Union[int, List[Tuple[int, int]]]=None,
                    save_alignment : bool=None,
                    run_manager: AsyncCallbackManagerForToolRun = None#pylint:disable=unused-argument
                    )-> str:
        """Use the tool asynchronously."""
        return self._run(query_table=query_table,
                         ref_table=ref_table,
                         radius=radius,
                         thread_ratio=thread_ratio,
                         distance_method=distance_method,
                         minkowski_power=minkowski_power,
                         alignment_method=alignment_method,
                         step_pattern=step_pattern,
                         save_alignment=save_alignment,
                         run_manager=run_manager)
