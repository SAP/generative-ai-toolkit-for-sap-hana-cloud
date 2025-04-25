"""
This module defines a agent tool for fft() function in hana-ml.
"""
import json
import logging
from typing import Type
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.fft import fft
from hdbcli.dbapi import ProgrammingError
from hana_ai.utility import remove_prefix_sharp
from hana_ai.tools.hana_ml_tools.utility import add_stopping_hint

logger = logging.getLogger(__name__)

class FFTInput(BaseModel):
    """
    Input for fft function.
    """
    table_name : str = Field(description="Table (or view) that contains the input sequence data for applying FFT or inverse FFT. If not provided, ask the user, do not guess")
    key : str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess")
    real_col : str = Field(description="Specifies the column name that contains the real sequence data for applying FFT. " +\
    "It cannot be None if parameter `imag_col` is None", default=None)
    imag_col : str = Field(description="Specifies the column name that contain the imaginary sequence data for applying FFT. " +\
    "It cannot be None if parameter `real_col` is None", default=None)
    inverse : bool = Field(description="Specifies whether inverse FFT is applied or not. Inverse FFT is applied if it is set as True," +\
    " otherwise regular forward FFT is applied.", default=None)
    window : str = Field(description="Specifies the window type for windowed fft, valid options including " +\
    "'none', 'hamming', 'hann', 'hanning', 'bartlett', 'triangular', 'bartlett_hann', 'blackman','blackman_harris', 'blackman_nuttall', " +\
    "'bohman', 'cauchy', 'cheb', 'chebwin', 'cosine', 'sine', 'flattop', 'gaussian', 'kaiser', 'lanczos', 'sinc', 'nuttall', 'parzen', " +\
    "'poisson', 'poisson_hann', 'poisson_hanning', 'rectangle', 'riemann', 'riesz', 'tukey'", default=None)
    window_start : int = Field(description="Specifies the starting point of tapering window", default=None)
    window_length : int = Field(description="Specifies the length of tapering window", default=None)
    alpha : float = Field(description="A parameter associated with the window types including 'blackman', 'cauchy', 'gaussian', 'poisson' and 'poisson_hann'", default=None)
    beta : float = Field(description="A parameter associated Kaiser window type", default=None)
    attenuation : float = Field(description="A parameter for the 'cheb' windown type", default=None)
    flattop_mode : str = Field(description="Specifies the sampling mode for 'flattop' window type, with valid options including 'symmetric' and 'periodic'", default=None)
    flattop_precision : str = Field(description="A parameter for the 'flattop' window type, with valid options including 'none' and 'octave'", default=None)
    r : float = Field(description="A parameter for the 'tukey' window type", default=None)

class FFT(BaseTool):
    r"""
    This tool applies FFT to time-series data and stores the transformed result.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The result string containing the FFT result table name.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - table_name
                  - Table (or view) that contains the input data for applying FFT. If not provided, ask the user, do not guess.
                * - key
                  - Specifies the key of the dataset. If not provided, ask the user. Do not guess
                * - real_col
                  - Specifies the column name that contains the real sequence data for applying FFT. It cannot be None if parameter
                    `imag_col` is None.
                * - imag_col
                  - Specifies the column name that contains the imaginary sequence data for applying FFT. It cannot be None if parameter
                    `real_col` is None.
                * - inverse
                  - If set as True, inverse FFT is applied, otherwise regular forward FFT is applied.
                * - window
                  - Specifies the window type for windowed FFT, with valid options including 'none', 'hamming', 'hann', 'hanning', 'bartlett', 'triangular', 'bartlett_hann', 'blackman', 'blackman_harris', 'blackman_nuttall', 'bohman', 'cauchy', 'cheb', 'chebwin', 'cosine', 'sine', 'flattop', 'gaussian', 'kaiser', 'lanczos', 'sinc', 'nuttall', 'parzen', 'poisson', 'poisson_hann', 'poisson_hanning', 'rectangle', 'riemann', 'riesz', 'tukey'.
                * - window_start
                  - Specifies the starting point of tapering window.
                * - window_length
                  - Specifies the length of tapering window.
                * - alpha
                  - A parameter associated with the window types including 'blackman', 'cauchy', 'gaussian', 'poisson' and 'poisson_hann'.
                * - beta
                  - A parameter associated with the Kaiser window type.
                * - attenuation
                  - A parameter for the 'cheb' window type.
                * - flattop_mode
                  - Specifies the sampling mode for 'flattop' window type, with valid options including 'symmetric' and 'periodic'.
                * - flattop_precision
                  - A parameter for the 'flattop' window type, with valid options including 'none' and 'octave'.
                * - r
                  - A parameter for the 'tukey' window type.
    """
    name: str = "fast_fourier_transform_for_timeseries"
    """Name of the tool."""
    description: str = "To compute the fast-Fourier transform (FFT) or inverse FFT of a single input time-series."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = FFTInput
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
        table_name : str,
        key : str,
        real_col : str=None,
        imag_col : str=None,
        inverse : bool=None,
        window : str=None,
        window_start : int=None,
        window_length : int=None,
        alpha : float=None,
        beta : float=None,
        attenuation : float=None,
        flattop_mode : str=None,
        flattop_precision : str=None,
        r : float=None,
        run_manager: CallbackManagerForToolRun = None#pylint:disable=unused-argument
        )-> str:
        stop_msg = "Please stop the execution."
        cols = [real_col, imag_col]
        if all(col is None for col in cols):
            msg = 'Parameter `real_col` and `imag_col` cannot both be None.'
            return add_stopping_hint(f'ValueError occurred: {msg}')
        if any(col is None for col in cols):
            cols.remove(None)
        num_type = 'imag' if (real_col is None and imag_col is not None) else None
        try:
            input_data = self.connection_context.table(table_name)[[key] + cols]
            if "INT" not in input_data.dtypes()[0][1]:#inclusive of column name check
                input_data = input_data.add_id(f"{key}" + "_int", ref_col=key).deselect(key)
            fft_res = fft(data=input_data,
                          num_type=num_type, inverse=inverse, window=window,
                          window_start=window_start, window_length=window_length,
                          alpha=alpha, beta=beta, attenuation=attenuation,
                          flattop_mode=flattop_mode,
                          flattop_precision=flattop_precision,
                          r=r)
        except ValueError as verr:
            # Handles invalid parameter values (e.g., alpha not in [0,1])
            return add_stopping_hint(f'ValueError occurred: {str(verr)}')
        except KeyError as kerr:
            # Handles missing columns in the DataFrame
            return add_stopping_hint(f'KeyError occurred: {str(kerr)}')
        except TypeError as terr:
            # Handles type mismatches (e.g., non-numeric input where number expected)
            return add_stopping_hint(f'TypeError occurred: {str(terr)}')
        except ProgrammingError as perr:
            # Handles invalid table name specifically
            if 'invalid table name' in str(perr):
                return add_stopping_hint(f'Invalid table name: Could not find table/view {table_name}')
        fft_res_tab = remove_prefix_sharp(f"{table_name}_FFT_RESULT")
        fft_res.save(fft_res_tab, force=True)
        return json.dumps({"fft_result_table" : fft_res_tab})

    async def _arun(self,#pylint:disable=too-many-positional-arguments
                    table_name : str,
                    key : str,
                    real_col : str=None,
                    imag_col : str=None,
                    inverse : bool=None,
                    window : str=None,
                    window_start : int=None,
                    window_length : int=None,
                    alpha : float=None,
                    beta : float=None,
                    attenuation : float=None,
                    flattop_mode : str=None,
                    flattop_precision : str=None,
                    r : float=None,
                    run_manager: AsyncCallbackManagerForToolRun = None#pylint:disable=unused-argument
                    )-> str:
        """Use the tool asynchronously."""
        return self._run(table_name=table_name,
                         key=key,
                         real_col=real_col,
                         imag_col=imag_col,
                         inverse=inverse,
                         window=window,
                         window_start=window_start,
                         window_length=window_length,
                         alpha=alpha,
                         beta=beta,
                         attenuation=attenuation,
                         flattop_mode=flattop_mode,
                         flattop_precision=flattop_precision,
                         r=r,
                         run_manager=run_manager)
