"""
This module defines a agent tool for fft() function in hana-ml.
"""
import json
import logging
from typing import Type, Tuple, Union
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.algorithms.pal.tsa.fft import fft
from hana_ai.utility import remove_prefix_sharp

logger = logging.getLogger(__name__)

class FFTInput(BaseModel):
    """
    Input for fft function.
    """
    table_name : str = Field(description="Table (or view) that contains the input sequence data for applying FFT or inverse FFT. If not provided, ask the user, do not guess")
    key : str = Field(description="the key of the dataset. If not provided, ask the user. Do not guess")
    data_cols : Union[str, Tuple[str, str]] = Field(description="The columns, i.e. real part and imaginary part of the sequence data" +\
    " involved for computing FFT. It can be a tuple of two columns, with the 1st column being the real part "+\
    "and the 2nd column being the imaginary part. It can also be a single column indicating the input sequence is pure real or imaginary. "+\
    "If not provided, ask the user. Do not guess")
    num_type : str = Field(escription="Specifies the number type of the sequence data, i.e. 'real' for pure real data" +\
    " and 'imag' for pure imaginary data if `data_cols` is a single column.", default=None)
    inverse : bool = Field(description="Specifies whether inverse FFT is applied or not. Regular forward FFT is applied if it is set as False," +\
    " otherwise inverse FFT is applied.", default=None)
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
                * - data_cols
                  - Specifies the columns, i.e. real part and imaginary part of the sequence data involved for computing FFT.
                    It can be a tuple of two columns, with the 1st column being the real part and the 2nd column being the
                    imaginary part. It can also be a single column indicating the input sequence is pure real or imaginary.
                * - num_type
                  - Specifies the number type of the sequence data, i.e. 'real' for pure real data
                    and 'imag' for pure imaginary data if `data_cols` is a single column.
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
    name: str = "Applying fast-Fourier transform (FFT) or inverse FFT to time-series data."
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
        data_cols : Union[str, Tuple[str, str]],
        num_type : str=None,
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
        cols = [data_cols] if isinstance(data_cols, str) else list(data_cols)
        input_data = self.connection_context.table(table_name)[[key] + cols]
        if "INT" not in input_data[[key]].dtypes()[0][1]:
            input_data = input_data.add_id(f"{key}" + "_int", ref_col=key).deselect(key)
        fft_res = fft(data=input_data,
                      num_type=num_type, inverse=inverse, window=window,
                      window_start=window_start, window_length=window_length,
                      alpha=alpha, beta=beta, attenuation=attenuation,
                      flattop_mode=flattop_mode,
                      flattop_precision=flattop_precision,
                      r=r)
        fft_res_tab = remove_prefix_sharp(f"{table_name}_FFT_RESULT")
        fft_res.save(fft_res_tab, force=True)
        return json.dumps({"fft_result_table" : fft_res_tab})

    async def _arun(self,#pylint:disable=too-many-positional-arguments
                    table_name : str,
                    key : str,
                    data_cols : Union[str, Tuple[str, str]],
                    num_type : str=None,
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
                         data_cols=data_cols,
                         num_type=num_type,
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
