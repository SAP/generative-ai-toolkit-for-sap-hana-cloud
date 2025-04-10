Changelog
=========

**Version 1.0.250410**

``Enhancements``
    - Enhanced the HANA SQL agent to support case-sensitive SQL queries.
    - Added create_hana_sql_toolkit function to create a toolkit for HANA SQL.

``Bug Fixes``
    - Fixed the accuracy_measure tool issue when evaluation_metric="spec".

**Version 1.0.250407**

``Enhancements``
    - Improved `forecast_line_plot` tool to automatically detect the confidence if it is not provided.
    - Serialized the tool's return if it is pandas DataFrame when `return_direct` is set to `False`.

``Bug Fixes``
    - Fixed the json serialization issue when the tool's return contains Timestamp.

**Version 1.0.250403**

``New Functions``
    - Added `list_models` tool to list all trained models in the model storage.
    - Added `accuracy_measure` tool to measure the accuracy of a model on a test dataset for time series forecasting.

``Enhancements``
    - Improved the `intermittent_forecast` tool to use CrostonTSB instead.
    - Added parameter `return_direct` to all tools and toolkit.
    - Improved the `fetch_data` tool to return a pandas DataFrame instead of a list of dictionaries. By default, the tool parameter `return_direct` is set to `True`, which means the tool will return a pandas DataFrame.
