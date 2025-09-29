====
FAQS
====

**Q: Get the error "'Mime type rendering requires nbformat>=4.2.0 but it is not installed'" in forecast_line_plot tool**
A: The issue is not due to the nbformat but plotly outputs the misleading error message. The issue is mainly due to the incompatibility plotly and Jupyter 4.0. You can find the solution from https://github.com/plotly/plotly.py/issues/4354#issuecomment-1851695343.
