"""Offline checks for grouped-index adaptation and truthful E2E tool evidence."""

from unittest.mock import Mock

import pytest

from hana_ai.tools.hana_ml_tools.massive_ts_outlier_detection_tools import _fit_predict_grouped


@pytest.mark.parametrize("data_type", ["INTEGER", "BIGINT", "SMALLINT"])
def test_integer_index_retains_original_pal_path(data_type):
    data = Mock()
    data.get_table_structure.return_value = {"DAY": data_type}
    detector = Mock()
    result = _fit_predict_grouped(detector, data, key="DAY", group_key="STORE", endog="VALUE")
    assert result is detector.fit_predict.return_value
    detector.fit_predict.assert_called_once_with(data, key="DAY", group_key="STORE", endog="VALUE")
    data.connection_context.sql.assert_not_called()


@pytest.mark.parametrize("data_type", ["DATE", "TIMESTAMP", "SECONDDATE", "NVARCHAR(30)"])
def test_noninteger_index_uses_partitioned_rows_and_data_join(data_type):
    data = Mock()
    data.get_table_structure.return_value = {'Day"Key': data_type}
    data.select_statement = 'SELECT * FROM "SOURCE"'
    numbered = Mock(select_statement='SELECT * FROM "NUMBERED"')
    mapped = Mock()
    data.connection_context.sql.side_effect = [numbered, mapped]
    detector = Mock()
    detector.fit_predict.return_value.columns = ["GROUP_ID", "TIMESTAMP", "VALUE", "IS_OUTLIER"]
    detector.fit_predict.return_value.select_statement = 'SELECT * FROM "PAL_RESULT"'
    result = _fit_predict_grouped(detector, data, key='Day"Key', group_key='Store"ID', endog="VALUE")
    assert result is mapped
    detector.fit_predict.assert_called_once_with(
        numbered.select.return_value, group_key="__GROUP", key="__ROW", endog="__VALUE",
    )
    numbering, mapping = [call.args[0] for call in data.connection_context.sql.call_args_list]
    assert 'PARTITION BY "Store""ID" ORDER BY "Day""Key"' in numbering
    assert 'AS INTEGER) AS "__ROW"' in numbering
    assert 'M."__KEY" AS "Day""Key"' in mapping
    assert 'TO_NVARCHAR(M."__GROUP") = R."GROUP_ID"' in mapping
    assert 'M."__ROW" = R."TIMESTAMP"' in mapping
    data.filter.assert_not_called()


@pytest.mark.parametrize("bad_column", ["KEY;SELECT", "KEY\x00"])
def test_invalid_identifier_is_rejected_before_execution(bad_column):
    data = Mock()
    data.get_table_structure.return_value = {bad_column: "TIMESTAMP"}
    data.select_statement = 'SELECT * FROM "SOURCE"'
    detector = Mock()
    with pytest.raises(ValueError):
        _fit_predict_grouped(detector, data, key=bad_column, group_key="STORE", endog="VALUE")
    detector.fit_predict.assert_not_called()
    data.connection_context.sql.assert_not_called()


def test_evidence_records_actual_structured_output_not_response_wording():
    from langchain_core.tools import StructuredTool
    from test_context_agent_e2e_scenarios import _ToolEvidence

    evidence = _ToolEvidence()

    def artifact_tool(table_name: str) -> str:
        """Return a synthetic report payload for testing callback capture only."""
        assert table_name == "DATA"
        return '{"html_file": "C:/reports/plot.html"}'

    tool = StructuredTool.from_function(artifact_tool, callbacks=[evidence])
    output = tool.invoke({"table_name": "DATA"})
    assert output == '{"html_file": "C:/reports/plot.html"}'
    assert evidence.records == [{"name": "artifact_tool", "inputs": {"table_name": "DATA"},
                                 "output": {"html_file": "C:/reports/plot.html"}}]
    assert evidence.pending == {}


def test_evidence_records_real_errors_without_inventing_success():
    from langchain_core.tools import StructuredTool
    from test_context_agent_e2e_scenarios import _ToolEvidence

    evidence = _ToolEvidence()

    def failing_tool(value: str) -> str:
        """Fail deliberately to verify E2E error capture."""
        raise ValueError(value)

    tool = StructuredTool.from_function(failing_tool, callbacks=[evidence])
    with pytest.raises(ValueError, match="synthetic failure"):
        tool.invoke({"value": "synthetic failure"})
    assert evidence.records[0]["output"] == {"error": "synthetic failure"}
    assert evidence.pending == {}


@pytest.mark.parametrize("over_budget", [False, True])
def test_scenario_prompt_budget_prevents_silently_truncated_requests(over_budget):
    from test_context_agent_e2e_scenarios import TestContextAgentE2EScenarios, _ToolEvidence

    case = TestContextAgentE2EScenarios(methodName="runTest")
    case._evidence = _ToolEvidence()
    agent = Mock()
    agent.config.budgets.budget_task = 100
    agent.chat.return_value = "completed"
    prompt = "x" * (100 - len("User question: ") + int(over_budget))
    if over_budget:
        with pytest.raises(AssertionError):
            case._chat(agent, prompt)
        agent.chat.assert_not_called()
    else:
        assert case._chat(agent, prompt) == "completed"
        agent.chat.assert_called_once_with(prompt)
