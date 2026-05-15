#!/usr/bin/env python3
"""E2E (opt-in): Use ContextAgent to call MCP admin_reload_connection_context_from_file.

This test is intentionally opt-in because it requires:
- a reachable HANA instance matching the credentials in the VCAP_SERVICES file
- a working LLM setup for gen_ai_hub.proxy.langchain.init_llm (tool calling)

Enable with:
  RUN_CONTEXT_AGENT_MCP_E2E=1

Optional overrides:
  VCAP_SERVICES_FILE=/absolute/or/relative/path/to/env.file
  CONTEXT_AGENT_E2E_LLM_MODEL=gpt-4.1
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import unittest
from pathlib import Path
from typing import Any, Dict


def _find_free_port(start: int = 8000, end: int = 8100) -> int:
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestContextAgentMCPReloadConnectionContextFromFileHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RUN_CONTEXT_AGENT_MCP_E2E") != "1":
            raise unittest.SkipTest("Set RUN_CONTEXT_AGENT_MCP_E2E=1 to run this live ContextAgent+MCP e2e test.")

        # Resolve VCAP_SERVICES file
        override = os.environ.get("VCAP_SERVICES_FILE")
        if override:
            cls.vcap_file = Path(override).expanduser().resolve()
        else:
            # Default: committed example file
            repo_root = Path(__file__).resolve().parents[2]
            cls.vcap_file = (repo_root / "nutest" / "testscripts" / "env.example").resolve()
        if not cls.vcap_file.is_file():
            raise unittest.SkipTest(f"VCAP_SERVICES file not found: {cls.vcap_file}")

        # Build connection context from file
        from hana_ai.tools import toolkit as tk_mod
        from hana_ml import ConnectionContext
        from hana_ai.tools.toolkit import HANAMLToolkit

        params = tk_mod._build_cc_params_from_vcap_services_file(str(cls.vcap_file))
        cls.cc = ConnectionContext(**params)

        # Start MCP server (HTTP)
        cls.tk = HANAMLToolkit(connection_context=cls.cc, used_tools=["fetch_data"])
        cls.port = _find_free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}/mcp"
        cls.tk.launch_mcp_server(transport="http", host="127.0.0.1", port=cls.port, max_retries=5)
        time.sleep(1.0)

        # Initialize LLM for ContextAgent (must support tool calling)
        try:
            from gen_ai_hub.proxy.langchain import init_llm
        except Exception as e:
            raise unittest.SkipTest(f"gen_ai_hub init_llm unavailable: {e}")
        model_name = os.environ.get("CONTEXT_AGENT_E2E_LLM_MODEL", "gpt-4.1")
        cls.llm = init_llm(model_name, temperature=0.0, max_tokens=800)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.tk.stop_mcp_server(host="127.0.0.1", port=cls.port, transport="http", force=True, timeout=3.0)
        except Exception:
            pass
        try:
            cls.cc.close()
        except Exception:
            pass

    def _make_reload_tool(self):
        from hana_ai.client.mcp_client import HTTPMCPClient
        from hana_ai.langchain_compat import Tool

        base_url = self.base_url

        def _call_reload(file_path: str, test_connection: bool = False) -> Any:
            async def _main() -> Any:
                client = HTTPMCPClient(base_url=base_url, timeout=30)
                try:
                    await client.initialize()
                    res = await client.call_tool(
                        "admin_reload_connection_context_from_file",
                        {"file_path": file_path, "test_connection": test_connection},
                    )
                    if not res.success:
                        return {"ok": False, "error": res.error}
                    return res.data
                finally:
                    try:
                        await client.close()
                    except Exception:
                        pass

            return asyncio.run(_main())

        return Tool.from_function(
            func=_call_reload,
            name="mcp_reload_connection_context_from_file",
            description=(
                "Calls the MCP server admin tool 'admin_reload_connection_context_from_file' to reload the "
                "HANA ConnectionContext from a VCAP_SERVICES env-style file path."
            ),
        )

    def test_context_agent_calls_mcp_reload_from_file(self):
        from hana_ai.iagents.context_agent import AgentConfig, ContextAgent

        tool = self._make_reload_tool()
        agent = ContextAgent(
            llm=self.llm,
            tools=[tool],
            storage_dir=str(Path(os.getenv("TMPDIR", "/tmp")) / "context_agent_mcp_e2e"),
            config=AgentConfig(skills_use_llm_selector=True, max_active_skills=2, skills_cache_turns=0),
            progress_bar=False,
        )

        prompt = (
            "You MUST call the tool mcp_reload_connection_context_from_file exactly once. "
            "Set file_path to this exact path and set test_connection=false. "
            f"file_path={str(self.vcap_file)}. "
            "Then return the tool result as-is."
        )

        out = agent.chat(prompt)
        self.assertIsInstance(out, str)
        self.assertIn("[Tool Return]", out)
        self.assertIn("mcp_reload_connection_context_from_file", out)
        # Tool output is expected to include ok=True and redacted password.
        self.assertRegex(out, r"\"ok\"\s*:\s*true")
        self.assertRegex(out, r"\"password\"\s*:\s*\"\*\*\*\"")


if __name__ == "__main__":
    unittest.main()
