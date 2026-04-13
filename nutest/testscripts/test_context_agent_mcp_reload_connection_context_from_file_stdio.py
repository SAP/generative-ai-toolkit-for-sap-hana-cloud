#!/usr/bin/env python3
"""E2E (opt-in): Use ContextAgent to reload MCP stdio server ConnectionContext from file.

Flow required by this test:
1) Start MCP server over stdio using HANA userkey 'RaysKey'
2) Through a ContextAgent chat prompt, call admin_reload_connection_context_from_file
   with the committed fixture file nutest/testscripts/env.example
3) Assert tool call succeeded and the server switched its ConnectionContext instance

Enable with:
  RUN_CONTEXT_AGENT_MCP_STDIO_E2E=1

Notes:
- Requires a live HANA connection for both RaysKey and the credentials inside env.example.
- Requires an LLM provider setup for gen_ai_hub.proxy.langchain.init_llm (tool calling).
"""

from __future__ import annotations

import asyncio
import os
import threading
import unittest
from pathlib import Path
from typing import Any, Callable, Optional


class _AsyncLoopThread:
    """Run an asyncio loop in a background thread and execute coroutines on it."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        ready = threading.Event()

        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_runner, name="async-loop-thread", daemon=True)
        self._thread.start()
        ready.wait(timeout=5)
        if self._loop is None:
            raise RuntimeError("Failed to start asyncio loop")

    def run(self, coro):
        if self._loop is None:
            raise RuntimeError("Async loop not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=60)

    def stop(self) -> None:
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass


class TestContextAgentMCPReloadConnectionContextFromFileSTDIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RUN_CONTEXT_AGENT_MCP_STDIO_E2E") != "1":
            raise unittest.SkipTest(
                "Set RUN_CONTEXT_AGENT_MCP_STDIO_E2E=1 to run this live ContextAgent+MCP stdio e2e test."
            )

        # Fixed fixture file (per requirement)
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.vcap_file = (cls.repo_root / "nutest" / "testscripts" / "env.example").resolve()
        if not cls.vcap_file.is_file():
            raise unittest.SkipTest(f"Missing fixture file: {cls.vcap_file}")

        # LLM for tool calling
        try:
            from gen_ai_hub.proxy.langchain import init_llm
        except Exception as e:
            raise unittest.SkipTest(f"gen_ai_hub init_llm unavailable: {e}")
        cls.llm = init_llm(os.environ.get("CONTEXT_AGENT_E2E_LLM_MODEL", "gpt-4.1"), temperature=0.0, max_tokens=800)

        # Start stdio MCP server as subprocess via StdioMCPClient (RaysKey init)
        from hana_ai.client.mcp_client import StdioMCPClient

        cls._loop_thread = _AsyncLoopThread()
        cls._loop_thread.start()

        cls._client = StdioMCPClient(
            command="python",
            args=[
                "examples/mcp_stdio_server.py",
                "--server-name",
                "HANATools",
                "--userkey",
                "RaysKey",
                "--encrypt",
                "true",
                "--ssl-validate",
                "false",
            ],
            server_name="HANATools",
        )

        cls._loop_thread.run(cls._client.initialize())

    @classmethod
    def tearDownClass(cls):
        try:
            cls._loop_thread.run(cls._client.close())
        except Exception:
            pass
        try:
            cls._loop_thread.stop()
        except Exception:
            pass

    def _make_stdio_tool(self):
        from hana_ai.langchain_compat import Tool

        client = self._client
        runner = self._loop_thread

        def _call_admin_reload(file_path: str, test_connection: bool = False) -> Any:
            res = runner.run(client.call_tool(
                "admin_reload_connection_context_from_file",
                {"file_path": file_path, "test_connection": test_connection},
            ))
            if not res.success:
                return {"ok": False, "error": res.error}
            data = res.data
            if isinstance(data, str):
                try:
                    import json
                    data = json.loads(data)
                except Exception:
                    pass
            return data

        return Tool.from_function(
            func=_call_admin_reload,
            name="admin_reload_connection_context_from_file",
            description="Reload the MCP server ConnectionContext from a VCAP_SERVICES file path.",
        )

    def test_context_agent_reload_switches_connection(self):
        from hana_ai.iagents.context_agent import AgentConfig, ContextAgent

        tool = self._make_stdio_tool()
        agent = ContextAgent(
            llm=self.llm,
            tools=[tool],
            storage_dir=str(Path(os.getenv("TMPDIR", "/tmp")) / "context_agent_mcp_stdio_e2e"),
            config=AgentConfig(skills_use_llm_selector=True, max_active_skills=2, skills_cache_turns=0),
            progress_bar=False,
        )

        prompt = (
            "You MUST call the tool admin_reload_connection_context_from_file exactly once. "
            f"Set file_path to {str(self.vcap_file)} and set test_connection=false. "
            "Then return the tool result as-is."
        )

        out = agent.chat(prompt)
        self.assertIsInstance(out, str)
        self.assertIn("[Tool Return]", out)
        self.assertRegex(out, r"\"ok\"\s*:\s*true")
        self.assertRegex(out, r"\"switched\"\s*:\s*true")
        # Ensure we show previous + new connection summaries in response
        self.assertIn("previous_connection", out)
        self.assertIn("connection", out)


if __name__ == "__main__":
    unittest.main()
