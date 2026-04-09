import tempfile
import unittest

from hana_ai.iagents.context_agent import AgentConfig, ContextAgent


class TestContextAgentMemoryCommands(unittest.TestCase):
	def setUp(self):
		self.temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(self.temp_dir.cleanup)

	def _make_agent(self):
		return ContextAgent(
			llm=lambda prompt: "ok",
			tools=[],
			storage_dir=self.temp_dir.name,
			session_id="test_session",
			config=AgentConfig(skills_use_llm_selector=True, max_active_skills=4, skills_cache_turns=0),
		)

	def test_clear_notes_resets_global_memory_files(self):
		agent = self._make_agent()
		agent._append_notes(
			{
				"decisions": ["keep the latest model"],
				"todos": ["rerun evaluation"],
				"constraints": ["must preserve chronology"],
				"preferences": ["prefer concise replies"],
				"facts": ["table SALES exists"],
			}
		)

		response = agent.chat("!clear_notes")

		self.assertEqual(response, "Cleared NOTES, TODO, DECISIONS, and CONTEXT.")
		self.assertEqual((agent.storage_dir / "NOTES.md").read_text(encoding="utf-8"), "# NOTES\n\n")
		self.assertEqual((agent.storage_dir / "TODO.md").read_text(encoding="utf-8"), "# TODO\n\n")
		self.assertEqual((agent.storage_dir / "DECISIONS.md").read_text(encoding="utf-8"), "# DECISIONS\n\n")
		self.assertEqual((agent.storage_dir / "CONTEXT.md").read_text(encoding="utf-8"), "# CONTEXT\n\n")

	def test_clear_session_resets_chat_and_summary(self):
		agent = self._make_agent()
		agent._append_chat("user", "hello")
		agent._write_session_summary("Goal\n- keep going")

		response = agent.chat("!clear_session")

		self.assertEqual(response, "Cleared chat and session summary for session 'test_session'.")
		self.assertEqual(agent._chat_path().read_text(encoding="utf-8"), "# Session test_session\n\n")
		self.assertEqual(agent._summary_path().read_text(encoding="utf-8"), "# Session test_session\n\n")

	def test_individual_note_commands_only_reset_their_target_files(self):
		agent = self._make_agent()
		agent._append_notes(
			{
				"decisions": ["keep model_b"],
				"todos": ["rerun scoring"],
				"constraints": ["keep source data immutable"],
				"preferences": ["prefer SQL fallback"],
				"facts": ["workspace is seeded"],
			}
		)

		self.assertEqual(agent.chat("!clear_notes_file"), "Cleared NOTES.")
		self.assertEqual((agent.storage_dir / "NOTES.md").read_text(encoding="utf-8"), "# NOTES\n\n")
		self.assertIn("rerun scoring", (agent.storage_dir / "TODO.md").read_text(encoding="utf-8"))

		self.assertEqual(agent.chat("!clear_todo"), "Cleared TODO.")
		self.assertEqual((agent.storage_dir / "TODO.md").read_text(encoding="utf-8"), "# TODO\n\n")
		self.assertIn("keep model_b", (agent.storage_dir / "DECISIONS.md").read_text(encoding="utf-8"))

		self.assertEqual(agent.chat("!clear_decisions"), "Cleared DECISIONS.")
		self.assertEqual((agent.storage_dir / "DECISIONS.md").read_text(encoding="utf-8"), "# DECISIONS\n\n")

		self.assertEqual(agent.chat("!clear_context"), "Cleared CONTEXT.")
		self.assertEqual((agent.storage_dir / "CONTEXT.md").read_text(encoding="utf-8"), "# CONTEXT\n\n")

	def test_individual_session_commands_only_reset_their_target_files(self):
		agent = self._make_agent()
		agent._append_chat("user", "hello")
		agent._write_session_summary("Current state\n- summary exists")

		self.assertEqual(agent.chat("!clear_chat"), "Cleared chat history for session 'test_session'.")
		self.assertEqual(agent._chat_path().read_text(encoding="utf-8"), "# Session test_session\n\n")
		self.assertIn("summary exists", agent._summary_path().read_text(encoding="utf-8"))

		agent._append_chat("assistant", "recreated chat")
		self.assertEqual(agent.chat("!clear_summary"), "Cleared session summary for session 'test_session'.")
		self.assertEqual(agent._summary_path().read_text(encoding="utf-8"), "# Session test_session\n\n")
		self.assertIn("recreated chat", agent._chat_path().read_text(encoding="utf-8"))

	def test_reset_memory_clears_both_global_and_session_memory(self):
		agent = self._make_agent()
		agent._append_notes(
			{
				"decisions": ["use model_a"],
				"todos": ["score holdout"],
				"constraints": [],
				"preferences": [],
				"facts": ["session has prior notes"],
			}
		)
		agent._append_chat("assistant", "stored something")
		agent._write_session_summary("Current state\n- something stored")

		response = agent.chat("!reset_memory")

		self.assertEqual(response, "Reset memory notes and session state for session 'test_session'.")
		self.assertEqual((agent.storage_dir / "NOTES.md").read_text(encoding="utf-8"), "# NOTES\n\n")
		self.assertEqual((agent.storage_dir / "TODO.md").read_text(encoding="utf-8"), "# TODO\n\n")
		self.assertEqual((agent.storage_dir / "DECISIONS.md").read_text(encoding="utf-8"), "# DECISIONS\n\n")
		self.assertEqual((agent.storage_dir / "CONTEXT.md").read_text(encoding="utf-8"), "# CONTEXT\n\n")
		self.assertEqual(agent._chat_path().read_text(encoding="utf-8"), "# Session test_session\n\n")
		self.assertEqual(agent._summary_path().read_text(encoding="utf-8"), "# Session test_session\n\n")


if __name__ == "__main__":
	unittest.main()