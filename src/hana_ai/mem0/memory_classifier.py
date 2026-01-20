from __future__ import annotations

from typing import Any, Dict, Optional
import json

DEFAULT_CLASSIFIER_PROMPT = (
    "You are a bilingual (中文/English) memory ingestion classifier. "
    "Classify the text into one: preference, fact, task, session_state, ephemeral. "
    "Output strictly one JSON object: {\"category\": string, \"tags\": string[], \"priority\": float 0-1, \"tier\": short|long, \"ttl_seconds\": integer|null}. "
    "Guidance: preferences/facts -> long-term; tasks/session_state/ephemeral -> short-term with appropriate TTL. "
    "Text:\n\n{content}\n\n"
    "Examples:\n"
    "- '我喜欢无糖拿铁。' -> {\"category\": \"preference\", \"tags\":[\"coffee\",\"preference\"], \"priority\":0.9, \"tier\":\"long\", \"ttl_seconds\": null}\n"
    "- 'Order #123 due tomorrow' -> {\"category\": \"task\", \"tags\":[\"order\",\"deadline\"], \"priority\":0.8, \"tier\":\"short\", \"ttl_seconds\": 86400}\n"
    "- 'SAP HANA is an in-memory database.' -> {\"category\": \"fact\", \"tags\":[\"sap\",\"hana\"], \"priority\":0.7, \"tier\":\"long\", \"ttl_seconds\": null}\n"
)


class Mem0IngestionClassifier:
    def __init__(self, llm: Any, prompt_template: Optional[str] = None, examples: Optional[str] = None) -> None:
        self.llm = llm
        base = prompt_template or DEFAULT_CLASSIFIER_PROMPT
        self.prompt_template = base if examples is None else (base + "\n" + examples)

    def classify(self, text: str) -> Dict[str, Any]:
        # Avoid str.format on JSON braces by only replacing the {content} token
        prompt = self.prompt_template.replace("{content}", text)
        try:
            # Prefer invoke API
            if hasattr(self.llm, "invoke"):
                out = self.llm.invoke(prompt)
                content = getattr(out, "content", None) if out is not None else None
                if content is None and isinstance(out, dict):
                    content = out.get("content")
                if content is None and isinstance(out, str):
                    content = out
            else:
                # Fallback to call/ __call__
                out = self.llm(prompt)
                content = getattr(out, "content", None) if out is not None else None
                if content is None and isinstance(out, dict):
                    content = out.get("content")
                if content is None and isinstance(out, str):
                    content = out

            data = json.loads(content) if isinstance(content, str) else {}
            category = str(data.get("category", "ephemeral"))
            tags = data.get("tags", [])
            priority = float(data.get("priority", 0.5))
            tier = str(data.get("tier", "short"))
            ttl_seconds_val = data.get("ttl_seconds")
            ttl_seconds = int(ttl_seconds_val) if ttl_seconds_val is not None else None
            return {
                "category": category,
                "tags": tags if isinstance(tags, list) else [],
                "priority": priority,
                "tier": tier if tier in ("short", "long") else "short",
                "ttl_seconds": ttl_seconds,
            }
        except Exception:
            return {
                "category": "ephemeral",
                "tags": [],
                "priority": 0.5,
                "tier": "short",
                "ttl_seconds": None,
            }
