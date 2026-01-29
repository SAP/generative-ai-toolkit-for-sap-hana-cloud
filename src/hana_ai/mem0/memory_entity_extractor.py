"""Entity extractor using LLMs."""
from __future__ import annotations

from typing import Any, Dict, Optional
import json
import re

BILINGUAL_ENTITY_PROMPT = (
    "You are an entity extraction assistant. Extract the PRIMARY entity mentioned in the text. "
    "Return strictly one JSON object: {\"entity_name\": string, \"entity_type\": string, \"entity_id\": string}. "
    "Entity types: person, organization, product, location, event, other. "
    "If no clear entity exists, use entity_name='unknown', entity_type='other', entity_id='unknown'. "
    "Text (中文/English supported):\n\n{content}\n\n"
    "Examples:\n"
    "- Input: '我喜欢星巴克的拿铁。' -> {\"entity_name\": \"星巴克\", \"entity_type\": \"organization\", \"entity_id\": \"xing-ba-ke\"}\n"
    "- Input: 'I enjoy using iPhone 15 Pro.' -> {\"entity_name\": \"iPhone 15 Pro\", \"entity_type\": \"product\", \"entity_id\": \"iphone-15-pro\"}\n"
    "- Input: 'Let's meet in Berlin next week.' -> {\"entity_name\": \"Berlin\", \"entity_type\": \"location\", \"entity_id\": \"berlin\"}\n"
)


def slugify(text: str) -> str:
    """Generate a simple slug from the given text."""
    t = text.strip().lower()
    t = re.sub(r"[\s_]+", "-", t)
    t = re.sub(r"[^a-z0-9\-]+", "", t)
    return t or "unknown"


class Mem0EntityExtractor:
    """Extract primary entity from text using an LLM."""
    def __init__(self, llm: Any, prompt_template: Optional[str] = None) -> None:
        self.llm = llm
        self.prompt_template = prompt_template or BILINGUAL_ENTITY_PROMPT

    def extract(self, text: str) -> Dict[str, str]:
        """Extract entity details from the given text."""
        # Avoid str.format on JSON braces by only replacing the {content} token
        prompt = self.prompt_template.replace("{content}", text)
        try:
            if hasattr(self.llm, "invoke"):
                out = self.llm.invoke(prompt)
                content = getattr(out, "content", None) if out is not None else None
                if content is None and isinstance(out, dict):
                    content = out.get("content")
                if content is None and isinstance(out, str):
                    content = out
            else:
                out = self.llm(prompt)
                content = getattr(out, "content", None) if out is not None else None
                if content is None and isinstance(out, dict):
                    content = out.get("content")
                if content is None and isinstance(out, str):
                    content = out
            data = json.loads(content) if isinstance(content, str) else {}
            name = str(data.get("entity_name", "unknown")).strip() or "unknown"
            etype = str(data.get("entity_type", "other"))
            eid = str(data.get("entity_id") or slugify(name))
            return {"entity_name": name, "entity_type": etype, "entity_id": eid}
        except Exception:
            return {"entity_name": "unknown", "entity_type": "other", "entity_id": "unknown"}
