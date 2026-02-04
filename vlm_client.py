import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass
class VLMClient:
    endpoint: str
    api_key: str
    model: str
    prompt: str
    timeout_s: int = 60

    @classmethod
    def from_env(cls) -> Optional["VLMClient"]:
        endpoint = os.getenv("VLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
        api_key = os.getenv("VLM_API_KEY")
        model = os.getenv("VLM_MODEL", "gpt-4o-mini")
        prompt = os.getenv("VLM_PROMPT", "请描述这张图片的主要内容，用中文回答。")
        if not api_key:
            return None
        return cls(endpoint=endpoint, api_key=api_key, model=model, prompt=prompt)

    def describe_image(self, image_path: Path) -> Optional[str]:
        try:
            payload = self._build_payload(image_path)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            return self._extract_text(data)
        except Exception:
            return None

    def _build_payload(self, image_path: Path) -> dict:
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        ext = image_path.suffix.lstrip(".") or "png"
        data_url = f"data:image/{ext};base64,{encoded}"
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 256,
        }

    @staticmethod
    def _extract_text(data: dict) -> Optional[str]:
        choices = data.get("choices")
        if not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        return None
