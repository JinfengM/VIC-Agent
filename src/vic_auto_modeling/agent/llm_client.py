import os

import httpx
from openai import OpenAI


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class QwenVllmClient:
    def __init__(
        self,
        base_url=None,
        model=None,
        api_key=None,
        temperature=None,
        timeout=None,
        proxy=None,
        trust_env=None,
    ):
        configured_base_url = base_url or os.getenv("VIC_AGENT_BASE_URL")
        if not configured_base_url:
            raise ValueError(
                "Set VIC_AGENT_BASE_URL or pass base_url when creating QwenVllmClient"
            )
        self.base_url = configured_base_url.rstrip("/")
        self.model = model or os.getenv("VIC_AGENT_MODEL", "opt-120b")
        self.api_key = api_key or os.getenv("VIC_AGENT_API_KEY", "EMPTY")
        self.temperature = float(
            temperature
            if temperature is not None
            else os.getenv("VIC_AGENT_TEMPERATURE", "0.2")
        )
        timeout = float(
            timeout if timeout is not None else os.getenv("VIC_AGENT_TIMEOUT", "120")
        )
        if trust_env is None:
            trust_env = _env_bool("VIC_AGENT_TRUST_ENV", False)
        proxy = proxy if proxy is not None else os.getenv("VIC_AGENT_PROXY")
        client_args = {"timeout": timeout, "trust_env": bool(trust_env)}
        if proxy:
            client_args["proxy"] = proxy
        self.http_client = httpx.Client(**client_args)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0,
            http_client=self.http_client,
        )

    def list_models(self):
        if self.base_url.endswith("/api/v1"):
            models_url = f"{self.base_url[:-7]}/api/models"
        else:
            models_url = f"{self.base_url}/models"
        response = self.http_client.get(
            models_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        return response.json().get("data", [])

    def chat(self, messages, max_tokens=1024):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream_chat(self, messages, max_tokens=1024):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
