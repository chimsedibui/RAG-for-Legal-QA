"""LLMProvider implementation backed by any OpenAI-compatible chat endpoint
(vLLM, LM Studio, llama.cpp server, or the real OpenAI API)."""
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI


class OpenAILLMProvider:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_name = model_name

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Generator[Any, None, None]:
        kwargs: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "stream": stream,
            # vLLM/Qwen3-specific knob to disable "thinking" traces; harmless
            # no-op extra_body for OpenAI-compatible backends that ignore it.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self._client.chat.completions.create(**kwargs)

            if stream:
                for chunk in response:
                    yield chunk
            else:
                yield response.choices[0].message
        except Exception as e:
            error_msg = f"Lỗi khi gọi LLM: {str(e)}"
            print(error_msg)
            if stream:
                yield {"error": error_msg}
            else:
                raise Exception(error_msg)
