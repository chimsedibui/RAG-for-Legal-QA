"""LLMProvider implementation backed by any OpenAI-compatible chat endpoint
(vLLM, LM Studio, llama.cpp server, or the real OpenAI API)."""
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI
from urllib.parse import urlparse


class OpenAILLMProvider:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_name = model_name
        self._is_openai = urlparse(base_url).hostname == "api.openai.com"

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> Generator[Any, None, None]:
        kwargs: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "stream": stream,
        }
        if not self._is_openai:
            # Chỉ có ý nghĩa với model self-host kiểu Qwen3 (real OpenAI không
            # có chat_template_kwargs này). Mặc định tắt "thinking mode" trừ
            # khi người dùng bật toggle "cho phép suy luận sâu" trên UI.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

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
