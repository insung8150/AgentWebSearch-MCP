"""
LM Studio Adapter

Handles LM Studio's OpenAI-compatible API with function calling support.
LM Studio 0.4+ supports Anthropic-style tool use.
"""

import json
import re
import httpx

from .base import BaseLLMAdapter, AdapterConfig, LLMResponse, ToolCall


class LMStudioAdapter(BaseLLMAdapter):
    """Adapter for LM Studio server

    LM Studio provides OpenAI-compatible API at /v1/chat/completions.
    Function calling support depends on the loaded model.
    """

    DEFAULT_URL = "http://localhost:1234"

    def __init__(self, url: str | None = None, model: str = "", **kwargs):
        config = AdapterConfig(
            url=url or self.DEFAULT_URL,
            model=model,  # Empty = use currently loaded model
            **kwargs
        )
        super().__init__(config)
        self._supports_tools = True  # LM Studio 0.4+ supports tools

    @property
    def name(self) -> str:
        return "lmstudio"

    @property
    def tool_format(self) -> str:
        return "function_calling"

    async def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send messages to LM Studio server"""
        api_url = f"{self.config.url}/v1/chat/completions"

        # Build request payload
        payload = {
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

        # Add model if specified
        if self.config.model:
            payload["model"] = self.config.model

        # Add tools if provided
        if tools and self._supports_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(api_url, json=payload)
            data = response.json()

        # Extract response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # Check for tool calls
        tool_calls_data = message.get("tool_calls", [])

        llm_response = LLMResponse(
            content=content,
            tool_calls=[],
            raw_response=data,
            model=data.get("model", self.config.model),
            usage=data.get("usage", {}),
        )

        # Parse tool calls
        if tool_calls_data:
            llm_response.tool_calls = self._parse_function_calls(tool_calls_data)
        else:
            # Try text-based parsing as fallback
            llm_response.tool_calls = self._parse_text_tool_calls(content)

        return llm_response

    def _parse_function_calls(self, tool_calls_data: list) -> list[ToolCall]:
        """Parse OpenAI function calling format"""
        tool_calls = []
        for tc in tool_calls_data:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(name=name, arguments=args))
        return tool_calls

    def _parse_text_tool_calls(self, content: str) -> list[ToolCall]:
        """Parse text-based tool calls (fallback for non-tool models)"""
        tool_calls = []

        # Try <tool_call> format
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                tool_calls.append(ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                ))
            except json.JSONDecodeError:
                continue

        return tool_calls

    def parse_tool_calls(self, response: LLMResponse) -> list[ToolCall]:
        """Parse tool calls from response (already done in call())"""
        return response.tool_calls

    def format_tool_result(self, tool_name: str, result: str) -> dict:
        """Format tool result for LM Studio (OpenAI format)"""
        return {
            "role": "tool",
            "content": result,
            "name": tool_name,
        }

    async def list_models(self) -> list[str]:
        """List available models in LM Studio"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.config.url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m["id"] for m in data.get("data", [])]
        except:
            pass
        return []

    async def get_loaded_model(self) -> str | None:
        """Get currently loaded model in LM Studio"""
        models = await self.list_models()
        return models[0] if models else None
