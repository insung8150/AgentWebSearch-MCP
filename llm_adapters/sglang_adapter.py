"""
SGLang Adapter for AgentCPM-Explore

Handles the <tool_call>...</tool_call> text format used by AgentCPM-Explore model.
"""

import json
import re
import httpx

from .base import BaseLLMAdapter, AdapterConfig, LLMResponse, ToolCall


class SGLangAdapter(BaseLLMAdapter):
    """Adapter for SGLang server (AgentCPM-Explore model)

    Tool calls are embedded in text as:
    <tool_call>
    {"name": "search", "arguments": {"query": ["q1", "q2"]}}
    </tool_call>
    """

    DEFAULT_URL = "http://localhost:30001"
    DEFAULT_MODEL = "AgentCPM-Explore"

    def __init__(self, url: str | None = None, model: str | None = None, **kwargs):
        config = AdapterConfig(
            url=url or f"{self.DEFAULT_URL}/v1/chat/completions",
            model=model or self.DEFAULT_MODEL,
            **kwargs
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "sglang"

    @property
    def tool_format(self) -> str:
        return "text"

    async def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send messages to SGLang server

        Note: Tools are included in system prompt, not as separate parameter.
        """
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                self.config.url,
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                }
            )
            data = response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        llm_response = LLMResponse(
            content=content,
            tool_calls=[],
            raw_response=data,
            model=self.config.model,
            usage=usage,
        )

        # Parse tool calls from text
        llm_response.tool_calls = self.parse_tool_calls(llm_response)

        return llm_response

    def parse_tool_calls(self, response: LLMResponse) -> list[ToolCall]:
        """Parse <tool_call>...</tool_call> tags from response content"""
        tool_calls = []

        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, response.content, re.DOTALL)

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

    def format_tool_result(self, tool_name: str, result: str) -> dict:
        """Format tool result for SGLang (text format)"""
        return {
            "role": "user",
            "content": f"<tool_result name=\"{tool_name}\">\n{result}\n</tool_result>",
        }

    def get_tool_call_instruction(self) -> str:
        """Get instruction for tool call format"""
        return """Tool call format:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1", ...}}
</tool_call>"""
