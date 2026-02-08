"""Base class for Pydantic AI agents."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

from app.config import settings

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract base class for Pydantic AI agents.

    Each agent should:
    1. Define the system_prompt property
    2. Define the output_type property
    3. Implement _build_prompt to construct the user prompt
    4. Optionally override _setup_tools to add agent-specific tools
    """

    # Default model from settings, can be overridden per agent
    model: str = settings.default_llm_model
    temperature: float = 0.7
    max_retries: int = settings.llm_max_retries

    def __init__(self, model_override: str | None = None) -> None:
        """Initialize the agent.

        Args:
            model_override: Optional model to use instead of the default.
        """
        self._model = model_override or self.model
        self._agent: Agent[Any, OutputT] | None = None

    @property
    def agent(self) -> Agent[Any, OutputT]:
        """Lazily initialize and return the Pydantic AI agent."""
        if self._agent is None:
            self._agent = Agent(
                model=self._model,
                result_type=self.output_type,
                system_prompt=self.system_prompt,
                retries=self.max_retries,
            )
            self._setup_tools()
        return self._agent

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for the agent."""
        pass

    @property
    @abstractmethod
    def output_type(self) -> type[OutputT]:
        """Pydantic model type for structured output."""
        pass

    def _setup_tools(self) -> None:
        """Override to add tools to the agent.

        Example:
            @self.agent.tool
            async def my_tool(ctx: RunContext, arg: str) -> str:
                return f"Processed: {arg}"
        """
        pass

    async def run(
        self,
        input_data: InputT,
        context: dict[str, Any] | None = None,
    ) -> OutputT:
        """Run the agent with input data.

        Args:
            input_data: Pydantic model with input parameters.
            context: Optional context dict passed as deps to the agent.

        Returns:
            Structured output as defined by output_type.
        """
        prompt = self._build_prompt(input_data)

        result = await self.agent.run(
            prompt,
            deps=context or {},
        )

        return result.data

    @abstractmethod
    def _build_prompt(self, input_data: InputT) -> str:
        """Build the user prompt from input data.

        Args:
            input_data: Input data for the agent.

        Returns:
            User prompt string.
        """
        pass

    async def run_with_messages(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> OutputT:
        """Run the agent with a list of messages (for multi-turn conversations).

        Args:
            messages: List of message dicts with 'role' and 'content'.
            context: Optional context dict passed as deps to the agent.

        Returns:
            Structured output as defined by output_type.
        """
        # Convert messages to prompt format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(prompt_parts)

        result = await self.agent.run(
            prompt,
            deps=context or {},
        )

        return result.data
