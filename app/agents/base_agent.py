"""Base class for Pydantic AI agents."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel
from pydantic_ai import Agent

from app.config import settings
from app.services.model_selector.registry import get_agent_model

logger = logging.getLogger(__name__)

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

    # Model tier for environment-aware resolution (reasoning / standard / fast)
    model_tier: str = "standard"
    # Explicit model override at the class level (bypasses tier resolution)
    model: str | None = None
    temperature: float = 0.7
    max_retries: int = settings.llm_max_retries

    def __init__(self, model_override: str | None = None) -> None:
        """Initialize the agent.

        Model resolution priority:
        1. model_override parameter (explicit runtime override)
        2. model class attribute (if set by subclass)
        3. model selector snapshot for this agent (if enabled)
        4. settings.get_model(self.model_tier) (environment-aware tier fallback)
        """
        model_source = "tier_default"
        if model_override:
            self._model = model_override
            model_source = "runtime_override"
        elif self.model:
            self._model = self.model
            model_source = "class_override"
        else:
            selected_model: str | None = None
            if settings.model_selector_enabled:
                selected_model = get_agent_model(settings.environment, self.__class__.__name__)
            if selected_model:
                self._model = selected_model
                model_source = "model_selector"
            else:
                self._model = settings.get_model(self.model_tier)
        self._agent: Agent[None, OutputT] | None = None

        logger.info(
            "Agent initialized",
            extra={
                "agent": self.__class__.__name__,
                "model": self._model,
                "model_tier": self.model_tier,
                "model_source": model_source,
                "temperature": self.temperature,
            },
        )

    @property
    def agent(self) -> Agent[None, OutputT]:
        """Lazily initialize and return the Pydantic AI agent."""
        if self._agent is None:
            self._agent = cast(
                Agent[None, OutputT],
                Agent(
                    model=self._model,
                    output_type=self.output_type,
                    system_prompt=self.system_prompt,
                    retries=self.max_retries,
                ),
            )
            self._setup_tools()
        agent = self._agent
        assert agent is not None
        return agent

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
        agent_name = self.__class__.__name__
        logger.info(
            "Agent run started",
            extra={
                "agent": agent_name,
                "input_type": type(input_data).__name__,
                "model": self._model,
            },
        )

        prompt = self._build_prompt(input_data)
        logger.info(
            "Prompt built, sending to LLM",
            extra={
                "agent": agent_name,
                "prompt_length": len(prompt),
                "model": self._model,
            },
        )

        t0 = time.perf_counter()
        result = await self.agent.run(prompt)
        elapsed = time.perf_counter() - t0

        usage = result.usage()
        logger.info(
            "Agent run completed",
            extra={
                "agent": agent_name,
                "duration_s": round(elapsed, 2),
                "request_tokens": usage.request_tokens,
                "response_tokens": usage.response_tokens,
                "total_tokens": usage.total_tokens,
                "output_type": type(result.output).__name__,
            },
        )

        return result.output

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
    ) -> OutputT:
        """Run the agent with a list of messages (for multi-turn conversations).

        Args:
            messages: List of message dicts with 'role' and 'content'.

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

        result = await self.agent.run(prompt)

        return result.output
