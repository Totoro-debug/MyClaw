"""Official SDK-backed Model Provider adapters."""

from myclaw.config import ProviderConfiguration
from myclaw.contracts import ModelProvider
from myclaw.providers.anthropic import AnthropicProvider
from myclaw.providers.openai_compatible import OpenAICompatibleProvider

__all__ = ["AnthropicProvider", "OpenAICompatibleProvider", "create_provider"]


def create_provider(configuration: ProviderConfiguration) -> ModelProvider:
    """Construct the adapter selected by one validated Provider configuration."""
    if configuration.protocol == "anthropic":
        return AnthropicProvider(configuration)
    if configuration.protocol == "openai-compatible":
        return OpenAICompatibleProvider(configuration)
    raise ValueError(f"Unsupported Provider protocol: {configuration.protocol}")
