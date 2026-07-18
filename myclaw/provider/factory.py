"""Model Provider adapter construction."""

from myclaw.config.config import ProviderConfiguration
from myclaw.provider.anthropic import AnthropicProvider
from myclaw.provider.openai_compatible import OpenAICompatibleProvider
from myclaw.provider.ports import ModelProvider


def create_provider(configuration: ProviderConfiguration) -> ModelProvider:
    """Construct the adapter selected by one validated Provider configuration."""
    if configuration.protocol == "anthropic":
        return AnthropicProvider(configuration)
    if configuration.protocol == "openai-compatible":
        return OpenAICompatibleProvider(configuration)
    raise ValueError(f"Unsupported Provider protocol: {configuration.protocol}")
