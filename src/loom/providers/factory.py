from loom.config import Config, ProviderConfig
from loom.contracts import LoomConfigurationError
from loom.providers.base import ModelAdapter


def create_adapter(profile: ProviderConfig, config: Config) -> ModelAdapter:
    if profile.kind == "ollama":
        from loom.providers.ollama import OllamaAdapter

        return OllamaAdapter(profile, config)
    from loom.providers.anthropic import AnthropicAdapter
    from loom.providers.azure_openai import AzureOpenAIAdapter
    from loom.providers.gemini import GeminiAdapter
    from loom.providers.openai import OpenAIAdapter

    adapters = {
        "openai": OpenAIAdapter,
        "azure_openai": AzureOpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "gemini": GeminiAdapter,
    }
    adapter = adapters.get(profile.kind)
    if adapter is None:
        raise LoomConfigurationError("invalid_provider_kind")
    return adapter(profile, config)
