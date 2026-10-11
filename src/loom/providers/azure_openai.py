from loom.providers.openai import OpenAIAdapter


class AzureOpenAIAdapter(OpenAIAdapter):
    provider_kind = "azure_openai"
