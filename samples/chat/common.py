from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from typing import Callable

def get_default_token_provider() -> Callable[[], str]:
    return get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")