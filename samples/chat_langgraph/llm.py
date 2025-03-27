import os
from typing import List
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.chat_models import init_chat_model

def get_model_on_azure(deployment_name: str, temperature: float = 0.0, callbacks: List = None):
    
    #If you stick to the default environment variables, you can remove the model_kwargs.
    
    model_kwargs={
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "callbacks": callbacks,
    }

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        model_kwargs["api_key"] = api_key
    else:
        model_kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

    return init_chat_model(deployment_name, model_provider="azure_openai", temperature=temperature, model_kwargs=model_kwargs)

def get_github_model():
    model_kwargs = {
        "api_key": os.getenv("GITHUB_MODELS_TOKEN"),
        "base_url": "https://models.inference.ai.azure.com",
    }
    return init_chat_model(model="gpt-4o", model_provider="openai", temperature=0.0, model_kwargs=model_kwargs)