import os
from typing import List
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

def prepare_azure_openai_completion_model(callbacks: List[callable]) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_ad_token_provider= token_provider if not api_key else None,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key= api_key if api_key else None,
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
        streaming=True,
        model_kwargs={"stream_options":{"include_usage": True}},
        callbacks=callbacks,
        openai_api_type= "api_key" if api_key else "azure_ad",
    )

def prepare_azure_openai_embeddings_model() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_ad_token_provider = token_provider if not api_key else None,
        api_key=api_key if api_key else None
    )