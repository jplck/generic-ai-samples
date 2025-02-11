from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from azure.ai.projects import AIProjectClient
import os
import asyncio
import dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from dataclasses import dataclass
import json

dotenv.load_dotenv()

token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

DOCUMENT_INDEX_NAME = "document-index"

@dataclass
class Document:
    id: str
    page_content: str
    metadata: dict

completion_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
    model=os.getenv("AZURE_OPENAI_COMPLETION_MODEL"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=token_provider,
    model_info={
        "json_output": False,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
    },
)

embeddings_model = AzureOpenAIEmbeddings(    
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    azure_ad_token_provider=token_provider,
)


def aquire_search_index(index_name: str) -> AzureSearch:
    return AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
        index_name=index_name,
        embedding_function=embeddings_model.embed_query,
    )

def index_documents(search_index_name: str, docs) -> None:
    search_index = aquire_search_index(search_index_name)
    search_index.add_texts(
        keys=[doc.id for doc in docs],
        texts=[doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs],
    )

def parse_templates(json_file_path: str) -> list[Document]:
    with open(json_file_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    docs = []
    for t in templates:
        docs.append(
            Document(
                id=str(t.get("id", "")),
                page_content=t.get("body", ""),
                metadata={
                    "topic": t.get("topic", ""),
                    "subject": t.get("subject", "")
                },
            )
        )
    return docs

def search_index(search_index_name: str, query: str, k: int = 5) -> list[Document]:
    search_index = aquire_search_index(search_index_name)
    return search_index.similarity_search(query, k=k, search_type="hybrid")

project_client = AIProjectClient.from_connection_string("westeurope.api.azureml.ms;f8543040-cba6-434e-a491-f1ca6f110652;rg-sample3;japollac-6805", DefaultAzureCredential())

def search_tool(query: str) -> list[Document]:
    """Tool that searches a vector database for fitting email templates."""
    search = search_index(DOCUMENT_INDEX_NAME, query)
    return search

async def main():

    docs = parse_templates("samples/chat/assets/templates.json")
    index_documents(DOCUMENT_INDEX_NAME, docs)

    search_agent = AssistantAgent(
        name = "search_agent",
        model_client = completion_model_client,
        tools = [search_tool],
        system_message = "I am a email template search agent. I can help you find email templates for various scenarios. Please provide me with a scenario and I will provide you with a template. Ask the user if he deam the result a good fit.",
        reflect_on_tool_use = True
    )

    compose_agent = AssistantAgent(
        name = "compose_agent",
        model_client = completion_model_client,
        system_message = "Take the selected template and compose an email with it. Ask for user input if required.",
        reflect_on_tool_use = True
    )

    user_proxy = UserProxyAgent("user_proxy", input_func=input)
    termination_condition = TextMentionTermination("approve")

    team = RoundRobinGroupChat([search_agent, compose_agent, user_proxy], termination_condition=termination_condition)

    await Console(team.run_stream(task="Find an email template for a product recall."))

asyncio.run(main())