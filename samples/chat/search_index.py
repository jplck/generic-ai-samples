from langchain_community.vectorstores.azuresearch import AzureSearch
import os
from langchain_openai import AzureOpenAIEmbeddings
from samples.chat.common import get_default_token_provider
from samples.chat.model import Document

embeddings_model = AzureOpenAIEmbeddings(    
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    azure_ad_token_provider=get_default_token_provider(),
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

def search_index(search_index_name: str, query: str, k: int = 5) -> list[Document]:
    search_index = aquire_search_index(search_index_name)
    return search_index.similarity_search(query, k=k, search_type="hybrid")