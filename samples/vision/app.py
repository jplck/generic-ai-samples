import logging
import os
from pathlib import Path
from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from dotenv import load_dotenv
from middleware import Middleware
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visionbot")

load_dotenv()

async def create_app():
    llm_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    llm_deployment = os.environ.get("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
    client: AzureOpenAI = None
    if "AZURE_OPENAI_API_KEY" in os.environ:
        client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = "2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    else:
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        client = AzureOpenAI(
            azure_ad_token_provider = token_provider,
            api_version = os.getenv("AZURE_OPENAI_VERSION"),
            azure_endpoint = "2024-02-01",
        )
    app = web.Application()

    azure_cv_endpoint = os.getenv("AZURE_CV_ENDPOINT")
    azure_cv_key = os.getenv("AZURE_CV_KEY")

    middleware = Middleware(client, llm_endpoint, llm_deployment, azure_cv_endpoint, azure_cv_key)  
    middleware.attach_embedding_to_app(app, "/pictures")
    middleware.attach_search_to_app(app, "/search")
    middleware.attach_vision_to_app(app, "/vision")

    # Serve static files and index.html
    current_directory = Path(__file__).parent  # Points to 'app' directory
    static_directory = current_directory / 'static'

    # Ensure static directory exists
    if not static_directory.exists():
        raise FileNotFoundError("Static directory not found at expected path: {}".format(static_directory))

    # Serve index.html at root
    async def index(request):
        return web.FileResponse(static_directory / 'index.html')

    app.router.add_get('/', index)
    app.router.add_static('/static/', path=str(static_directory), name='static')

    return app

if __name__ == "__main__":
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 8765))
    web.run_app(create_app(), host=host, port=port)
