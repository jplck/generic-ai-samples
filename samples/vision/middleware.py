import json
import requests

from typing import Any, Callable, Optional
from aiohttp import web
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from imagelibrary import VectorDatabase

class Middleware:
    endpoint: str
    deployment: str
    key: Optional[str] = None
    cv_endpoint: Optional[str] = None
    cv_key: Optional[str] = None    

    # Server-enforced configuration, if set, these will override the client's configuration
    # Typically at least the model name and system message will be set by the server
    model: Optional[str] = None
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    disable_audio: Optional[bool] = None

    _tools_pending = {}
    _token_provider = None

    def __init__(self, endpoint: str, deployment: str, cv_endpoint: str, cv_key: str, credentials: AzureKeyCredential | DefaultAzureCredential):
        self.endpoint = endpoint
        self.deployment = deployment
        self.cv_endpoint = cv_endpoint
        self.cv_key = cv_key

        if isinstance(credentials, AzureKeyCredential):
            self.key = credentials.key
        else:
            self._token_provider = get_bearer_token_provider(credentials, "https://cognitiveservices.azure.com/.default")
            self._token_provider() # Warm up during startup so we have a token cached when the first request arrives


    def image_embedding_with_url(self, imageurl):
        """
        Embedding image using Azure Computer Vision 4
        """

        version = "?api-version=2024-02-01&model-version=2023-04-15"

        vec_img_url = (
            self.cv_endpoint + "/computervision/retrieval:vectorizeImage" + version
        )  # For doing the image vectorization

        headers = {
            "Content-type": "application/json",
            "Ocp-Apim-Subscription-Key": self.cv_key,
        }

        image = {"url": imageurl}
        r = requests.post(vec_img_url, image, headers=headers)
        image_emb = r.json()["vector"]

        return image_emb
    
    def image_embedding_with_file(self, imagepath):
        """
        Embedding image using Azure Computer Vision 4
        """

        version = "?overload=stream&api-version=2024-02-01&model-version=2023-04-15"

        vec_img_url = (
            self.cv_endpoint + "/computervision/retrieval:vectorizeImage" + version
        )  # For doing the image vectorization

        headers = {
            "Content-type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": self.cv_key,
        }

        with open(imagepath, "rb") as img:
            data = img.read()
        r = requests.post(vec_img_url, data=data, headers=headers)
        if r.status_code == 200:
            image_vector = r.json()["vector"]
            return image_vector
        else:
            return f"An error occurred while processing image Error code: {r.status_code}."

    async def _create_embedding_handler(self, request):
        print("_create_embedding_handler handler")
        reader = await request.multipart()
        field = await reader.next()
        if field.name != 'file':
            return web.Response(status=400, text="File field missing")
        
        filename = field.filename
        size = 0
        with open(f'./images/{filename}', 'wb') as f:
            while True:
                chunk = await field.read_chunk()  # 8192 bytes by default.
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)
        
        print(f"Received file {filename} with size {size} bytes")

        file_path = "./images/" + filename
        image_emb = self.image_embedding_with_file(file_path)
        # print("Image embedding:", image_emb)
        print(len(image_emb))
        return web.Response(status=200, text=f"{image_emb}")

    async def _search_images(self, request):
        print("_search_images handler")
        details = await request.json()
        # print(details)
        vector = json.loads(details["vector"])
        # print("Search vector:", vector)
        print(len(vector))
        database = VectorDatabase()
        results = database.search(vector)
        print("Search results:", results)
        return web.json_response(results)

    def attach_embedding_to_app(self, app, path):
        app.router.add_post(path, self._create_embedding_handler)

    def attach_search_to_app(self, app, path):
        app.router.add_post(path, self._search_images)
