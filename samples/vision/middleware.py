import json
import requests
import base64
from io import BytesIO
from mimetypes import guess_type
from typing import Any, Callable, Optional
from aiohttp import web
from imagelibrary import VectorDatabase
from openai import AzureOpenAI

class Middleware:
    endpoint: str
    deployment: str
    key: Optional[str] = None
    cv_endpoint: Optional[str] = None
    cv_key: Optional[str] = None    
    client: AzureOpenAI = None

    def __init__(self, client: AzureOpenAI, endpoint: str, deployment: str, cv_endpoint: str, cv_key: str):
        self.endpoint = endpoint
        self.deployment = deployment
        self.cv_endpoint = cv_endpoint
        self.cv_key = cv_key
        self.client = client

        print("Middleware initialized")
        print(f"Endpoint: {self.endpoint}")
        print(f"Deployment: {self.deployment}")
        print(f"CV Endpoint: {self.cv_endpoint}")

    # Function to encode a local image into data URL 
    def local_image_to_data_url(self, image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

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
        if (not vector):
            return web.Response(status=400, text="Vector field missing")
        # print("Search vector:", vector)
        print(len(vector))
        database = VectorDatabase()
        results = database.search(vector)
        print("Search results:", results)
        return web.json_response(results)
    
    async def _look_at_pictures(self, request):
        print("_look_at_pictures handler")
        details = await request.json()
        picture1 = details["picture1"]
        picture2 = details["picture2"]

        picture1_url = self.local_image_to_data_url(picture1)
        picture2_url = self.local_image_to_data_url(picture2)

        prompt = f"Look at these two pictures. Image 1 and Image 2. Are they similar List all the differences according to category, color, position and size."

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": prompt 
                    },
                    { 
                        "type": "image_url",
                        "image_url": {
                            "url": picture1_url
                        }
                    },
                    { 
                        "type": "image_url",
                        "image_url": {
                            "url": picture2_url
                        }
                    }
                  ] 
                } 
            ],
            max_tokens=2000 
        )
        comparison = response.choices[0].message.content
        print(comparison)
        return web.json_response(comparison)

    def attach_embedding_to_app(self, app, path):
        app.router.add_post(path, self._create_embedding_handler)

    def attach_search_to_app(self, app, path):
        app.router.add_post(path, self._search_images)

    def attach_vision_to_app(self, app, path):
        app.router.add_post(path, self._look_at_pictures)
