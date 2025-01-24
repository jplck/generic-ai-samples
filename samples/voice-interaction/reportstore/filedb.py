import os
import logging
import json
from logging import INFO
from typing import Any
from typing import List, Optional, Union, TYPE_CHECKING
from backend.rtmt import RTMiddleTier, Tool, ToolResult, ToolResultDirection

class FileDBStore:
    logging.basicConfig(level=logging.INFO)

    categories = []

    def load_from_file(self, file_path: str):
        with open(file_path, "r") as file:
            return json.load(file)

    def init_data(self):
        self.logger.info("Creating container in database")
        templates_path = os.path.join(os.path.dirname(__file__), 'categories.json')
        self.categories = self.load_from_file(templates_path)
        # print(self.categories)

    def __init__(self):
        self.logger = logging.getLogger("filedb")
        self.logger.info("Initializing FileDBStore")
        self.init_data()  
    
    async def show_product_information(self, args: Any) -> ToolResult:
        print("showing information")
        information = {
            "title": args["title"],
            "text": args["text"],
            "image": args["image"]
        }
        # Return the result to the client
        return ToolResult(information, ToolResultDirection.TO_CLIENT)
    
    async def show_product_categories(self, args: Any) -> ToolResult:
        print("showing product categories")

        product_categories = []

        for item in self.categories:
            option = {}
            option["title"] = item["title"]
            option["text"] = item["text"]
            option["image"] = item["image"]
            product_categories.append(option)

        # Return the result to the client
        return ToolResult(product_categories, ToolResultDirection.TO_CLIENT)
    
    async def show_product_models(self, args: Any) -> ToolResult:
        print("showing product models for ", args)

        product_models = []

        for item in self.categories:
            if ("variations" in item):
                for varation in item["variations"]:
                    if ("products" in varation):
                        # print(varation)
                        for product in varation["products"]:
                            print(product)
                            option = {}
                            option["title"] = product["title"]
                            option["text"] = product["text"]
                            option["image"] = product["image"]
                            product_models.append(option)  
        
        # Return the result to the client
        return ToolResult(product_models, ToolResultDirection.TO_CLIENT)
    
    async def get_available_categories(self, args: Any) -> ToolResult:
        print("retreiving available categories", args)

        responses = []
        try:
            for item in self.categories:
                option = {}
                option["category_description"] = item["description"]
                option["image"] = item["image"]
                option["text"] = item["text"]
                option["category_name"] = item["category"]
                option["question"] = item["question"]
                responses.append(option)
        except Exception as e:
            print(e)
            return ToolResult("Error", ToolResultDirection.TO_SERVER)
        
        return ToolResult(responses, ToolResultDirection.TO_SERVER)

    async def get_products_by_category(self, args: Any) -> ToolResult:
        category = args["category"].lower()
        print("retreiving category: ", category)

        responses = []

        for item in self.categories:
            if (item["category"].lower() != category.lower().strip()):
                continue
            option = {}
            option["title"] = item["title"]
            option["description"] = item["description"]
            option["image"] = item["image"]
            option["text"] = item["text"]
            option["category"] = item["category"]
            option["question"] = item["question"]
            responses.append(option)

        return ToolResult(responses, ToolResultDirection.TO_SERVER)
            
    async def get_products(self, args: Any) -> ToolResult:
        keywords = args["keywords"].lower().strip()
        print("retreiving keywords: ", keywords)

        responses = []

        for item in self.categories:
            for varation in item["variations"]:
                if (varation["title"].lower() != keywords):
                    continue
                option = {}
                option["name"] = varation["name"]
                option["description"] = varation["description"]
                option["image"] = varation["image"]
                option["text"] = varation["text"]
                option["category"] = item["category"]
                responses.append(option)

        return ToolResult(responses, ToolResultDirection.TO_SERVER)
