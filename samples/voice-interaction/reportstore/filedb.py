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
        print(args)
        information = {
            "name": args["name"],
            "text": args["text"],
            "image": args["image"]
        }
        # Return the result to the client
        return ToolResult(information, ToolResultDirection.TO_CLIENT)
            
    async def get_products(self, args: Any) -> ToolResult:
        print("getting products")
        print(args)
        responses = []

        for item in self.categories:
            option = {}
            option["name"] = item["name"]
            option["description"] = item["description"]
            option["image"] = item["image"]
            option["text"] = item["text"]
            option["category"] = item["category"]
            responses.append(option)

        print(responses)

        return ToolResult(responses, ToolResultDirection.TO_SERVER)
