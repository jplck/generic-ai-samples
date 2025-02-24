import json
from enum import Enum
from typing import Any, Callable
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

class ToolResultDirection(Enum):
    TO_SERVER = 1
    TO_CLIENT = 2

class ToolResult:
    text: str
    destination: ToolResultDirection

    def __init__(self, text: str, destination: ToolResultDirection):
        self.text = text
        self.destination = destination

    def to_text(self) -> str:
        if self.text is None:
            return ""
        return self.text if type(self.text) == str else json.dumps(self.text)

class Tool:
    target: Callable[..., ToolResult]
    schema: Any

    def __init__(self, target: Any, schema: Any):
        self.target = target
        self.schema = schema

class RTToolCall:
    tool_call_id: str
    previous_id: str

    def __init__(self, tool_call_id: str, previous_id: str):
        self.tool_call_id = tool_call_id
        self.previous_id = previous_id

_get_products_tool_schema = {
    "type": "function",
    "name": "get_product_data",
    "description": "Search the product database for available product variations that the user might be interestd in. The knowledge base is in English, translate to and from English if " + \
                   "needed. Results are returned in JSON format with a set of questions that need to be answered by the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "keywords that the user has provided to search for product variations."
            }
        },
        "required": ["keywords"],
        "additionalProperties": False
    }
}

_show_product_information_tool_schema = {
    "type": "function",
    "name": "show_product_information",
    "description": "Shows the user a piece of information to support the conversation. The information should be a name, supporting text and reference to an image a that can be displayed to the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The title of the information."
            },
            "text": {
                "type": "string",
                "description": "Additional context that should be displayed to the user."
            },
            "image": {
                "type": "string",
                "description": "The url of the image that should be displayed to the user."
            }
        },
        "required": ["name", "text", "image"],
        "additionalProperties": False
    }
}
