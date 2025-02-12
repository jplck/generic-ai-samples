from dataclasses import dataclass

@dataclass
class Document:
    id: str
    page_content: str
    metadata: dict

@dataclass
class User:
    id: str
    email: str
    name: str