
import os
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast
from dataclasses import dataclass
import json
import dotenv
from pathlib import Path
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from token_counter import TokenCounterCallback
from langchain_core.tools import tool
import uuid
from agent import AgentSystem
from llm import get_model_on_azure, get_github_model

dotenv.load_dotenv()

callback = TokenCounterCallback()

llm = get_model_on_azure(os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"), temperature=1.0, callbacks=[callback])
#llm = get_github_model()
agents = AgentSystem()

#-----------------------------------------------------------------------------------------------

@tool
def product_search_tool(query: str) -> str:
    """
    A tool that searches for furniture in a product database and returns the results.
    :return: A list of product names and descriptions.
    """
    with open(Path(__file__).parent / "assets/products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    return products

@tool
def order_tool(order_details: str) -> str:
    """
    A tool that sends out product order orders and returns the shipping number.
    Required order details are:
    - User shipping address and user name
    - User payment method
    - User email address or phone number for SMS for shipping updates
    - Product name and description + quantity
    :return: The order confirmation number.
    """
    return str(uuid.uuid4())

#-----------------------------------------------------------------------------------------------

agents.create_hil_agent(
    agent_name="human_input_agent",
    next_agents=["product_search_agent", "order_agent"],
)

agents.create_agent(
    prompt="""
        You are an expert that helps the user to find a product in a furniture store.
        You have access to a product search tool that can search for products in a database.
        If you found products, return a summary of a maximum of five products to the user.

        Call the human_input_agent to ask the user what product to pick. If the user picks a product, continue to the order_agent.

        Use the following context:
        {context}
    """,
    llm=llm,
    agent_name="product_search_agent",
    tools=[product_search_tool],
    next_agents=["human_input_agent", "order_agent", "__end__"],
)

agents.create_agent(
    prompt="""
        You are an expert that helps the user to order products he selected.

        You have access to the previous conversation context

        Context:
        {context}

        You process your workflow in the following way:
        - Summarize the products the user selected and format them as a markdown table. And ask the user
        if he wants to order the products. 
        Example:
        | Product Name | Description | Price |
        |--------------|-------------|-------|
        | Chair        | A comfy chair| $50   |
        | Table        | A wooden table| $100  |
        | Sofa         | A leather sofa| $500  |
        |--------------|-------------|-------|
        | Total        |             | $650  |
        |--------------|-------------|-------|
        - Call the human_input_agent and ask the user if he wants to order the products.
        - If the user wants to order the products, call the order_tool tool and pass the order details. If required, call the human_input_agent to ask the user for the order details.
        - If the user does not want to order the products, call the product_search_agent and ask the user what product to pick.
        - If the user does not want to pick a product, call the __end__ agent and end the workflow.

    """,
    llm=llm,
    agent_name="order_agent",
    tools=[order_tool],
    next_agents=["human_input_agent", "product_search_agent", "__end__"],
)

#-----------------------------------------------------------------------------------------------

graph = agents.compile_graph(initial_agent="product_search_agent")
graph.name = "Product Search Graph"

