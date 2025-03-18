
import os
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast
from dataclasses import dataclass
import json
import dotenv
from pathlib import Path
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace, trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from token_counter import TokenCounterCallback
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from llm import prepare_azure_openai_completion_model, prepare_azure_openai_embeddings_model
import uuid
from agent import Agent

dotenv.load_dotenv()

def setup_tracing():
    exporter = AzureMonitorTraceExporter.from_connection_string(
        os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
    )
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)
    span_processor = BatchSpanProcessor(exporter, schedule_delay_millis=60000)
    trace.get_tracer_provider().add_span_processor(span_processor)
    LangchainInstrumentor().instrument()
    return tracer

tracer = setup_tracing()

callback = TokenCounterCallback()
llm = prepare_azure_openai_completion_model([callback])

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a new graph
workflow = StateGraph(State)

agents = Agent(workflow)

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
    A tool that sends out product order orders and returns the shipping details.
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
    name="product_search_agent",
    tools=[product_search_tool],
    next_agents=["human_input_agent", "order_agent", "__end__"],
)

agents.create_agent(
    prompt="""
        Your are an expert that prepares an order based on an input context. 
        Do not call your order_tool before you have all the information in your context.
        
        {context}

        Call the human_input_agent agent to gather user input.
        - User shipping address and user name
        - User payment method
        - User email address or phone number for SMS for shipping updates

        If you have all the information in your context, call the order_tool to prepare the order.
        If the order_tool returns a confirmation number, return it to the user.
    """,
    llm=llm,
    name="order_agent",
    tools=[order_tool],
    next_agents=["human_input_agent", "product_search_agent", "__end__"],
)

#-----------------------------------------------------------------------------------------------

workflow.add_edge(START, "product_search_agent")
graph = agents.compile_graph()
graph.name = "Prodcut Search Graph"

