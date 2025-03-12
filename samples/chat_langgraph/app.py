
import os
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast, Sequence
from dataclasses import dataclass
import json
import dotenv
from pathlib import Path
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
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
from agents import create_agent, State, Goto

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

# Define a new graph
workflow = StateGraph(State)

@dataclass
class Route:
    data: Dict[str, Any]
    goto: str
    def __call__(self) -> Command:
        return Command(update=self.data, goto=self.goto)

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
def human_input_tool():
    """
    A tool that waits for user input.
    """
    user_input = interrupt(value="Ready for user input.")
    return Command(
        update={"messages": [user_input]},
    )

@tool 
def route_to_order_agent():
    """
    A tool that routes the conversation to the order agent.
    """
    return Command(
        goto="order_agent",
        update={"messages": ["Redirecting to order agent."]},
    )

create_agent(
    graph=workflow,
    llm=llm,
    agent_name="product_search_agent",
    prompt=""""
        You are a product search agent that searches for furniture, based on user input.
        Use your tools to search for products in the database.
        If you don't find any product, return 'No products found'.
        Ask the user if he is interested in ordering the product. If yes, redirect the conversation to the order agent.
        """,
    tools=[product_search_tool, route_to_order_agent, human_input_tool],
    reiterate_after_run=True,
    graph_destinations=["order_agent"],
)

create_agent(
    graph=workflow,
    llm=llm,
    agent_name="order_agent",
    prompt=""""
        You are an order agent that takes the user input and orders the product.
        Use your tools to prepare the order by asking the user for the shipping address, payment method, and email address.
        """,
    tools=[human_input_tool],
    reiterate_after_run=True,
    graph_destinations=["product_search_agent"],
)

workflow.add_edge(START, "product_search_agent")
workflow.add_edge("product_search_agent", END)

graph = workflow.compile()
graph.name = "Prodcut Search Graph"

