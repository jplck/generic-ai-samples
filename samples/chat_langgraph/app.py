import ast
import os
import sys
import random
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast
from uuid import UUID
import dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from IPython.display import Image
from langchain.agents.agent import AgentAction
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.tools import tool, BaseTool
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
import hashlib
from langchain_core.documents import Document
from langchain_community.vectorstores.azuresearch import AzureSearch
from token_counter import TokenCounterCallback

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

llm: AzureChatOpenAI = None
if "AZURE_OPENAI_API_KEY" in os.environ:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
        streaming=True,
        model_kwargs={"stream_options":{"include_usage": True}},
        callbacks=[callback]
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

else:
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    llm = AzureChatOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
        openai_api_type="azure_ad",
        streaming=True,
        model_kwargs={"stream_options":{"include_usage": True}},
        callbacks=[callback]
    )
    embeddings_model = AzureOpenAIEmbeddings(    
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
        model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
        azure_ad_token_provider = token_provider
    )

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a new graph
workflow = StateGraph(State)

#-----------------------------------------------------------------------------------------------

@tool
def weather_tool(location: str) -> str:
    """
    A tool that provides the weather for a given location
    """
    return "Warm, 25 degrees"

@tool
def tool2(query: str) -> str:
    """
    Generic sample tool 2
    """
    return "Return of tool 2"

#-----------------------------------------------------------------------------------------------

def agent1(state: State) -> dict[str, list[AIMessage]]:
    prompt = """
        Your are an expert returning the weather forecast for a given location using your provided tools. Return the weather forecast for the location provided.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm.bind_tools([weather_tool], tool_choice="auto")
    return {"messages": [call.invoke({"input": state["messages"]})]}


workflow.add_node("weather_agent", agent1)
workflow.add_node("weather_tool", ToolNode([weather_tool]))

#-----------------------------------------------------------------------------------------------

def agent2(state: State) -> dict[str, list[AIMessage]]:
    prompt = """Your are an expert giving closing an preparation recommendation for a given location using the context provided."""

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm
    return {"messages": [call.invoke({"input": [state["messages"][-1]]})]}

workflow.add_node("recommendation_agent", agent2)

#-----------------------------------------------------------------------------------------------

def weather_router(state: State) -> Literal["weather_tool", "recommendation_agent"]:
    last_message = state["messages"][-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "recommendation_agent"
    # Otherwise we execute the requested actions
    return "weather_tool"

# Specify the edges between the nodes
workflow.add_edge(START, "weather_agent")

workflow.add_conditional_edges(
    "weather_agent",
    weather_router
)

workflow.add_edge("weather_tool", "weather_agent")
workflow.add_edge("recommendation_agent", END)

graph = workflow.compile()
graph.name = "Weather Agent Graph"

