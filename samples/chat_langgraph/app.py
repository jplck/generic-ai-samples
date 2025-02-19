
import os
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast
import dotenv
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
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
from llm import prepare_azure_openai_completion_model, prepare_azure_openai_embeddings_model

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

