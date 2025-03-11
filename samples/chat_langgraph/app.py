
import os
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast, Sequence
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from dataclasses import dataclass
import json
import dotenv
from pathlib import Path
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from langchain_core.messages import AIMessage, MessageLikeRepresentation, ToolMessage
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
from langgraph.types import Command, interrupt
from llm import prepare_azure_openai_completion_model, prepare_azure_openai_embeddings_model
import uuid
import random
import string

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

def create_agent(graph: StateGraph, llm, agent_name: str, prompt: str, inputs: Sequence[MessageLikeRepresentation] = (), tools: List[callable] = None):
    tool_node_name = f"{agent_name}_tools"
    
    def agent_runner(state: State):
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt), ("placeholder", "{input}"), *inputs]
        )
        call = prompt_template | llm.bind_tools(tools, tool_choice="auto")
        results = call.invoke({"input": state["messages"]})
        if len(results.tool_calls) > 0:
            return Command(goto=tool_node_name, update={"messages": [results]})
        
        return {"messages": [results]}
    
    graph.add_node(agent_name, agent_runner, destinations=tuple([tool_node_name]) if tools else None)

    if tools:
        def tool_runner(state: State):
            results = []
            message:AIMessage = state["messages"][-1]
            tool_calls = message.tool_calls
            if not tool_calls:
                return results
            tools_by_name = {tool.name: tool for tool in tools}
            for tool_call in tool_calls:
                tool = tools_by_name[tool_call["name"]]
                tool_response = tool.invoke(tool_call)
                if isinstance(tool_response, Command):
                    results.append(tool_response)
                elif isinstance(tool_response, ToolMessage):
                    results.append(Command(update={"messages": [tool_response]}))
            return results
        
        graph.add_node(tool_node_name, tool_runner)
        graph.add_edge(tool_node_name, agent_name)

def human_input_agent(
    state: State, config
) -> Command[Literal["product_search_agent"]]:
    """A node for collecting user input."""

    user_input = interrupt(value="Ready for user input.")

    # identify the last active agent
    # (the last active node before returning to human)
    langgraph_triggers = config["metadata"]["langgraph_triggers"]
    if len(langgraph_triggers) != 1:
        raise AssertionError("Expected exactly 1 trigger in human node")

    active_agent = langgraph_triggers[0].split(":")[1]

    return Command(
        update={
            "messages": [
                {
                    "role": "human",
                    "content": user_input,
                }
            ]
        },
        goto=active_agent,
    )  

create_agent(
    graph=workflow,
    llm=llm,
    agent_name="product_search_agent",
    prompt=""""
        You are a product search agent that takes user input to search for furniture.
        if you find products, return a list of product names and descriptions to the user.
        If you don't find any product, return 'No products found'.
        """,
    tools=[product_search_tool],
)

create_agent(
    graph=workflow,
    llm=llm,
    agent_name="human_input_agent",
    prompt=""""
        You are a human input agent that is called whenever the user is required to provide input.
        """,
)

workflow.add_edge(START, "product_search_agent")

graph = workflow.compile()
graph.name = "Prodcut Search Graph"

