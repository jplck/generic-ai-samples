
import os
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast
from dataclasses import dataclass
import json
import dotenv
from pathlib import Path
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
from langgraph.types import Command, interrupt
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
    A tool that searches for prodcuts in a prodcut database and returns the results.
    """
    with open(Path(__file__).parent / "assets/products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    return products

#-----------------------------------------------------------------------------------------------

def product_search_agent(state: State) -> Command[Literal["order_agent", "human_input_agent", "product_search_tool"]]:
    prompt = """
        Your are an expert providing information about available products(furniture). Use your tools to search for products.
        If you find a product, return the product name and description. Ask the user if he is satisfied with the result by calling the human_input_agent.
        If the user is not satisfied, ask for more details. If the user is satisfied, continue to the order_agent.
        Add the agent name you want to call to the end of your message. Use the form "call: <agent_name>".
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm.bind_tools([product_search_tool], tool_choice="auto")
    result = call.invoke({"input": state["messages"]})
    return generate_route(result)()

workflow.add_node("product_search_agent", product_search_agent)
workflow.add_node("product_search_tool", ToolNode([product_search_tool]))

#-----------------------------------------------------------------------------------------------

def order_agent(state: State) -> Command[Literal["human_input_agent", "__end__", "product_search_agent"]]:
    prompt = """
    Your are an expert that orders products based on an input context. Ask the user to provide the order details by calling human_input_agent.
    If the user or agent is satisfied with the order details, return the order details and call __end__.
    If the user or agent is not satisfied, ask for more details.
    if the user wants to revisit his search, call product_search_agent.
    Add the agent name you want to call to the end of your message. Use the form "call: <agent_name>".
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm
    result = call.invoke({"input": state["messages"]})
    return generate_route(result)()

workflow.add_node("order_agent", order_agent)

#-----------------------------------------------------------------------------------------------

def human_input_agent(
    state: State, config
) -> Command[Literal["order_agent", "product_search_agent"]]:
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

def generate_route(result: AIMessage) -> Route:
    update = {"messages": [result]}
    if not result.tool_calls:
        agent_name = result.content.split("call: ")[-1]
        #Return END if no agent name is provided
        goto = agent_name if agent_name else END
    else:
        tool_calls = result.additional_kwargs["tool_calls"]
        goto=[call["function"]["name"] for call in tool_calls]

    return Route(update, goto=goto)

workflow.add_node("human_input_agent", human_input_agent)

# Specify the edges between the nodes
workflow.add_edge(START, "product_search_agent")

workflow.add_edge("product_search_tool", "product_search_agent")
workflow.add_edge("order_agent", END)

graph = workflow.compile()
graph.name = "Prodcut Search Graph"

