
import os
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
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

class RouteTemplate:
    description: str
    goto: str

@tool
def goto_human_input_agent():
    """
    Use this tool as soon as you want to call the human_input_agent.
    """
    return "call: human_input_agent"

@dataclass
class TransferDefinition:
    description: str
    goto: str
    tool_name: str = None
    def __call__(self, *args, **kwds):
        def tool_call (tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
            return Command(
                update={"messages": [{"role": "tool", "content": "transfer", "tool_call_id": tool_call_id}]},
                goto=self.goto,
            )

        return tool(
            name_or_callable=self.tool_name if self.tool_name else f"{self.goto}-transfer-tool",
            runnable=tool_call,
            description=self.description,
        )

def agent(graph:StateGraph, transfer_definitions: List[TransferDefinition], agent_name: str = None, tools: List[callable] = None, tool_node_name: str = None):
    def wrapper(func):
        transfer_node_name_length = 10
        print(f"Registering agent {func.__name__} with transfer definitions: {transfer_definitions}")
        transfer_node_name = ''.join(random.choices(string.ascii_letters + string.digits, k=transfer_node_name_length))
        agent_node_name = func.__name__ if agent_name is None else agent_name
        graph.add_node(transfer_node_name, ToolNode([transfer.__call__() for transfer in transfer_definitions]))
        graph.add_node(agent_node_name, func, destinations=tuple([transfer.goto for transfer in transfer_definitions] + [transfer_node_name]))
        graph.add_edge(transfer_node_name, agent_node_name)
        # tools
        if tools is not None:
            t_name = tool_node_name if tool_node_name is not None else f"{agent_node_name}-tools"
            graph.add_node(t_name, ToolNode(tools))
            graph.add_edge(t_name, agent_node_name)

        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner
    return wrapper
    
@dataclass
class RouteMapping():
    tool_name: str
    via: str #Name of tool node that bundles the tools

#-----------------------------------------------------------------------------------------------
tools = [product_search_tool]

@agent(workflow, [
    TransferDefinition(
        description="Use this tool as soon as you want to call the human_input_agent.",
        goto="human_input_agent",
    )
], tools=tools)
def product_search_agent(state: State):
    prompt = """
        Your are an expert providing information about available furniture to purchase.

        Search for a product based on the user input, using your tool.
        - If you find a product, return the product name and description. And ask the human if he is satisfied with the result.
        If you don't find a product, ask the user if he wants to change his search.
        If the user wants to change his search, return to step 1.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm.bind_tools(tools, tool_choice="auto")
    result = call.invoke({"input": state["messages"]})
    return generate_route(result, "human_input_agent",[
        RouteMapping("product_search_tool", "product_search_tool"),
        RouteMapping("goto_order_agent", "product_search_tool"),
        RouteMapping("goto_human_input_agent", "product_search_tool")
    ])()

#-----------------------------------------------------------------------------------------------

def order_agent(state: State) -> Command[Literal["human_input_agent", "__end__", "product_search_agent", "order_tool"]]:
    prompt = """
    Your are an expert that prepares an order based on an input context. 
    Do not call your order_tool before you have all the information in your context.
    
    Call the human_input_agent agent to gather user input.
    - User shipping address and user name
    - User payment method
    - User email address or phone number for SMS for shipping updates

    1. Ask the user if he wants to proceed with the order.
    2. If the user wants to proceed, call the order_tool to place the order. If the order is successful, return the order confirmation number and end the conversation by calling the __end__ agent.
    3. If the user wants to revisit his search, call product_search_agent.
    4. If the user wants to cancel, return END.
    5. If the user wants to change his order, return to step 1.

    Add the agent name you want to call to the end of your message. Use the form "call: <agent_name>".
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm.bind_tools([order_tool], tool_choice="auto")
    result = call.invoke({"input": state["messages"]})
    return generate_route(result, "human_input_agent")()

#workflow.add_node("order_agent", order_agent)
#workflow.add_node("order_tool", ToolNode([order_tool]))

#-----------------------------------------------------------------------------------------------

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

def generate_route(result: AIMessage, default_node: str, mappings: List[RouteMapping]) -> Route:
    update = {"messages": [result]}
    if not result.tool_calls:
        goto = default_node
    else:
        gotos = []
        for tool_call in result.tool_calls:
            for mapping in mappings:
                if mapping.tool_name == tool_call["name"]:
                    gotos.append(mapping.via)
                    
        goto=gotos

    return Route(update, goto=goto)

workflow.add_node("human_input_agent", human_input_agent)

workflow.add_edge(START, "product_search_agent")

#workflow.add_edge("product_search_tool", "product_search_agent")
#workflow.add_edge("order_tool", "order_agent")
#workflow.add_edge("order_agent", END)

graph = workflow.compile()
graph.name = "Prodcut Search Graph"

