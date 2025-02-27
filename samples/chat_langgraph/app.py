
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
import uuid

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
    A tool that sends out orders and returns the shipping details.
    :return: The order confirmation number.
    """
    return str(uuid.uuid4())

#-----------------------------------------------------------------------------------------------

def product_search_agent(state: State) -> Command[Literal["order_agent", "human_input_agent", "product_search_tool"]]:
    prompt = """
        Your are an expert providing information about available furniture to purchase.

        ONLY call your tool if you have no products in your context. Otherwise use your context.
        
        Search for a product based on the user input, using your tool.

        If you find products, return the product names and descriptions.

        Ask the user if he is satisfied with the result by calling the human_input_agent

        If the user is not satisfied, ask for more details and call the products_search_tool again.

        If the user is satisfied, continue to the order_agent.

        If you don't find a product, ask the user if he wants to change his search.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm.bind_tools([product_search_tool], tool_choice="auto")
    result = call.invoke({"input": state["messages"]})
    return generate_route(result, prompt)()

workflow.add_node("product_search_agent", product_search_agent)
workflow.add_node("product_search_tool", ToolNode([product_search_tool]))

#-----------------------------------------------------------------------------------------------

def order_agent(state: State) -> Command[Literal["human_input_agent", "__end__", "product_search_agent", "order_tool"]]:
    prompt = """
    Your are an expert that helps the user to place an order for a product.

    Gather user information by calling the human_input_agent if not available in your context.
    You need the following information:
    - User shipping address and user name
    - User payment method
    - User email address or phone number for SMS for shipping updates
    - Quantity of the product

    If you have all the info, ask the user if he wants to proceed with the order.
    If the user wants to proceed, call the order_tool to place the order. If the order is successful, return the order confirmation number and end the conversation by calling the __end__ agent.
    
    If the user wants to revisit his search, call product_search_agent.
    
    If the user wants to cancel, return END.
    
    If the user wants to change his order, return to step 1.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm.bind_tools([order_tool], tool_choice="auto")
    result = call.invoke({"input": state["messages"]})
    return generate_route(result, prompt)()

workflow.add_node("order_agent", order_agent)
workflow.add_node("order_tool", ToolNode([order_tool]))

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

def plan(message: str, agent_prompt: str) -> str:
    prompt = """
    You are an expert, that takes an input message and an agent prompt and returns the next agent to call.
    You are given the following input message:
    {input}
    You are given the following agent prompt:
    {agent_prompt}
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm
    result = call.invoke({"input": [message], "agent_prompt": agent_prompt})
    return result.content

def generate_route(result: AIMessage, agent_prompt: str) -> Route:
    update = {"messages": [result]}
    goto = [END]
    if result.tool_calls:
        tool_calls = result.additional_kwargs["tool_calls"]
        goto=[call["function"]["name"] for call in tool_calls]
    elif type(result) == AIMessage:
        goto = [plan(result.content, agent_prompt)]

    return Route(update, goto=goto)

workflow.add_node("human_input_agent", human_input_agent)

workflow.add_edge(START, "product_search_agent")

workflow.add_edge("product_search_tool", "product_search_agent")
workflow.add_edge("order_tool", "order_agent")
workflow.add_edge("order_agent", END)

graph = workflow.compile()
graph.name = "Prodcut Search Graph"

