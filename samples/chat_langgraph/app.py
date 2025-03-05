
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
import math

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
embedding_model = prepare_azure_openai_embeddings_model()

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

class AgentDefinition(TypedDict):
    name: str
    func: Any
    description: str
    vector: List[float]
    score: float
    targets: List[str]

class agent_library():
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.agents: List[AgentDefinition] = []

    def add_agent(self, name, targets=[]):
        def decorator(func):	
            description = func.__doc__
            if description is None:
                raise ValueError("Agent function must have a docstring.")
            agent_def = AgentDefinition(name=name, func=func, description=description, vector=self._embed(description), targets=targets)
            self.agents.append(agent_def)
            return func
        return decorator

    def get_agent(self, name):
        return self.agents[name]
    
    def search_agents(self, query, k=5, score_threshold=0.5) -> List[AgentDefinition]:
        query_vector = self._embed(query)
        results = []
        for agent in self.agents:
            if agent["name"] in agent.filters or not agent.filters:
                distance = self._cosine_distance(agent["vector"], query_vector)
                if distance >= score_threshold:
                    results.append(agent)
        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]

    def _cosine_distance(a, b):
        if len(a) != len(b):
            raise ValueError("Vectors must be the same length.")

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        # Cosine similarity
        cos_sim = dot_product / (norm_a * norm_b)

        # Cosine distance
        return 1.0 - cos_sim

    def _embed(self, text):
        if self.embedding_model is None:
            raise ValueError("No embedding model provided.")
        return self.embedding_model.embed_query(text)

library = agent_library(embedding_model)

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

@tool 
def agent_selector_tool(query: str) -> Command:
    """
    A tool that selects the most relevant agent based on the user input.
    :return: The name of the selected agent.
    """
    agent = library.search_agents(query)[0]["name"]
    return Command(update={"messages": [{"role": "system", "content": f"call: {agent}"}]}, goto=agent)

#-----------------------------------------------------------------------------------------------
@library.add_agent("product_search_agent")
def product_search_agent(state: State) -> Command[Literal["order_agent", "human_input_agent", "product_search_tool", "agent_selector_tool"]]:
    """
    Your are an expert providing information about available furniture to purchase.
    """

    prompt = """
        Your are an expert providing information about available furniture to purchase.

        1. Search for a product based on the user input, using your tool.
        2. If you find a product, return the product name and description.
        3. Ask the user if he is satisfied with the result by calling the human_input_agent
        4. If the user is not satisfied, ask for more details and call the products_search_tool again.
        5. If the user is satisfied, continue to the order_agent.
        6. If you don't find a product, ask the user if he wants to change his search.
        7. If the user wants to change his search, return to step 1.

        To select the next agent to call, use your provided tool.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt), ("placeholder", "{input}")]
    )
    call = prompt_template | llm.bind_tools([product_search_tool, agent_selector_tool], tool_choice="auto")
    result = call.invoke({"input": state["messages"]})
    return generate_route(result)()

workflow.add_node("product_search_agent", product_search_agent)
workflow.add_node("product_search_tool", ToolNode([product_search_tool]))
workflow.add_node("agent_selector_tool", ToolNode([agent_selector_tool]))

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
    return generate_route(result)()

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

workflow.add_edge(START, "product_search_agent")

workflow.add_edge("product_search_tool", "product_search_agent")
workflow.add_edge("agent_selector_tool", "product_search_agent")
workflow.add_edge("order_tool", "order_agent")
workflow.add_edge("order_agent", END)

graph = workflow.compile()
graph.name = "Prodcut Search Graph"

