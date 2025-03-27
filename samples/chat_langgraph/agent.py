from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from tracer import AppInsightsTracer

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class AgentSystem:
    agents: Dict[str, callable]
    links: Dict[str, List[str]]

    def __init__(self):
        self.agents = {}
        self.links = {}
        self._graph = StateGraph(State)
        self._tracer = AppInsightsTracer()

    def _add_to_registry(self, agent_name: str, agent: callable = None):
        if agent_name not in self.agents:
            self.agents[agent_name] = agent
        else:
            raise AssertionError(f"Agent {agent_name} already exists in registry")

    def _add_links(self, agent_name: str, linked_agents: List[str]):
        """Adds links to the registry."""
        if agent_name not in self.links:
            self.links[agent_name] = linked_agents

    def _generate_destinations(self, agent_name: str) -> List[str]:
        """Returns the destinations for a given agent."""
        return self.links.get(agent_name, [])
    
    def compile_graph(self, initial_agent: str):
        self._graph.add_edge(START, initial_agent)
        for agent_name in self.agents:
            self._graph.add_node(agent_name, self.agents[agent_name], destinations=tuple(self._generate_destinations(agent_name)))
        return self._graph.compile()

    def create_hil_agent(self, agent_name: str, next_agents: List[str]) -> str:
        def _create(state: State, config: RunnableConfig):
            """A node for collecting user input."""
            
            user_input = interrupt(value="Ready for user input.")
            with self._tracer.get_tracer().start_as_current_span(agent_name):
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
        self._add_links(agent_name, next_agents)
        self._add_to_registry(agent_name, _create)
        return agent_name

    def create_agent(self, prompt: str, llm: BaseChatModel, agent_name: str, tools: List[callable] = [], next_agents: List[str] = []) -> str:
        """Creates an agent."""
        
        tool_node_name = f"{agent_name}_tools"
        
        class Route(BaseModel):
            result: str = Field(..., description="The agent's response")
            goto: str = Field(..., description="The next agent to call")
            capability_description: str = Field(
                description="A query that can be used to search a vector database for the agent's capabilities",
            )

        parser = PydanticOutputParser(pydantic_object=Route)

        def _create_agent_node(state: State):

            #create list of docstring from next_agents
            next_agent_descriptions = {}
            for agent in next_agents:
                if agent in self.agents:
                    next_agent_descriptions[agent] = self.agents[agent].__doc__
                elif agent == "__end__":
                    next_agent_descriptions[agent] = "End of workflow"
                else:
                    next_agent_descriptions[agent] = "No description available"

            #convert to string
            next_agent_descriptions = "\n".join([f"{k}: {v}" for k, v in next_agent_descriptions.items()])

            extended_prompt = """
            -----------------------------
            Use the folowing information to help you decide which agent to call next. Ignore the info below, if you want to call a tool.:
            Next agents available: {next_agents}
            -----------------------------
            Format instructions: {format_instructions}
            """

            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(f"{prompt}\n\n{extended_prompt}"),
                MessagesPlaceholder("context"),
                MessagesPlaceholder("next_agents"),
                MessagesPlaceholder("format_instructions"),
            ])

            with self._tracer.get_tracer().start_as_current_span(agent_name):
                call = prompt_template | llm.bind_tools(tools, tool_choice="auto")
                raw_result = call.invoke({"context": state["messages"], "next_agents": [next_agent_descriptions], "format_instructions": [parser.get_format_instructions()]})

                if raw_result.tool_calls:
                    return Command(
                        update={"messages": [raw_result]},
                        goto=tool_node_name,
                    )

                result = parser.parse(raw_result.content)

                #vector search for the agent's capabilities

                return Command(
                    update={
                        "messages": [
                            {
                                "role": "ai",
                                "content": result.result,
                            }
                        ]
                    },
                    goto=result.goto,
                ) 
        self._add_links(agent_name, next_agents + [tool_node_name] if tools else [])
        self._add_to_registry(agent_name, _create_agent_node)

        if tools:
            self._graph.add_node(tool_node_name, ToolNode(tools))
            self._graph.add_edge(tool_node_name, agent_name)
        
        return agent_name