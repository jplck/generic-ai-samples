from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables.config import RunnableConfig

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Agent:
    agents: Dict[str, callable]
    links: Dict[str, List[str]]

    def __init__(self, graph: StateGraph):
        self.agents = {}
        self.links = {}
        self._graph = graph

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
    
    def compile_graph(self):
        for agent_name in self.agents:
            self._graph.add_node(agent_name, self.agents[agent_name], destinations=tuple(self._generate_destinations(agent_name)))
        return self._graph.compile()

    def create_hil_agent(self, agent_name: str, next_agents: List[str]) -> str:
        def _create(state: State, config: RunnableConfig):
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
        self._add_links(agent_name, next_agents)
        self._add_to_registry(agent_name, _create)
        return agent_name

    def create_agent(self, prompt: str, llm: BaseChatModel, name: str, tools: List[callable] = [], next_agents: List[str] = []) -> str:
        """Creates an agent."""
        
        tool_node_name = f"{name}_tools"
        
        class Route(BaseModel):
            result: str = Field(..., description="The result of the agent's action")
            goto: str = Field(..., description="The next agent to call")
            capability_description: str = Field(
                description="A query that can be used to search a vector database for the agent's capabilities",
            )

        parser = PydanticOutputParser(pydantic_object=Route)

        def _create_agent_node(state: State):
            template = prompt + "\n\n" + "{format_instructions}"

            prompt_template = PromptTemplate(
                template=template,
                input_variables=["input", "next_agents"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            call = prompt_template | llm.bind_tools(tools, tool_choice="auto")
            raw_result = call.invoke({"input": state["messages"], "next_agents": next_agents})

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
        self._add_links(name, next_agents + [tool_node_name] if tools else [])
        self._add_to_registry(name, _create_agent_node)

        if tools:
            self._graph.add_node(tool_node_name, ToolNode(tools))
            self._graph.add_edge(tool_node_name, name)
        
        return name