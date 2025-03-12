from langchain_core.tools.base import InjectedToolCallId, ToolCall
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState
from langchain_core.messages import AIMessage, MessageLikeRepresentation, ToolMessage
from typing import Any, Dict, List, Literal, Annotated, TypedDict, cast, Sequence
from dataclasses import dataclass
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
import inspect

@dataclass
class Goto:
    goto: str
    payload: str = None
    type: str = None
    def __call__(self):
        return {
            "type": self.type,
            "content": self.payload,
        }

# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def create_agent(graph: StateGraph, llm, agent_name: str, prompt: str, inputs: Sequence[MessageLikeRepresentation] = (), tools: List[callable] = None, reiterate_after_run: bool = False, graph_destinations: List[str] = None):
    tool_node_name = f"{agent_name}_tools"
    destinations = ([tool_node_name] if tools else []) + (graph_destinations or [])

    def agent_runner(state: State):
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt), ("placeholder", "{input}"), *inputs]
        )
        if tools:
            call = prompt_template | llm.bind_tools(tools, tool_choice="auto")
        else:
            call = prompt_template | llm
        results:AIMessage = call.invoke({"input": state["messages"]})

        if len(results.tool_calls) > 0:
            return Command(goto=tool_node_name, update={"messages": [results]})

        if reiterate_after_run:
            return Command(goto=agent_name, update={"messages": [results]})
        
        return {"messages": [results]}

    graph.add_node(agent_name, agent_runner, destinations=tuple(destinations))

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
                tool_call = {**tool_call, "args": {**tool_call["args"], "state": state}}
                tool_response = tool.invoke(tool_call)
                if isinstance(tool_response, ToolMessage):
                    results.append(Command(update={"messages": [tool_response]}))
                elif isinstance(tool_response, Command):
                    goto = tool_response.goto
                    updates = tool_response.update
                    response = ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=updates,
                        type="tool",
                    )
                    results.append(Command(update={"messages": [response]}, goto=goto))
            return results
        
        graph.add_node(tool_node_name, tool_runner)
        graph.add_edge(tool_node_name, agent_name)