# Decision based agentic network routing
Having an agent based network, including external inputs from humands tends to be challenging. Langgraph introduced a command based routing mechanism that offers an alternative approach on how to route by using direct directions instead of edge definitions and conditional if/else statements.

Why langgraph. As of now, langgraph gives the developer the most flexibility when it comes to developing agent based systems. I found that especially when using human in the loop patterns, other frameworks like autogen offer a most opinmioted approach that works in some but not all cases as intended.

Langgraph on the other side works nice, but requires boilerplate to work as intended.

Purpose of this sample is to demonstrate a simple but generic network agent setup with langgraph. The systems decided what routes to pick by including it into the completion steps of each agent. Idea was to decentralize the routing and not having a supervised setup.

## Idea
As already mentioned, langraph introduced the Command. With a Command you can directly influence the next jump in an agent system, independend of any pre defined routes.

What that means, is that you could, as long as the agents are registered as nodes in the graph, target any other agent as next agent.

Compared to other frameworks, that losely compares to a team with a broader scope.

I have tested multiple solutions when it came to selecting the next agents. The most promising one was using tools as routers to the next agent, and using tool/functions calls as automatic decission mechanism. Using this method added two things that eventually made me to decide against it. First of was the huge amount of additional "dummy" tools used for routing. Second was that a decision had the potential of getting stuck whenever a summary was created by an llm that should have triggered another routing.

An example for this was the following:
1. User Input: I need a table.
2. Tool call: Search products for table
3. Summarize the products and provide it to the human.

In step three, a summary could look like this: "I have a list of tables for you. Do you like on of them?".
In theory, a call to a human input agent is required at this point. By how LLMs work, the function call for this needs to happen before the LLM summarized the rest of the state and therefore could not have provided and decided to call the tool in the first place.

A solution to that, was to call the same agent one more time after completion. That felt wrong and added side effects like introducing a run counter to stop the llm iterating again and again. Also this increased the token usage by doing essentially one additional full run, including the entire state to decide what agent to call next.

## Solution
The solution is quite intuitive as it uses a somewhat "human" approach. The general idea is to leave the decision on what agent to call next to the LLM whenever it generates its final answer/summary.

How does that work? If you look at the code snipped below, you'll see that the solution uses a structured output definition, that includes the original result of the LLM, the next agent to target and a capability description. The capability description I will describe at a later point.

```
class Route(BaseModel):
    result: str = Field(..., description="The result of the agent's action")
    goto: str = Field(..., description="The next agent to call")
    capability_description: str = Field(
        description="A query that can be used to search a vector database for the agent's capabilities",
    )
```

To tell the LLM to return a formatted result the system prompt of each agent is altered with the following:

```
extended_prompt = """
    ################################################
    Use the folowing information to help you decide which agent to call next:
    Next agents available: {next_agents}
    ################################################
    Format instructions: {format_instructions}
"""
```

Defining both the pydantic output object and the extended prompt enables the model to return the answer for the user and the next agent to call in one single step.

An additional benefit of the provided solution is, that it allows for addional (mechanical) filtering of the next agents to call. Instead of just relying on the "goto" field, fields like the capability_description can be used to call vector databases to search for the next targets to call.

Implementation wise the solution required to create a wrapper around the common agent implementation in langgraph.

## Setup & Run
How to run the sample application?

```
pip install -r requirements.txt

langgraph dev --debug-port 2026

# If in VSCode run Remote Attach - Langgraph Dev Debug Config

```