from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, Swarm
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from azure.ai.projects import AIProjectClient
import os
import asyncio
import dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
import json
from samples.chat.model import Document, User
from samples.chat.search_index import index_documents, search_index
from samples.chat.common import get_default_token_provider
from azure.identity import DefaultAzureCredential

dotenv.load_dotenv()

DOCUMENT_INDEX_NAME = "document-index"

completion_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
    model=os.getenv("AZURE_OPENAI_COMPLETION_MODEL"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=get_default_token_provider(),
    model_info={
        "json_output": False,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
    },
)

def parse_templates(json_file_path: str) -> list[Document]:
    with open(json_file_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    docs = []
    for t in templates:
        docs.append(
            Document(
                id=str(t.get("id", "")),
                page_content=t.get("body", ""),
                metadata={
                    "topic": t.get("topic", ""),
                    "subject": t.get("subject", "")
                },
            )
        )
    return docs

#project_client = AIProjectClient.from_connection_string("#####", DefaultAzureCredential())

def search_tool(query: str) -> list[Document]:
    """Tool that searches database for fitting email templates that are relevant to the query."""
    search = search_index(DOCUMENT_INDEX_NAME, query)
    return search

def find_relevant_user_tool(query: str) -> list[User]:
    """Tool that finds the relevant users for a given query."""
    return [User(id="1", email="example@example.com", name="John Doe")]

def send_email_tool(email: str, subject: str, body: str) -> str:
    """Tool that sends an email."""
    return f"Email sent to {email} with subject {subject} and body {body}"

async def main():

    #docs = parse_templates("samples/chat/assets/templates.json")
    #index_documents(DOCUMENT_INDEX_NAME, docs)

    search_agent_prompt = """I am an agent that searches for email templates, based on an input query. 
        I handover to the user_proxy agent to ask for a template selection."""
    search_agent = AssistantAgent(
        description=search_agent_prompt,
        name = "search_agent",
        model_client = completion_model_client,
        tools = [search_tool],
        system_message = search_agent_prompt,
        reflect_on_tool_use = True
    )

    compose_agent_prompt = """I am an agent that customizes the selected template by user input. If required iterate with the user_proxy agent until the user is satisfied with the result. Needs user approval to proceed."""
    compose_agent = AssistantAgent(
        description="Customizes the selected email template by user input.",
        name = "compose_agent",
        model_client = completion_model_client,
        system_message = compose_agent_prompt,
        reflect_on_tool_use = True
    )

    send_email_agent_prompt = """I am an agent preparing email for being send with the final and approved template. Ask the user to give input on what recipient shall be queried. Send the emails to those recipients if the user approves."""
    send_email_agent = AssistantAgent(
        description="Define mail recipients and send emails",
        name = "send_email_agent",
        model_client = completion_model_client,
        tools=[send_email_tool, find_relevant_user_tool],
        system_message = send_email_agent_prompt,
        reflect_on_tool_use = True
    )

    user_proxy = UserProxyAgent("user_proxy", input_func=input)
    termination_condition = TextMentionTermination("EXIT") | MaxMessageTermination(50)

    #team = RoundRobinGroupChat([search_agent, compose_agent, user_proxy], termination_condition=termination_condition)

    selector_prompt = """Your job is to orchestrate the agents to find an email template for an input query.

    You select the agents based on the following flow:
    search_agent <-> user_proxy -> compose_agent <-> user_proxy -> send_email_agent

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the user_proxy is involved whenever input is required.
    Only select one agent.
    """

    participants = [search_agent, compose_agent, send_email_agent, user_proxy]

    team = SelectorGroupChat(
        participants=participants, 
        termination_condition=termination_condition,
        model_client=completion_model_client,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False
    )

    await Console(team.run_stream(task="Find an email template for a product recall."), output_stats=True)

asyncio.run(main())