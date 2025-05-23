import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from vector_store import query_vector_store
from RAG import RAGSystem
import asyncio

load_dotenv()

async def main():
    #model client
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    #create assistant agent
    query_agent = AssistantAgent(
        name="Query_Agent",
        model_client=model_client,
        description="A helpful assistant that can answer questions and provide information",
        system_message="You are a helpful assistant that uses the query_vector_store tool to search and retrieve relevant information",
        tools=[query_vector_store]
    )

    rag_agent = AssistantAgent(
        name="RAG_Agent",
        model_client=model_client,
        description="A helpful assistant that can answer questions and provide information",
        system_message="You are a helpful assistant that provide comprehensive answers based on retrieved information from chromadb",
    )


    termination_condition = TextMentionTermination("TERMINATE")

    #create team
    team = RoundRobinGroupChat(
        [query_agent, rag_agent],
        termination_condition=termination_condition
    )

    await Console(team.run_stream(task='what is the model of the laptop?'))

    await model_client.close()
    
if __name__ == "__main__":
    asyncio.run(main())
