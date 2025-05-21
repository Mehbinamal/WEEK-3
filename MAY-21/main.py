import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from io import StringIO

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

from DataFetcher import fetch_csv_data
from Visualizer import visualize_csv_data, visualize_statistics

# Load environment variables
load_dotenv()

async def main():
    # Create model client
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Define tools
    fetch_tool = FunctionTool(
        fetch_csv_data,
        description="Loads CSV data from a specified file path."
    )

    vis_tool = FunctionTool(
        visualize_csv_data,
        description="Creates line, bar, and scatter plots from CSV data."
    )

    stats_tool = FunctionTool(
        visualize_statistics,
        description="Generates statistical plots (mean, variance, etc.) from data."
    )

    # Define agents
    user_proxy = UserProxyAgent(name="user")

    fetcher = AssistantAgent(
        name="Fetcher",
        model_client=model_client,
        description="Loads and validates CSV data",
        system_message=(
            "You're responsible for loading and verifying CSV data. "
            "Use the `fetch_csv_data` tool. Ensure there are no missing values, and data is clean."
        ),
        tools=[fetch_tool]
    )

    analyst = AssistantAgent(
        name="Analyst",
        model_client=model_client,
        description="Analyzes data patterns and suggests visualizations",
        system_message=(
            "Analyze the fetched data. Suggest which columns to visualize, statistical summaries, and insights."
        )
    )

    visualizer = AssistantAgent(
        name="Visualizer",
        model_client=model_client,
        description="Creates plots and statistical graphs",
        system_message=(
            "Use the `visualize_csv_data` and `visualize_statistics` tools to generate appropriate plots "
            "based on the analyst's suggestions. Save all plots to the 'plots' directory."
        ),
        tools=[vis_tool, stats_tool]
    )

    # Define group chat
    termination_condition = TextMentionTermination("Terminate")
    group_chat = RoundRobinGroupChat(
        [user_proxy, fetcher, analyst, visualizer],
          termination_condition=termination_condition)

    # Task prompt
    task = (
        "Analyze and visualize the dataset in 'MAY-21/data.csv'.\n"
        "1. Fetcher: Load and verify the CSV.\n"
        "2. Analyst: Analyze and suggest visualizations.\n"
        "3. Visualizer: Create the plots and save them in 'MAY-21/plots'.\n"
        "Reply TERMINATE when all visualizations are done."
    )

    print("Starting multi-agent data analysis...\n")
    await Console(group_chat.run_stream(task=task))
    print("\n✅ All tasks completed.")

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
