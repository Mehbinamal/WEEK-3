import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from codeDebugger import debug_code_with_pylint
from codewriter import codewriter


async def main():
    # Create the model client
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key="AIzaSyCojNDEiD2Lhphj_9vRKbSRItedf-PUQ7o",
    )

    code_writer_agent = AssistantAgent(
        "code_writer_agent",
        model_client=model_client,
        description="An assistant that writes Python code based on requirements.",
        system_message="""You are a helpful assistant that can write Python code based on given requirements. 
        You have access to the following function:
        - codewriter(prompt, file_path="MAY-20/generated_function.py"): Generates code using Gemini AI and saves it to a file.
        Use this function to implement the code when needed."""
    )

    debugger_agent = AssistantAgent(
        "debugger_agent",
        model_client=model_client,
        description="An assistant that debugs Python code using pylint.",
        system_message="""You are a helpful assistant that can debug Python code using pylint. 
        You have access to the following function:
        - debug_code_with_pylint(file_path): Analyzes code using Pylint and returns the score and messages.
        Use this function to analyze and fix code issues when needed."""
    )

    termination = TextMentionTermination("TERMINATE")
    group_chat = RoundRobinGroupChat(
        [code_writer_agent, debugger_agent],
        termination_condition=termination,
        max_turns=5
    )
    
    await Console(group_chat.run_stream(
        task="Write a Python function that: \
              1. Takes a list of numbers as input \
              2. Returns the sum of all even numbers in the list"
    ))

    await model_client.close()


# Run the async function
if __name__ == "__main__":
    asyncio.run(main())