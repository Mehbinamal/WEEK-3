import asyncio
from agents import ResearcherAgent, SummarizerAgent
from tools import WebBrowserTool, TextSummarizerTool
from group_chat import SelectorGroupChat
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file")

# Instantiate tools
browser_tool = WebBrowserTool()
summarizer_tool = TextSummarizerTool()

# Create agents
researcher = ResearcherAgent(tool=browser_tool)
summarizer = SummarizerAgent(tool=summarizer_tool)

# Set up chat controller
chat_controller = SelectorGroupChat(agents=[researcher, summarizer])

async def main():
    try:
        tasks = [
            "fetch new's about ipl",
            "summarize the content retrieved"
        ]
        print("Starting task execution...")
        results = await chat_controller.run(tasks)
        print("\nResults:")
        for i, res in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(res)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up browser
        browser_tool.__del__()

if __name__ == "__main__":
    asyncio.run(main())
