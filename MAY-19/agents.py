class ResearcherAgent:
    def __init__(self,tool):
        self.tool = tool

    async def handle(self, task):
        if "fetch" in task.lower():
            return await self.tool.fetch(task)
        return None
    
class SummarizerAgent:
    def __init__(self, tool):
        self.tool = tool

    async def handle(self, content):
        # If content is a string and not a task, summarize it directly
        if isinstance(content, str) and not content.lower().startswith(("fetch", "summarize")):
            return await self.tool.summarize(content)
        return None