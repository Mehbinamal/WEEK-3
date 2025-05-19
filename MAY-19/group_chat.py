from agents import ResearcherAgent, SummarizerAgent

class SelectorGroupChat:
    def __init__(self, agents):
        self.agents = agents
        self.fetched_content = None

    async def run(self, tasks):
        results = []
        
        for task in tasks:
            print(f"\nProcessing task: {task}")
            task_completed = False
            
            # Determine if this is a fetch or summarize task
            is_fetch_task = "fetch" in task.lower()
            is_summarize_task = "summarize" in task.lower()
            
            for agent in self.agents:
                try:
                    # Handle summarization task
                    if is_summarize_task and isinstance(agent, SummarizerAgent):
                        if not self.fetched_content:
                            print("Warning: No content available for summarization")
                            continue
                        print("Passing content to summarizer...")
                        result = await agent.handle(self.fetched_content)
                        if result:
                            results.append(result)
                            task_completed = True
                            break
                    
                    # Handle fetch task
                    elif is_fetch_task and isinstance(agent, ResearcherAgent):
                        print(f"Passing task to {agent.__class__.__name__}...")
                        result = await agent.handle(task)
                        if result:
                            print("Content fetched successfully")
                            self.fetched_content = result
                            results.append(result)
                            task_completed = True
                            break
                    
                    # Handle other tasks
                    else:
                        print(f"Passing task to {agent.__class__.__name__}...")
                        result = await agent.handle(task)
                        if result:
                            results.append(result)
                            task_completed = True
                            break
                            
                except Exception as e:
                    print(f"Error in {agent.__class__.__name__}: {str(e)}")
                    continue
            
            if not task_completed:
                print(f"Warning: Task '{task}' was not completed by any agent")
                
        return results
