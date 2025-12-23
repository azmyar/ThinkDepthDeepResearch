import os
import asyncio
import openai
 
openai.api_key = os.environ["OPENAI_API_KEY"]

from langgraph.checkpoint.memory import InMemorySaver
from src.research_agent_full import deep_researcher_builder
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
import warnings

warnings.filterwarnings("ignore")

async def main():
    # Build the agent
    checkpointer = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
    
    # Run the research task (can take 10 ~ 30 minutes)
    thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}
    result = await full_agent.ainvoke(
        {"messages": [HumanMessage(content="What is AI?")]}, 
        config=thread
    )

    console = Console()
    
    if "final_report" in result:
        console.print(Markdown(result["final_report"]))
    elif "messages" in result:
        # Clarification was requested
        console.print("Clarification needed:")
        console.print(result["messages"][-1].content if hasattr(result["messages"][-1], 'content') else str(result["messages"][-1]))
    else:
        console.print("Unexpected result structure")
        console.print(result)

if __name__ == "__main__":
    asyncio.run(main())