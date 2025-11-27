import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


async def main():
    # Use async context manager so sessions are managed correctly
    client =  MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["math_server.py"],  # ensure correct path
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",  # ensure server is running
                "transport": "streamable_http",
            },
        }
    )

        # Defensive: catch network / tool-loading errors
    try:
        tools = await client.get_tools()
    except Exception as e:
        print("Failed to load MCP tools:", repr(e))
        print("Check that your 'weather' service is running at http://localhost:8000/mcp")
        return

    model = ChatGroq(model="llama-3.1-8b-instant")
    agent = create_agent(model, tools)

    # math query
    try:
        math_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
        )
        print("Math response:", math_response["messages"][-1].content)
    except Exception as e:
        print("Math invocation failed:", repr(e))

    # weather query
    try:
        weather_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
        )
        print("Weather response:", weather_response["messages"][-1].content)
    except Exception as e:
        print("Weather invocation failed:", repr(e))


if __name__ == "__main__":
    asyncio.run(main())
