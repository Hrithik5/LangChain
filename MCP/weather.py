from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location:str) -> str:
    """Get the weather of the location""" 
    return "It is always rainy in Boston"

# 
if __name__ == "__main__":
    mcp.run(transport="streamable-http")