# Environment that can be integrated with MCP servers
# This mostly built upon https://modelcontextprotocol.io/quickstart/client
# The MCP client reads the config that follows the same format as the one used in Claude Desktop
# Example config:
# {
#   "mcpServers": {
#     "filesystem": {
#       "command": "npx",
#       "args": [
#         "-y",
#         "@modelcontextprotocol/server-filesystem",
#         "/Users/username/Desktop",
#         "/path/to/other/allowed/dir"
#       ]
#     }
#   }
# } 
# The current implementation build MCP connections on the fly, i.e., build-use-destroy. 
# The is minaly because the challenge of managing the long lifecycle with multiple MCP sessions.
# The issue is tracked in https://github.com/modelcontextprotocol/python-sdk/issues/577

import os
import json
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.types import TextContent
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, command: str, args: list[str], env: Optional[dict] = None):
        self.command = command
        self.args = args
        self.tools = None
        self.env = env
        self.server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env
        )

    async def initialize(self):
        """Connect to an MCP server
        """
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(
                read, write,
            ) as session:
                await session.initialize()
                self.tools = (await session.list_tools()).tools

    def get_tool_json_schema(self) -> list[dict]:
        """Get the JSON schema for the tools"""
        assert self.tools is not None, "Tools not initialized, call connect_to_server first"
        schema = []
        for tool in self.tools:
            schema.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            })
        return schema

    async def execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(
                read, write,
            ) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args)

        assert all([isinstance(c, TextContent) for c in result.content]), "Result must be a list of TextContent, other types of MCP content are not supported yet"
        # print(result)
        # return result
        result_text = "\n".join([c.text for c in result.content])
        return result_text
    
    async def cleanup(self):
        """Clean up resources"""
        self.tools = None
    
if __name__ == "__main__":
    async def main():
        mcp_config = {
            "command": "npx",
            "args": [
                "-y",
                "@adenot/mcp-google-search"
            ],
            "env": {
                "GOOGLE_API_KEY": os.environ["GOOGLE_API_KEY"],
                "GOOGLE_SEARCH_ENGINE_ID": os.environ["GOOGLE_SEARCH_ENGINE_ID"]
            }
        }
        client = MCPClient(**mcp_config)
        await client.initialize()
        print(json.dumps(client.get_tool_json_schema(), indent=2))
        tool_result = await client.execute_tool("search", {"query": "What is the Tesla stock price?"})
        print(tool_result)


        client2 = MCPClient(**mcp_config)
        await client2.initialize()
        tool_result = await client2.execute_tool("search", {"query": "What is the NVDA stock price?"})
        print(tool_result)

    asyncio.run(main())