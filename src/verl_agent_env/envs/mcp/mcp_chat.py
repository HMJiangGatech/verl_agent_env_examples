import os
import asyncio
import json
from typing import Optional, Tuple, List, Dict, Any
from verl_agent_env.envs.base import LLMAgentEnv
from verl_agent_env.envs.mcp.mcp_client import MCPClient

from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.types import TextContent
from mcp.client.stdio import stdio_client

class MCPChatEnv(LLMAgentEnv):
    """
    Single turn chat environment.
    This is only an example of how to use the MCPClient.
    The reward is always 0, and should be overridden by the subclass.
    """
    def __init__(self, 
                 chat_history: List[Dict[str, Any]] = None, 
                 task_prompt: str = "",
                 mcp_config: Optional[dict] = None,
                 ) -> None:
        super().__init__()
        self.chat_history = chat_history or []
        self._task_prompt = task_prompt
        self.mcp_config = mcp_config

        if self.mcp_config is None:
            self.mcp_config = {}
        self.mcp_client_dict = None
        self._action_space_json_schema = None
        self._tool_mcp_client_map = None

    async def _cleanup_mcp_servers(self):
        if self.mcp_client_dict is not None:
            await asyncio.gather(*[client.cleanup() for client in self.mcp_client_dict.values()])
        self.mcp_client_dict = None
        self._action_space_json_schema = None
        self._tool_mcp_client_map = None
    
    def _get_info(self) -> dict:
        return {}
    
    async def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if options is not None and 'reset_mcp_servers' in options and options['reset_mcp_servers'] is True:
            # explicitly reset the mcp servers
            await self._cleanup_mcp_servers()

        if self.mcp_client_dict is None:
            self.mcp_client_dict = {}
            self._action_space_json_schema = []
            self._tool_mcp_client_map = {}
            for server_name, server_config in self.mcp_config.items():
                self.mcp_client_dict[server_name] = MCPClient(**server_config)
                await self.mcp_client_dict[server_name].initialize()
                tool_schema = self.mcp_client_dict[server_name].get_tool_json_schema()
                self._action_space_json_schema.extend(tool_schema)
                for tool in tool_schema:
                    self._tool_mcp_client_map[tool["name"]] = server_name
                    
        return self.chat_history, self._get_info()
    
    async def step(self, action):
        reward = 0

        # handle the tool call
        tool_calls = action["tool_calls"]
        obs = []
        done = True # Assume the episode is done if the action is not a tool call
        for tool_call in tool_calls:
            done = False
            tool_name = tool_call["function"]["name"]
            tool_id = tool_call["id"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            mcp_client = self.mcp_client_dict[self._tool_mcp_client_map[tool_name]]
            tool_result = await mcp_client.execute_tool(tool_name, tool_args)
            obs.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result
            })
        
        # Fill in your own logic here
        # e.g., compute the reward based on the tool calls

        return obs, reward, done, False, self._get_info()
        
    @property
    def task_prompt(self) -> str:
        return self._task_prompt
    
    @property
    def action_space_json_schema(self):
        if self._action_space_json_schema is None:
            raise ValueError("Action space JSON schema not initialized. Call reset() first.")

        return self._action_space_json_schema

    async def close(self):
        await self._cleanup_mcp_servers()
    
if __name__ == "__main__":
    async def main():
        mcp_config = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/jhaoming/Desktop"]
            },
            "google-search": {
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
        }
        env = MCPChatEnv(chat_history=[
            {
                "role": "user",
                "content": "Hello, how are you?",
            },
            {
                "role": "assistant",
                "content": "I'm good, thank you!",
            },
            {
                "role": "user",
                "content": "What is the weather in Tokyo?",
            }
        ], mcp_config=mcp_config)
        obs, info = await env.reset()
        print(obs)
        print(info)
        print(json.dumps(env.action_space_json_schema, indent=2))
        action = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": json.dumps({"query": "What is the weather in Tokyo?"})}},
                {"id": "call_124", "type": "function", "function": {"name": "list_allowed_directories", "arguments": json.dumps({})}}
            ]
        }
        print(json.dumps((await env.step(action))[0], indent=2))


    asyncio.run(main())