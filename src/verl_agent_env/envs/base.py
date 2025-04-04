import copy
import gymnasium as gym
from gymnasium import Env
from typing import List, Optional
import json
import string

class LLMAgentEnv(Env):

    def __init__(self) -> None:
        super().__init__()
        # action and observation space is following https://platform.openai.com/docs/guides/function-calling?api-mode=chat
        # An example of action is:
        # {
        #     "role": "assistant",
        #     "content": "Here are some thinking process",
        #     "tool_calls": [
        #         {
        #             "id": "call_12345xyz",
        #             "type": "function",
        #             "function": {
        #                 "name": "get_weather",
        #                 "arguments": "{\"location\":\"Paris, France\"}"
        #             }
        #         },
        #         {
        #             "id": "call_67890abc",
        #             "type": "function",
        #             "function": {
        #                 "name": "get_weather",
        #                 "arguments": "{\"location\":\"Bogotá, Colombia\"}"
        #             }
        #         },
        #         {
        #             "id": "call_99999def",
        #             "type": "function",
        #             "function": {
        #                 "name": "send_email",
        #                 "arguments": "{\"to\":\"bob@email.com\",\"body\":\"Hi bob\"}"
        #             }
        #         }
        #     ]
        # }
        self.action_space = gym.spaces.Dict({
            "role": gym.spaces.Text(16),
            "content": gym.spaces.Text(1024, charset=string.printable),
            "tool_calls": gym.spaces.Sequence(
                gym.spaces.Dict(
                    {
                        "id": gym.spaces.Text(256, charset=string.printable),
                        "type": gym.spaces.Text(16),
                        "function": gym.spaces.Dict(
                            {
                                "name": gym.spaces.Text(1024, charset=string.printable),
                                "arguments": gym.spaces.Text(1024, charset=string.printable)
                            }
                        )
                    }
                    )
            )
        })

        # An example of observation is:
        # [
        #     {
        #         "role": "tool",
        #         "tool_call_id": "call_12345xyz",
        #         "content": "The weather in Paris today is sunny with a temperature of 20 degrees Celsius."
        #     },
        #     {
        #         "role": "tool",
        #         "tool_call_id": "call_67890abc",
        #         "content": "The weather in Bogotá today is cloudy with a temperature of 25 degrees Celsius."
        #     },
        #     {
        #         "role": "tool",
        #         "tool_call_id": "call_99999def",
        #         "content": "The email has been sent to bob@email.com."
        #     }
        # ]
        self.observation_space = gym.spaces.Sequence(
            gym.spaces.Dict(
                {
                    "role": gym.spaces.Text(16),
                    "tool_call_id": gym.spaces.Text(256, charset=string.printable),
                    "content": gym.spaces.Text(1024, charset=string.printable)
                }
            )
        )


    @property
    def task_prompt(self) -> str:
        """
        Returns the task prompt for the environment.

        Returns:
            str: A string containing the task prompt that describes what the agent should do.
        """
        raise NotImplementedError("task_prompt property is not implemented")
    
    @property
    def action_space_json_schema(self):
        """
        Retrieve the action space of the environment, and return the json schema.
        The Json Schema is following https://json-schema.org/docs , which can be naturally integrated
        with popular agent framework, including OpenAI, Claude, or Nova.

        Returns:
            The json schema of action space of the environment.
        """
        raise NotImplementedError("action_space_json_schema property is not implemented")
    
    @property
    def tools_json_schema_openai(self):
        """
        Retrieve the tools json schema of the environment, and return the json schema.
        This is built upon the action_space_json_schema, and is used for OpenAI function calling.
        See https://platform.openai.com/docs/guides/function-calling?api-mode=chat for more details.
        """
        schema = self.action_space_json_schema
        tools = []
        for tool in schema:
            tools.append({
                "type": "function",
                "function": tool
            })
        return tools
    
    @property
    def tools_json_schema_anthropic(self):
        """
        Retrieve the tools json schema of the environment, and return the json schema.
        This is built upon the action_space_json_schema, and is used for Anthropic function calling.
        See https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview for more details.
        """
        schema = self.action_space_json_schema
        tools = []
        for tool in schema:
            tool = copy.deepcopy(tool)
            tool['input_schema'] = tool.pop('parameters')
            tools.append(tool)
        return tools

