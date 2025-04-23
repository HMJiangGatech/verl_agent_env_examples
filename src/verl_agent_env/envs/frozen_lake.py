"""
Adapted from https://gymnasium.farama.org/environments/toy_text/frozen_lake/
"""

import asyncio
from typing import Optional, Tuple
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from verl_agent_env.envs.base import LLMAgentEnv


class FrozenLakeEnv(LLMAgentEnv):
    """
    Frozen Lake environment.
    """
    def __init__(self, 
                 map_size: int = 8,
                 frozen_prob: float = 0.8,
                 is_slippery: bool = False) -> None:
        super().__init__()
        self.map_size = map_size
        self.frozen_prob = frozen_prob
        self.frozen_lake_env = None
        self._is_slippery = is_slippery
        
        self._action_space_json_schema = [
            {
                "name": "move_left",
                "description": "Move left",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "move_right",
                "description": "Move right",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "move_up",
                "description": "Move up",
                    "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "move_down",
                "description": "Move down",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
        self._tool_name_action_id_map = {
            "move_left": 0,
            "move_down": 1,
            "move_right": 2,
            "move_up": 3,
        }
        
        self._last_tool_call_id = None
        
    
    def _get_obs(self) -> Tuple[dict, ...]:
        """Generate a string representation of the current state of the environment.

        Returns:
            Tuple[dict, ...]: A tuple containing a dictionary with the tool response and results.
        """
        desc = self.frozen_lake_env.unwrapped.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        # replace "S" with "F"
        for i, row in enumerate(desc):
            for j, col in enumerate(row):
                if col == "S":
                    desc[i][j] = "F"
        row, col = self.frozen_lake_env.unwrapped.s // self.frozen_lake_env.unwrapped.ncol, self.frozen_lake_env.unwrapped.s % self.frozen_lake_env.unwrapped.ncol
        desc[row][col] = "P"
        obs = f"You are at position {row},{col} in the world.\n"
        obs += "The current map of the world is: \n"
        obs += "\n".join(["".join(row) for row in desc])
        
        if self._last_tool_call_id is not None:
            return (
                {
                    "role": "tool",
                    "tool_call_id": self._last_tool_call_id,
                    "content": obs
                },
            )
        else:
            return (
                {
                    "role": "user",
                    "content": obs
                },
            )
    
    def _get_info(self) -> dict:
        return {}
    
    async def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        await super().reset(seed=seed)
        desc = generate_random_map(size=self.map_size, p=self.frozen_prob, seed=seed)
        self.frozen_lake_env = gym.make("FrozenLake-v1", desc=desc, map_name=None, is_slippery=self._is_slippery, render_mode="ansi")
        self.frozen_lake_env.reset(seed=seed)
        self._last_tool_call_id = None
        return self._get_obs(), self._get_info()
    
    async def step(self, action):
        action = action['tool_calls']
        if len(action) == 0:
            return [], 0.0, True, False, self._get_info()
        
        assert len(action) == 1, "Only one action is allowed"
        action = action[0]
        assert action["type"] == "function", "Only function call is allowed"
        self._last_tool_call_id = action["id"]
        action = action["function"]
        action_id = self._tool_name_action_id_map[action["name"]]
        obs, reward, terminated, truncated, info = self.frozen_lake_env.step(action_id)
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
        
    
    @property
    def task_prompt(self) -> str:
        # The game starts with the player at location [0,0] of the frozen lake grid world with the goal located at far extent of the world e.g. [3,3] for the 4x4 environment.

        # Holes in the ice are distributed in set locations when using a pre-determined map or in random locations when a random map is generated.

        # The player makes moves until they reach the goal or fall in a hole.
        prompt = (
            "Welcome, robot player! Your mission is to navigate across the frozen lake to reach the goal.\n"
            "The frozen lake is a treacherous path where you must avoid falling into holes while making your way from the start to the goal.\n"
        )
        if self._is_slippery:
            prompt += "Be cautious: the icy surface is slippery, and you might not always move in the direction you intend.\n"
        prompt += (
            "You will begin your journey at the starting point [0,0] on the grid, with the goal located at the opposite corner, such as [3,3] in a 4x4 grid.\n"
            "Holes in the ice are strategically placed on predetermined maps or randomly generated on others.\n"
            "Continue moving until you successfully reach the goal or encounter a hole.\n"
        )
        if self._is_slippery:
            prompt += "The lake is slippery, so you may move perpendicular to the intended direction sometimes.\n"
            prompt += "More specifically, you will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.\n"
            prompt += "For example, if you are at position [x,y] and you move left, you may end up at position [x,y-1] or [x-1,y] or [x+1,y] with equal probability.\n"
            
        prompt += "Randomly generated worlds will always have a path to the goal. "
        
        prompt += "You will be given the current map of the world every time you move. It is a 2D grid where each cell is either the player's position, a hole, a frozen cell, or the goal. "
        prompt += "The letter 'P' represents the player's position, the letter 'G' represents the goal, the letter 'H' represents a hole, and the letter 'F' represents a frozen cell."
        prompt += (
            "\n\n"
            "Here is the reward structure of the game:\n"
            "- You will be rewarded 1 point for reaching the goal.\n"
            "- You will be rewarded 0 points for moving in the ice.\n"
            "- You will be rewarded 0 point for falling into a hole.\n"
        )
        
        return prompt
    
    @property
    def action_space_json_schema(self):
        return self._action_space_json_schema
    
if __name__ == "__main__":
    async def main():
        env = FrozenLakeEnv()
        obs, info = await env.reset()
        print(obs)
        print(info)
        action = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "move_left", "arguments": "{}"}}]
        }
        print(await env.step(action))
        action = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "move_right", "arguments": "{}"}}]
        }
        print(await env.step(action))
        action = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "move_up", "arguments": "{}"}}]
        }
        print(await env.step(action))
        action = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "move_down", "arguments": "{}"}}]
        }
        print(await env.step(action))
    asyncio.run(main())