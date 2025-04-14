"""
Adapted from https://gymnasium.farama.org/environments/toy_text/frozen_lake/
"""

from typing import Optional, Tuple, List, Dict, Any
from verl_agent_env.envs.base import LLMAgentEnv


class SingleTurnChatEnv(LLMAgentEnv):
    """
    Single turn chat environment.
    """
    def __init__(self, 
                 chat_history: List[Dict[str, Any]] = None, 
                 task_prompt: str = "") -> None:
        super().__init__()
        self.chat_history = chat_history
        self._task_prompt = task_prompt
        
        self._action_space_json_schema = []
    
    def _get_info(self) -> dict:
        return {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.chat_history, self._get_info()
    
    def step(self, action):
        reward = 0
        
        return [], reward, True, True, self._get_info()
        
    
    @property
    def task_prompt(self) -> str:
        return self._task_prompt
    
    @property
    def action_space_json_schema(self):
        return self._action_space_json_schema
    
if __name__ == "__main__":
    env = SingleTurnChatEnv(chat_history=[
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
    ])
    obs, info = env.reset()
    print(obs)
    print(info)
    action = {
        "role": "assistant",
        "content": "The weather in Tokyo is sunny today.",
    }
    print(env.step(action))
    