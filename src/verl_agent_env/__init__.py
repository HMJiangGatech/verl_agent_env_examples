"""
VeRL Agent Environment - Environment Examples of LLM Agents for VeRL integration
"""
import gymnasium as gym
from verl_agent_env.envs.countdown import CountdownEnv
from verl_agent_env.envs.frozen_lake import FrozenLakeEnv
from verl_agent_env.envs.sokoban.sokoban import SokobanEnv
from verl_agent_env.envs.single_turn_chat import SingleTurnChatEnv
from verl_agent_env.envs.mcp.mcp_chat import MCPChatEnv

__version__ = "0.1.0"

ALL_VERL_ENVS = {
    "verl_env/countdown-v0": CountdownEnv,
    "verl_env/frozen_lake-v1": FrozenLakeEnv,
    "verl_env/sokoban-v0": SokobanEnv,
    "verl_env/single_turn_chat-v0": SingleTurnChatEnv,
    "verl_env/mcp_chat-v0": MCPChatEnv,
}

# check if src/verl_agent_env/amzn_env/__init__.py exists
# if it does, import and register all the environments in that file
try:
    from verl_agent_env.amzn_env import INTERNAL_AMZN_VERL_ENVS
    ALL_VERL_ENVS.update(INTERNAL_AMZN_VERL_ENVS)
except ImportError:
    pass