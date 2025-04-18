"""
VeRL Agent Environment - Environment Examples of LLM Agents for VeRL integration
"""
import gymnasium as gym

__version__ = "0.1.0"

ALL_VERL_ENVS = []

gym.register(
    id="verl_env/countdown-v0",
    entry_point="verl_agent_env.envs.countdown:CountdownEnv",
)
ALL_VERL_ENVS.append("verl_env/countdown-v0")

gym.register(
    id="verl_env/frozen_lake-v1",
    entry_point="verl_agent_env.envs.frozen_lake:FrozenLakeEnv",
)
ALL_VERL_ENVS.append("verl_env/frozen_lake-v1")

gym.register(
    id="verl_env/sokoban-v0",
    entry_point="verl_agent_env.envs.sokoban.sokoban:SokobanEnv",
)
ALL_VERL_ENVS.append("verl_env/sokoban-v0")

gym.register(
    id="verl_env/single_turn_chat-v0",
    entry_point="verl_agent_env.envs.single_turn_chat:SingleTurnChatEnv",
)
ALL_VERL_ENVS.append("verl_env/single_turn_chat-v0")

# check if src/verl_agent_env/amzn_env/__init__.py exists
# if it does, import and register all the environments in that file
try:
    from verl_agent_env.amzn_env import INTERNAL_AMZN_VERL_ENVS
    for env in INTERNAL_AMZN_VERL_ENVS:
        ALL_VERL_ENVS.append(env)
except ImportError:
    pass