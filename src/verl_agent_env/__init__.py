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
