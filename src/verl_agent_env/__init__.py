"""
VeRL Agent Environment - Environment Examples of LLM Agents for VeRL integration
"""
import gymnasium as gym

__version__ = "0.1.0"

ALL_VERL_ENVS = []

gym.register(
    id="verl_env/countdown-v0",
    entry_point="verl_agent_env.env:CountdownEnv",
)
ALL_VERL_ENVS.append("verl_env/countdown-v0")