import gymnasium as gym
from verl_agent_env.env import LLMAgentEnv as Env
from verl_agent_env import ALL_VERL_ENVS
import uuid
from typing import Optional

# A simple in-memory store for environments
environments = {}

def initialize_environment(env_name: str, seed: Optional[int] = None):
    """
    Initialize a new environment with the given name and optional seed.

    Args:
        env_name (str): The name of the environment to initialize.
        seed (Optional[int]): An optional seed for the environment's random number generator.

    Returns:
        dict: A dictionary containing a success message, the environment ID, 
              the initial observation, and additional info.
    """
    # Loop until a unique env_id is generated

    assert env_name in ALL_VERL_ENVS, f"Environment '{env_name}' not found in registered environments. Available environments: {ALL_VERL_ENVS}"

    while True:
        env_id = str(uuid.uuid4())
        if env_id not in environments:
            break
    
    env: Env = gym.make(env_name)
    environments[env_id] = env
    observation, info = env.reset(seed=seed)
    return {
        "message": f"Environment '{env_name}' initialized successfully.",
        "env_id": env_id,
        "observation": observation,
        "info": info
    }

def close_environment(env_id: str):
    """
    Close the environment with the given ID.

    Args:
        env_id (str): The ID of the environment to close.

    Returns:
        dict: A dictionary containing a message indicating whether the environment 
              was closed successfully or if it was not found.
    """
    env: Env = environments.pop(env_id, None)
    if env is not None:
        env.close()
        return {"message": f"Environment with ID '{env_id}' closed successfully."}
    else:
        raise KeyError(f"Environment with ID '{env_id}' not found.")
    
def action_space_json_schema(env_id: str):
    """
    Retrieve the action space of the environment with the given ID in a Json schema format.
    The Json Schema is following https://json-schema.org/docs , which can be naturally integrated
    with popular agent framework, including OpenAI, Claude, or Nova.

    Args:
        env_id (str): The ID of the environment.

    Returns:
        dict: A dictionary containing the action space in a JSON-serializable format.

    Raises:
        KeyError: If the environment with the given ID is not found.
    """
    env: Env = environments.get(env_id, None)
    
    if env is None:
        raise KeyError(f"Environment with ID '{env_id}' not found.")
    
    # Assuming the action space can be represented as a dictionary
    action_space_json_schema = env.unwrapped.action_space_json_schema
    
    return action_space_json_schema

def get_task_prompt(env_id: str):
    """
    Retrieve the task prompt of the environment with the given ID.

    Args:
        env_id (str): The ID of the environment.

    Returns:
        str: The task prompt of the environment.

    Raises:
        KeyError: If the environment with the given ID is not found.
    """
    env: Env = environments.get(env_id, None)
    
    if env is None:
        raise KeyError(f"Environment with ID '{env_id}' not found.")
    
    # Assuming the environment has a method or attribute `task_prompt`
    task_prompt = env.unwrapped.task_prompt
    
    return task_prompt

def tools_json_schema_openai(env_id: str):
    """
    Retrieve the tools JSON schema of the environment with the given ID.

    Args:
        env_id (str): The ID of the environment.

    Returns:
        dict: The tools JSON schema of the environment.

    Raises:
        KeyError: If the environment with the given ID is not found.
    """
    env: Env = environments.get(env_id, None)
    
    if env is None:
        raise KeyError(f"Environment with ID '{env_id}' not found.")
    
    # Assuming the environment has a method or attribute `tools_json_schema`
    tools_schema = env.unwrapped.tools_json_schema_openai
    
    return tools_schema

def take_step(env_id: str, action):
    """
    Take a step in the environment with the given ID using the specified action.

    Args:
        env_id (str): The ID of the environment.
        action: The action to take in the environment. This is a message block following the OpenAI message format.

    Returns:
        dict: A dictionary containing the observation, reward, done status, truncated status, and additional info.

    Raises:
        KeyError: If the environment with the given ID is not found.
    """
    env: Env = environments.get(env_id, None)
    
    if env is None:
        raise KeyError(f"Environment with ID '{env_id}' not found.")
    
    observation, reward, done, truncated, info = env.step(action)
    
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info
    }

# Simple test code
if __name__ == "__main__":
    # Initialize an environment
    result = initialize_environment("verl_env/countdown-v0", seed=42)
    print("Initialize Result:", result)

    # Get action space
    tool_schema = action_space_json_schema(result["env_id"])
    print("Action Space:", tool_schema)

    # Get task prompt
    task_prompt = get_task_prompt(result["env_id"])
    print("Task Prompt:", task_prompt)

    # Close the environment
    close_result = close_environment(result["env_id"])
    print("Close Result:", close_result) 