import json
import asyncio
from verl_agent_env.envs.base import LLMAgentEnv as Env
from verl_agent_env import ALL_VERL_ENVS
import uuid
from typing import Optional

# A simple in-memory store for environments
environments = {}

async def initialize_environment(env_name: str, seed: Optional[int] = None, env_kwargs: Optional[dict] = None):
    """
    Initialize a new environment with the given name and optional seed.

    Args:
        env_name (str): The name of the environment to initialize.
        seed (Optional[int]): An optional seed for the environment's random number generator.
        env_kwargs (Optional[dict]): An optional dictionary of keyword arguments for the environment.
        
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
    
    if env_kwargs is None:
        env_kwargs = {}
    env: Env = ALL_VERL_ENVS[env_name](**env_kwargs)
    environments[env_id] = env
    observation, info = await env.reset(seed=seed, options=env_kwargs)
    return {
        "message": f"Environment '{env_name}' initialized successfully.",
        "env_id": env_id,
        "observation": observation,
        "info": info
    }
    
async def reset_environment(env_id: str, seed: Optional[int] = None, options: Optional[dict] = None):
    """
    Reset the environment with the given ID and return the initial observation and additional info.

    Args:
        env_id (str): The ID of the environment to reset.
        seed (Optional[int]): An optional seed for the environment's random number generator.
        options (Optional[dict]): An optional dictionary of keyword arguments for the environment.

    Returns:
        dict: A dictionary containing the initial observation and additional info.

    Raises:
        KeyError: If the environment with the given ID is not found.
    """
    env: Env = environments.get(env_id, None)
    
    if env is None:
        raise KeyError(f"Environment with ID '{env_id}' not found.")
    
    observation, info = await env.reset(seed=seed, options=options)
    return {
        "observation": observation,
        "info": info
    }

async def close_environment(env_id: str):
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
        await env.close()
        return {"message": f"Environment with ID '{env_id}' closed successfully."}
    else:
        return {"message": f"Environment with ID '{env_id}' not found."}
    
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

def allow_parallel_tool_call(env_id: str):
    """
    Retrieve the allow_parallel_tool_call of the environment with the given ID.
    """
    env: Env = environments.get(env_id, None)
    return env.unwrapped.allow_parallel_tool_call

def tools_json_schema_openai(env_id: str):
    """
    Retrieve the tools JSON schema of the environment with the given ID.

    Args:
        env_id (str): The ID of the environment.

    Returns:
        list: The tools JSON schema of the environment.

    Raises:
        KeyError: If the environment with the given ID is not found.
    """
    env: Env = environments.get(env_id, None)
    
    if env is None:
        raise KeyError(f"Environment with ID '{env_id}' not found.")
    
    # Assuming the environment has a method or attribute `tools_json_schema`
    tools_schema = env.unwrapped.tools_json_schema_openai
    
    return tools_schema

def tools_json_schema_anthropic(env_id: str):
    """
    Retrieve the Anthropic tools JSON schema of the environment with the given ID.

    Args:
        env_id (str): The ID of the environment.

    Returns:
        list: The Anthropic tools JSON schema of the environment.

    Raises:
        KeyError: If the environment with the given ID is not found.
    """
    env: Env = environments.get(env_id, None)
    
    if env is None:
        raise KeyError(f"Environment with ID '{env_id}' not found.")
    
    # Assuming the environment has a method or attribute `tools_json_schema_anthropic`
    tools_schema = env.unwrapped.tools_json_schema_anthropic
    
    return tools_schema

async def take_step(env_id: str, action):
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
    
    observation, reward, done, truncated, info = await env.step(action)
    
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info
    }

def convert_claude_action_to_openai_action(action):
    """
    Convert the action from Claude to OpenAI style.
    """
    text_content = ""
    tool_calls = []
    for c in action["content"]:
        if c["type"] == "text":
            assert text_content == "", "Only one text block is allowed"
            text_content = c["text"]
        elif c["type"] == "tool_use":
            tool_calls.append({
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c["input"])
                }
            })
    return {
        "role": action["role"],
        "content": text_content,
        "tool_calls": tool_calls
    }
        
def convert_openai_tool_obs_to_claude_obs(obs):
    """
    Convert the observation from OpenAI to Claude style.
    """
    claude_obs = []
    for o in obs:
        if o["role"] == "user":
            claude_obs.append({
                "type": "text",
                "text": o["content"]
            })
        elif o["role"] == "tool":
            claude_obs.append({
                "type": "tool_result",
                "tool_use_id": o["tool_call_id"],
                "content": o["content"]
            })
    claude_obs = {
        "role": "user",
        "content": claude_obs
    }
    return claude_obs
        
# Simple test code
if __name__ == "__main__":
    async def main():
        # Initialize an environment
        result = await initialize_environment("verl_env/countdown-v0", seed=42)
        print("Initialize Result:", result)

        # Get action space
        tool_schema = action_space_json_schema(result["env_id"])
        print("Action Space:", tool_schema)

        # Get task prompt
        task_prompt = get_task_prompt(result["env_id"])
        print("Task Prompt:", task_prompt)

        # Close the environment
        close_result = await close_environment(result["env_id"])
        print("Close Result:", close_result) 

    asyncio.run(main())
