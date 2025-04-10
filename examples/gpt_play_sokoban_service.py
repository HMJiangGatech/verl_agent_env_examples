import requests
import json
import os
from openai import OpenAI

# Try to get key from environment variable first, fallback to file
key = os.getenv("OPENAI_KEY")
if not key:
    with open(os.path.join(os.path.dirname(__file__), "OPENAI_KEY")) as f:
        key = f.read().strip()
# create client
client = OpenAI(api_key=key)

# Base URL for the FastAPI service
base_url = "http://localhost:8000/api"

# Initialize the environment
print("\n\n###### Initialize Environment ######\n\n")
init_response = requests.post(
    f"{base_url}/environment/initialize", 
    json={
        "env_name": "verl_env/sokoban-v0", 
        "env_kwargs": {
            "dim_room": [5, 5],
            "num_boxes": 1
        }
    }
)
init_obj = init_response.json()
env_id = init_obj["env_id"]
print("Initialized Environment ID:", env_id)
print("Initial Message:", init_obj["message"])
print("Initial Observation:", init_obj["observation"])
print("Initial Info:", init_obj["info"])

# Get task prompt
prompt_response = requests.get(f"{base_url}/environment/{env_id}/task-prompt")
task_prompt = prompt_response.json()["task_prompt"]
print("Task Prompt:", task_prompt)

# Get tools schema
schema_response = requests.get(f"{base_url}/environment/{env_id}/tools-schema-openai")
tools_schema = schema_response.json()["tools_schema"]
print("Tools Schema:", tools_schema)

# Agent Loop
print("\n\n###### Agent Loop ######\n\n")
max_turns = 20

messages = [
    {"role": "system", "content": task_prompt + "\n\n" + "You need to generate step by step thought process and reason about your next action. You have to generate tool call for each turn."},
]
for m in init_obj["observation"]:
    messages.append(m)
for turn in range(max_turns):
    print("\n## Turn:", turn)

    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools_schema,
        parallel_tool_calls=False,
        tool_choice="required",
    )
    message = completion.choices[0].message.to_dict()
    message = {
        "role": message['role'],
        "content": message['content'],
        "tool_calls": message['tool_calls'] if 'tool_calls' in message else []
    }
    messages.append(message)
    print("Message Content:\n", message['content'])
    print("Message Tool Calls:\n", json.dumps(message['tool_calls'], indent=2))

    # Take a step in the environment
    step_response = requests.post(f"{base_url}/environment/{env_id}/step", json={"action": message})
    result = step_response.json()
    if len(result['observation']) > 0:
        for m in result['observation']:
            messages.append(m)
            print("Result Observation Content:\n", m['content'])
    print("Reward:", result['reward'])

    if result['done'] or result['truncated']:
        break


print("\n\n###### Close Environment ######\n\n")
# Close the environment
close_response = requests.post(f"{base_url}/environment/{env_id}/close")
close_obj = close_response.json()

print("\n\n###### Final Messages ######\n\n")
print(json.dumps(messages, indent=2)) 