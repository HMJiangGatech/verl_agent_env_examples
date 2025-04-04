from openai import OpenAI
import json
import os

from verl_agent_env import interface

# Try to get key from environment variable first, fallback to file
key = os.getenv("OPENAI_KEY")
if not key:
    with open(os.path.join(os.path.dirname(__file__), "OPENAI_KEY")) as f:
        key = f.read().strip()
# create client
client = OpenAI(api_key=key)

# Initialize the environment
print("\n\n###### Initialize Environment ######\n\n")
init_obj = interface.initialize_environment("verl_env/frozen_lake-v1", env_kwargs={"map_size": 8})
env_id = init_obj["env_id"]
print("Initialized Environment ID:", env_id)
print("Initial Message:", init_obj["message"])
print("Initial Observation:", init_obj["observation"])
print("Initial Info:", init_obj["info"])

task_prompt = interface.get_task_prompt(env_id)
print("Task Prompt:", task_prompt)

tools_schema = interface.tools_json_schema_openai(env_id)
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
        model="gpt-4o",
        messages=messages,
        tools=tools_schema,
        parallel_tool_calls=False,
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
    result = interface.take_step(env_id, message)
    if len(result['observation']) > 0:
        for m in result['observation']:
            messages.append(m)
            print("Result Observation Content:\n", m['content'])
    print("Reward:", result['reward'])

    if result['done'] or result['truncated']:
        break


print("\n\n###### Close Environment ######\n\n")
# Close the environment
close_obj = interface.close_environment(env_id)

print("\n\n###### Final Messages ######\n\n")
print(json.dumps(messages, indent=2))