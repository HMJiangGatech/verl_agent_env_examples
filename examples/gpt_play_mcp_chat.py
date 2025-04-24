from openai import OpenAI
import json
import os
import asyncio

from verl_agent_env import interface

async def main():
    # Try to get key from environment variable first, fallback to file
    key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        with open(os.path.join(os.path.dirname(__file__), "OPENAI_KEY")) as f:
            key = f.read().strip()
    # create client
    client = OpenAI(api_key=key)

    # Initialize the environment
    print("\n\n###### Initialize Environment ######\n\n")
    mcp_config = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", os.path.expanduser("~/")]
        },
        "google-search": {
            "command": "npx",
            "args": [
                "-y",
                "@adenot/mcp-google-search"
            ],
            "env": {
                "GOOGLE_API_KEY": os.environ["GOOGLE_API_KEY"],
                "GOOGLE_SEARCH_ENGINE_ID": os.environ["GOOGLE_SEARCH_ENGINE_ID"]
            }
        },
        "fetch": {
            "command": "python",
            "args": ["-m", "mcp_server_fetch"]
        }
    }
    init_obj = await interface.initialize_environment("verl_env/mcp_chat-v0",
                                                env_kwargs={
                                                    "task_prompt": "You are a helpful assistant.",
                                                    "chat_history": [
                                                        {
                                                            "role": "user",
                                                            "content": "Find the weather in top-5 cities in the US, and write your response in markdown format. Save them in a file called weather.md in the home directory:" + os.path.expanduser("~/")
                                                        }
                                                    ],
                                                    "mcp_config": mcp_config
                                                })
                                            
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
    max_turns = 15

    messages = [
        {"role": "system", "content": task_prompt},
    ]
    for m in init_obj["observation"]:
        messages.append(m)
    for turn in range(max_turns):
        print("\n## Turn:", turn)

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
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
        result = await interface.take_step(env_id, message)
        if len(result['observation']) > 0:
            for m in result['observation']:
                messages.append(m)
                print("Result Observation Content:\n", m['content'])
        print("Reward:", result['reward'])

        if result['done'] or result['truncated']:
            break


    print("\n\n###### Close Environment ######\n\n")
    # Close the environment
    close_obj = await interface.close_environment(env_id)

    print("\n\n###### Final Messages ######\n\n")
    print(json.dumps(messages, indent=2))

if __name__ == "__main__":
    asyncio.run(main())