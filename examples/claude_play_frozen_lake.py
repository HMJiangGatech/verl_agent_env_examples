import json
import boto3
import asyncio
from verl_agent_env import interface

async def main():
    # Get Bedrock Client
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    def build_claude_request_with_tools(messages, system_prompt, tools):
        return json.dumps(
            {
                "max_tokens": 2048,
                "tools": tools,
                "messages": messages,
                "system": system_prompt,
                "anthropic_version": "bedrock-2023-05-31",
                # "tool_choice": {
                #     "type": "any",
                # }
            }
        )

    # Initialize the environment
    print("\n\n###### Initialize Environment ######\n\n")
    init_obj = await interface.initialize_environment("verl_env/frozen_lake-v1", env_kwargs={"map_size": 8})
    env_id = init_obj["env_id"]
    print("Initialized Environment ID:", env_id)
    print("Initial Message:", init_obj["message"])
    print("Initial Observation:", init_obj["observation"])
    print("Initial Info:", init_obj["info"])

    task_prompt = interface.get_task_prompt(env_id)
    print("Task Prompt:", task_prompt)

    tools_schema = interface.tools_json_schema_anthropic(env_id)
    print("Tools Schema:", tools_schema)

    # Agent Loop
    print("\n\n###### Agent Loop ######\n\n")
    max_turns = 20
    system_prompt = task_prompt + "\n\n" + "Before answering, explain your reasoning step-by-step in tags. You HAVE TO generate tool call for each turn!!!"
    messages = [interface.convert_openai_tool_obs_to_claude_obs(init_obj["observation"])]
    for turn in range(max_turns):
        print("\n## Turn:", turn)

        body = build_claude_request_with_tools(messages, system_prompt, tools_schema)
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=body,
        )
        response = json.loads(response['body'].read())
        message = {
            "role": response["role"],
            "content": response["content"]
        }
        messages.append(message)

        message_openai_style = interface.convert_claude_action_to_openai_action(message)

        print("Message Content:\n", message_openai_style['content'])
        print("Message Tool Calls:\n", json.dumps(message_openai_style['tool_calls'], indent=2))
        result = await interface.take_step(env_id, message_openai_style)
        if len(result['observation']) > 0:
            for m in result['observation']:
                print("Result Observation Content:\n", m['content'])
            obs = interface.convert_openai_tool_obs_to_claude_obs(result['observation'])
            if result['truncated'] or result['done']:
                obs['content'].append({
                    "type": "text",
                    "text": "The game is over. You have reached the goal or fallen into a hole."
                })
            else:
                obs['content'].append({
                    "type": "text",
                    "text": "The game is still ongoing. You have not reached the goal or fallen into a hole. Keep playing!"
                })
            messages.append(obs)
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
