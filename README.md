# VeRL Agent Environment

Environment Examples of LLM Agents, designed to be integrated with VeRL.

This project strictly follows the conventions of Gymnasium (previously OpenAI Gym) for creating and managing environments.

## Local Development Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

   This will install:
   - All required dependencies
   - Development tools:
     - pytest: For running tests
     - pytest-cov: For test coverage reporting
     - black: Code formatter
     - isort: Import sorter
     - mypy: Static type checker

## Development Commands

### Running Tests
   ```bash
   pytest
   ```

### Format Code
   ```bash
   # Run black formatter
   black .

   # Sort imports
   isort .
   ```

### Type Checking
   ```bash
   mypy src
   ```

## Interface Design

### 1. Python Interface

- **Initialize an Environment**
  ```python
  import verl_agent_env.interface as interface

  env_id = interface.initialize_environment("ENV-NAME")
  observation, info = interface.reset(env_id)
  ```

- **Step Through the Environment**
  ```python
  action = ...  # Define your action here
  observation, reward, done, truncated, info = interface.step(env_id, action)
  ```

- **Close and Clean Up the Environment**
  ```python
  interface.close_environment(env_id)
  ```

**Compatability with LLM Chat Message List**: To make the environment compatible with LLM Chat Message List, the `observation` are designed to be a list of dictionaries (messages) with the following keys:
- `role`: The role of the message.
- `content`: The content of the message.
- (Optional) `tool_call_id`: The tool call id of the message. If the `role` is `tool`, the `tool_call_id` is the id of the tool call. If the `role` is `user`, there is no `tool_call_id`.

The `action` is also directly compatible with the `messages` in the popular LLM API (e.g. OpenAI, Anthropic, etc.). More specifically, the `action` is a list of dictionaries with the following keys:
- `role`: The role of the message, which is usually `assistant`.
- `content`: The content of the message.
- `tool_calls`: The tool calls of the message. This is a list of dictionaries with the following keys:
  - `id`: The id of the tool call.
  - `type`: The type of the tool call, which is usually `function`.
  - `function`: The function of the tool call. This is a dictionary with the following keys:
    - `name`: The name of the tool.
    - `arguments`: The arguments of the tool call.


### 2. Service Endpoint

- **Initialize an Environment**
  - **Endpoint:** `POST /api/environment/initialize`
  - **Description:** Initializes a new environment instance.
  - **Request Body:** JSON object with `env_name` field.
  - **Response:** JSON object with a message and `env_id`.

- **Close and Clean Up the Environment**
  - **Endpoint:** `POST /api/environment/{env_id}/close`
  - **Description:** Closes and cleans up the environment instance.
  - **Path Parameter:** `env_id` - The ID of the environment.
  - **Response:** JSON object indicating success or failure.

- **Retrieve Action Space JSON Schema**
  - **Endpoint:** `GET /api/environment/{env_id}/action-space`
  - **Description:** Retrieves the action space of the environment in a JSON schema format.
  - **Path Parameter:** `env_id` - The ID of the environment.
  - **Response:** JSON object containing the action space schema or an error message if the environment is not found.

## Running the FastAPI Server

To start the FastAPI server, run the following command:

```bash
uvicorn src.verl_agent_env.app:app --reload
```

This will start the server on `http://127.0.0.1:8000`, and you can access the API documentation at `http://127.0.0.1:8000/docs`.

## Docker Setup

To serve the FastAPI application using Docker, follow these steps:

### Build the Docker Image

1. Ensure Docker is installed and running on your machine.
2. Navigate to the root directory of the project where the `Dockerfile` is located.
3. Build the Docker image using the following command:
   ```bash
   docker build -t verl-agent-env .
   ```

### Run the Docker Container

1. Run the Docker container using the following command:
   ```bash
   docker run -p 8000:8000 verl-agent-env
   ```

This will start the FastAPI server inside a Docker container, and it will be accessible at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

## TODO List

- [ ] Add serving code
- [ ] Add docker container building logic
- [ ] High Concurrency for supporting >= 10K batch size
- [ ] Multi-Node Hosting
- [x] Implement sokoban
- [x] Implement Countdown
- [x] Implement Frozen Lake
- [ ] RL Example with VerL + sokoban
    - [x] Data Curation
    - [x] Verl Dataset
    - [x] Verl Rollout
    - [x] Verl PPO Training
- [ ] MCP Env

## Example Usage

To run the example of using the VeRL Agent Environment with a countdown task, execute the following command in your terminal:

```bash
python examples/gpt_play_countdown.py
```

Ensure that you have set the `OPENAI_KEY` environment variable or have the `OPENAI_KEY` file in the same directory as the script for authentication. This example demonstrates initializing an environment, running an agent loop, and closing the environment.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.