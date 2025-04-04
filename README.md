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

- **Close and Clean Up the Environment**
  ```python
  interface.close_environment(env_id)
  ```

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

## TODO List

- [ ] Add serving code
- [ ] Add docker container building logic
- [ ] High Concurrency for supporting >= 10K batch size
- [ ] Multi-Node Hosting
- [ ] Implement sokoban
- [x] Implement Countdown

## Example Usage

To run the example of using the VeRL Agent Environment with a countdown task, execute the following command in your terminal:

```bash
python examples/gpt_play_countdown.py
```

Ensure that you have set the `OPENAI_KEY` environment variable or have the `OPENAI_KEY` file in the same directory as the script for authentication. This example demonstrates initializing an environment, running an agent loop, and closing the environment.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.