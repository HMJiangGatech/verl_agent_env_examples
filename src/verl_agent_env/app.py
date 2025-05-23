from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from verl_agent_env.interface import initialize_environment, close_environment, action_space_json_schema, get_task_prompt, tools_json_schema_openai, take_step, reset_environment, allow_parallel_tool_call

app = FastAPI()

class InitializeRequest(BaseModel):
    env_name: str
    seed: Optional[int] = None
    env_kwargs: Optional[Dict[str, Any]] = None

class EnvironmentResponse(BaseModel):
    message: str
    env_id: str = None
    observation: Any = None
    info: Dict[str, Any] = None

class ActionSpaceResponse(BaseModel):
    action_space: Dict[str, Any]

class TaskPromptResponse(BaseModel):
    task_prompt: str

class AllowParallelToolCallResponse(BaseModel):
    allow_parallel_tool_call: bool

class OpenAIToolsSchemaResponse(BaseModel):
    tools_schema: List[Dict[str, Any]]

class StepRequest(BaseModel):
    action: Any

class StepResponse(BaseModel):
    observation: Any
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None

class ResetEnvironmentResponse(BaseModel):
    observation: Any = None
    info: Dict[str, Any] = None

@app.post("/api/environment/initialize", response_model=EnvironmentResponse)
async def initialize_env(request: InitializeRequest):
    return await initialize_environment(request.env_name, request.seed, request.env_kwargs)

@app.post("/api/environment/{env_id}/close", response_model=EnvironmentResponse)
async def close_env(env_id: str):
    return await close_environment(env_id)

@app.get("/api/environment/{env_id}/action-space", response_model=ActionSpaceResponse)
async def get_action_space(env_id: str):
    try:
        action_space = action_space_json_schema(env_id)
        return action_space
    except KeyError as e:
        return {"message": str(e)}

@app.get("/api/environment/{env_id}/task-prompt", response_model=TaskPromptResponse)
async def get_task_prompt_endpoint(env_id: str):
    try:
        task_prompt = get_task_prompt(env_id)
        return {"task_prompt": task_prompt}
    except KeyError as e:
        return {"message": str(e)}

@app.get("/api/environment/{env_id}/allow-parallel-tool-call", response_model=AllowParallelToolCallResponse)
async def get_allow_parallel_tool_call_endpoint(env_id: str):
    try:
        parallel_tool_call = allow_parallel_tool_call(env_id)
        return {"allow_parallel_tool_call": parallel_tool_call}
    except KeyError as e:
        return {"message": str(e)}

@app.get("/api/environment/{env_id}/tools-schema-openai", response_model=OpenAIToolsSchemaResponse)
async def get_openai_tools_schema(env_id: str):
    try:
        tools_schema = tools_json_schema_openai(env_id)
        return {"tools_schema": tools_schema}
    except KeyError as e:
        return {"message": str(e)}

@app.post("/api/environment/{env_id}/step", response_model=StepResponse)
async def take_step_endpoint(env_id: str, request: StepRequest):
    try:
        result = await take_step(env_id, request.action)
        return result
    except KeyError as e:
        return {"message": str(e)}

@app.post("/api/environment/{env_id}/reset", response_model=ResetEnvironmentResponse)
async def reset_env(env_id: str, request: ResetRequest):
    return await reset_environment(env_id, request.seed, request.options)

@app.get("/")
async def health_check():
    return {"status": "running"}