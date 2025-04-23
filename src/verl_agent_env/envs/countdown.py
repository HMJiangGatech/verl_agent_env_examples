"""
Environment for generating equations to reach a target number using given numbers and operations.
Implementation adapted from https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py
"""

import json
from typing import List, Optional, Tuple
from verl_agent_env.envs.base import LLMAgentEnv


class CountdownEnv(LLMAgentEnv):
    def __init__(self, 
                 num_operands: int = 6, 
                 max_target: int = 100, 
                 min_number: int = 1, 
                 max_number: int = 100, 
                 operations: List[str] = None
                 ) -> None:
        """Initialize the countdown environment.
        
        Args:
            num_operands (int): Number of numbers provided in each sample.
            max_target (int): Maximum value for target number.
            min_number (int): Minimum value for provided numbers.
            max_number (int): Maximum value for provided numbers.
            operations (List[str], optional): List of allowed operations, defaults to ['+', '-', '*', '/'].
        """
        super().__init__()
        self.num_operands = num_operands
        self.max_target = max_target
        self.min_number = min_number
        self.max_number = max_number
        self._operations = operations if operations is not None else ['+', '-', '*', '/']
        self._operations_str = '(' + ', '.join(self._operations) + ')'

        self._action_space_json_schema = [
            {
                "name": "test_equation",
                "description": "Submit an equation using the provided numbers and operations to reach the target number",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "equation": {
                            "type": "string",
                            "description": f"A mathematical equation using the provided numbers and operations {self._operations_str} that evaluates to the target number. Brackets are allowed. For example: '5 + 3 * 2' or '(1 + 2) / 3'"
                        }
                    },
                    "required": ["equation"]
                }
            }
        ]

        self._target_num = 0
        self._numbers = [0] * self.num_operands
        self._target_equation = ' + '.join(map(str, self._numbers))

        self._attempts = []

    def _get_obs(self) -> Tuple[dict, ...]:
        """Generate a string representation of the agent's attempts.

        Returns:
            Tuple[dict, ...]: A tuple containing a dictionary with the tool response and results.
        """
        if len(self._attempts) == 0:
            return (
                {
                    "role": "user",
                    "content": "No attempts yet. Let's start!"
                },
            )
        obs_str = ""
        attempt = self._attempts[-1]
        if attempt['result'] in ['pass', 'fail']:
            obs_str += f"Tried equation: {attempt['equation']} = {attempt['eval']}, "
            if attempt['result'] == 'pass':
                obs_str += f"which equals the target number {self._target_num} and passes the task."
            if attempt['result'] == 'fail':
                obs_str += f"which does not equal the target number {self._target_num} and fails the task."
        if attempt['result'] == 'parsing error':
            obs_str += f"Tried equation: {attempt['equation']}, which is not a well formulated equation and got parsing error: {attempt['error']}"
        obs_str.strip()
        return (
            {
                "role": "tool",
                "tool_call_id": attempt["tool_id"],
                "content": obs_str
            },
        )

    
    def _get_info(self) -> dict:
        """Retrieve information about the current state of the environment.

        Returns:
            dict: A dictionary containing attempts, target number, and target equation.
        """
        return {
            "attempts": self._attempts,
            "target_num": self._target_num,
            "target_equation": self._target_equation,
        }

    async def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to a new random state.

        Args:
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            Tuple[str, dict]: The initial observation and information about the environment.
        """
        await super().reset(seed=seed)

        # Generate random numbers between 1 and 9
        self._numbers = self.np_random.integers(1, 10, size=self.num_operands)

        # Generate a random ground truth equation by randomly combining numbers and operations
        numbers_copy = self._numbers.copy()
        self.np_random.shuffle(numbers_copy)
        numbers_copy = list(map(str, numbers_copy))

        # Keep adding brackets with 20% probability in numbers_copy
        while self.np_random.random() < 0.2:
            # find a pair of numbers
            left_index = self.np_random.choice(range(len(numbers_copy)-1))
            right_index = self.np_random.choice(range(left_index + 1, len(numbers_copy)))
            # add brackets around the pair
            numbers_copy[left_index] = f"({numbers_copy[left_index]}"
            numbers_copy[right_index] = f"{numbers_copy[right_index]})"
        
        # Start with first number
        equation = str(numbers_copy[0])
        
        # Add remaining numbers with random operations
        for i in range(1, len(numbers_copy)):
            operation = self.np_random.choice(self._operations)
            equation = f"{equation} {operation} {numbers_copy[i]}"
        
        # Calculate target number by evaluating the equation
        self._target_num = eval(equation)
        self._target_equation = equation

        # Clear previous attempts
        self._attempts = []

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    async def step(self, action):
        action = action['tool_calls']
        if len(action) == 0:
            return [], 0.0, True, False, self._get_info()
        
        assert len(action) == 1, "Only one action is allowed"
        action = action[0]
        assert action["type"] == "function", "Only function call is allowed"
        tool_id = action["id"]
        action = action["function"]
        assert action["name"] == "test_equation", "Only test_equation is allowed"
        action = action["arguments"]
        action = json.loads(action)

        try:
            # Parse the equation from the action
            equation = action["equation"]

            # Evaluate the equation
            equation_result = eval(equation)
            result = "pass" if equation_result == self._target_num else "fail"
            reward = 1.0 if result == "pass" else 0.0
            terminated = result == "pass"
            truncated = False

            self._attempts.append({
                "tool_id": tool_id,
                "equation": equation,
                "eval": equation_result,
                "result": result
            })

            return self._get_obs(), reward, terminated, truncated, self._get_info()

        except Exception as e:
            # Invalid equation format or division by zero
            self._attempts.append({
                "tool_id": tool_id,
                "equation": equation,
                "result": "parsing error",
                "error": str(e)
            })
            
            return self._get_obs(), 0.0, False, False, self._get_info()
    
    @property
    def task_prompt(self) -> str:
        """Returns the task prompt describing what the agent needs to do."""
        prompt = f"Your task is to find an equation using {self.num_operands} numbers and arithmetic operators {self._operations_str} "
        prompt += f"that equals {self._target_num}.\n"
        prompt += f"The numbers you must use are: {', '.join(map(str, self._numbers))}\n"
        prompt += "You must use each number exactly once. Brackets are allowed.\n"
        prompt += "Provide your answer as a valid mathematical equation."
        return prompt

    @property
    def action_space_json_schema(self):
        """Returns the JSON schema for the action space."""
        return self._action_space_json_schema