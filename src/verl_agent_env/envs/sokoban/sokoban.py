"""
sokoban environment adapted from:
https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/env/sokoban/env.py
https://github.com/mpSchrader/gym-sokoban/tree/default
"""

import copy
from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from verl_agent_env.envs.base import LLMAgentEnv
from verl_agent_env.envs.sokoban.room_utils import generate_room
from verl_agent_env.envs.sokoban.render_utils import room_to_rgb, room_to_tiny_world_rgb


class SokobanEnv(LLMAgentEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 room_setup=None):
        super().__init__()

        # General Configuration
        self.dim_room = tuple(dim_room)
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        # observation space and action space are initialized in super().__init__(), to follow the LLM Agent Env interface
        # screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        # self.action_space = gym.spaces.Discrete(len(ACTION_LOOKUP))
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        
        # Initialize Room
        if room_setup is not None:
            _ = self.reset(options={"room_setup": room_setup})
        else:
            _ = self.reset()
        
        self._action_space_json_schema = []
        self._tool_name_action_id_map = {}
        for action_key in ACTION_LOOKUP:
            if action_key == 0:
                # Do not add the no operation action to the action space
                continue
            func_name = ACTION_LOOKUP[action_key].replace(" ", "_")
            self._action_space_json_schema.append({
                "name": func_name,
                "description": ACTION_DESCRIPTION[action_key],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            })
            self._tool_name_action_id_map[func_name] = action_key
    
    def _get_obs(self, error_msg: Optional[str] = None) -> Tuple[dict, ...]:
        arr_walls, arr_goals, arr_boxes, arr_player = self.render(mode='raw')
        # construct the map
        # step 1, set the floor
        grid_map = np.zeros((self.dim_room[0], self.dim_room[1]), dtype=np.uint8)
        # step 2, set the walls
        grid_map[arr_walls == 1] = 1
        # step 3, set the targets
        grid_map[arr_goals == 1] = 2
        # step 4, set the boxes
        grid_map[arr_boxes == 1] = 3
        # step 5, set the boxes on target
        grid_map[(arr_boxes == 1) & (arr_goals == 1)] = 4
        # step 6, set the player
        grid_map[arr_player == 1] = 5
        # step 7, set the player on target
        grid_map[(arr_player == 1) & (arr_goals == 1)] = 6
        
        # turn the grid map into a string
        grid_map_str = ""
        for i in range(self.dim_room[0]):
            for j in range(self.dim_room[1]):
                grid_map_str += GRID_LOOKUP[grid_map[i, j]]
            grid_map_str += "\n"
        grid_map_str = grid_map_str.rstrip()
        
        if error_msg is not None:
            obs = f"There some ERROR occurred . The error message is: {error_msg}\n"
        else:
            obs = ""
        if self._last_tool_call_id is not None:
            obs += f"After the action, the map of the world is: \n{grid_map_str}\n You are at position {self.player_position[0]},{self.player_position[1]} in the world."
            return (
                {
                    "role": "tool",
                    "tool_call_id": self._last_tool_call_id,
                    "content": obs
                },
            )
        else:
            obs += f"The current map of the world is: \n{grid_map_str}\n You are at position {self.player_position[0]},{self.player_position[1]} in the world."
            return (
                {
                    "role": "user",
                    "content": obs
                },
            )
    
    def step(self, action):
        action = action['tool_calls']
        error_msg = None
        if len(action) == 0:
            self._last_tool_call_id = None
            action = 0
        else:
            if len(action) > 1:
                error_msg = f"You can only take one action at a time. But you tried to take {len(action)} actions at the same time. So we will only take the first action: {action[0]['function']}"
            self._last_tool_call_id = action[0]['id']
            action = action[0]['function']
            action = self._tool_name_action_id_map[action['name']]

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        done = self._check_if_done()

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return self._get_obs(error_msg), self.reward_last, done, False, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)
    
    def serialize_room(self):
        # Convert tuple keys in box_mapping to strings for serialization
        serialized_box_mapping = {}
        for key, value in self.box_mapping.items():
            assert isinstance(key, tuple)
            serialized_box_mapping[str(key)] = [int(cell) for cell in value]
        
        # Convert NumPy arrays to native Python lists of lists of integers
        room_fixed_list = []
        room_state_list = []
        
        for row in self.room_fixed:
            room_fixed_list.append([int(cell) for cell in row])
            
        for row in self.room_state:
            room_state_list.append([int(cell) for cell in row])
                
        return {
            "room_fixed": room_fixed_list,
            "room_state": room_state_list,
            "box_mapping": serialized_box_mapping
        }
    
    def deserialize_room(self, room_dict):
        # Convert lists of lists to NumPy arrays
        self.room_fixed = np.array(room_dict["room_fixed"], dtype=np.int8)
        self.room_state = np.array(room_dict["room_state"], dtype=np.int8)
        
        # Convert string keys back to tuples during deserialization
        deserialized_box_mapping = {}
        for key, value in room_dict["box_mapping"].items():
            assert key.startswith('(') and key.endswith(')')
            # Parse string representation of tuple back to actual tuple
            # Format is like '(1, 2)' to (1, 2)
            # Extract numbers from the string
            nums = key.strip('()').split(',')
            # Convert to integers and create tuple
            tuple_key = tuple(int(num.strip()) for num in nums)
            deserialized_box_mapping[tuple_key] = value
                
        self.box_mapping = deserialized_box_mapping
        
        assert self.room_fixed.shape == self.dim_room, f"{self.room_fixed.shape=} {self.dim_room=}"
        assert self.room_state.shape == self.dim_room, f"{self.room_state.shape=} {self.dim_room=}"
        assert len(self.box_mapping) == self.num_boxes

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        if options is not None and options.get("room_setup", None) is not None:
            self.deserialize_room(options["room_setup"])
        else:
            try:
                self.room_fixed, self.room_state, self.box_mapping = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    second_player=options.get("second_player", False) if options is not None else False
                )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                return self.reset(seed=seed, options=options)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self._last_tool_call_id = None

        return self._get_obs(), {}

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        if 'rgb_array' in mode or 'human' in mode:
            img = self.get_image(mode, scale)

            if 'rgb_array' in mode:
                return img
            elif 'human' in mode:
                if self.viewer is None:
                    self.viewer = gym.envs.classic_control.rendering.SimpleImageViewer()
                self.viewer.imshow(img)
                return self.viewer.isopen

        elif 'raw' in mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
            arr_player = (self.room_state == 5).view(np.int8)

            return arr_walls, arr_goals, arr_boxes, arr_player

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP
    
    @property
    def task_prompt(self) -> str:
        return GUIDE
    
    @property
    def action_space_json_schema(self):
        return self._action_space_json_schema

GRID_LOOKUP = {
        0: " _ \t",  # floor
        1: " # \t",  # wall
        2: " O \t",  # target
        3: " X \t",  # box
        4: " √ \t",  # box on target
        5: " P \t",  # player
        6: " S \t",  # player on target
        # Use tab separator to separate columns and \n\n to separate rows.
    }

ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}
ACTION_DESCRIPTION = {
    0: 'Do nothing and stay in the same position',
    1: 'Push the box up, if the box is adjacent to the up direction of the player and the field in front of the box is empty (floor or target without a box). The box will not move if the field is a wall or already has a box. Up direction means moving the coordinate by [0, 1]',
    2: 'Push the box down, if the box is adjacent to the down direction of the player and the field in front of the box is empty (floor or target without a box). The box will not move if the field is a wall or already has a box. Down direction means moving the coordinate by [0, -1]',
    3: 'Push the box left, if the box is adjacent to the left direction of the player and the field in front of the box is empty (floor or target without a box). The box will not move if the field is a wall or already has a box. Left direction means moving the coordinate by [-1, 0]',
    4: 'Push the box right, if the box is adjacent to the right direction of the player and the field in front of the box is empty (floor or target without a box). The box will not move if the field is a wall or already has a box. Right direction means moving the coordinate by [1, 0]',
    5: 'Move the player up, if the field is empty (floor or target without a box). The player will not move if the field is a wall or a box. Up direction means moving the coordinate by [0, 1]',
    6: 'Move the player down, if the field is empty (floor or target without a box). The player will not move if the field is a wall or a box. Down direction means moving the coordinate by [0, -1]',
    7: 'Move the player left, if the field is empty (floor or target without a box). The player will not move if the field is a wall or a box. Left direction means moving the coordinate by [-1, 0]',
    8: 'Move the player right, if the field is empty (floor or target without a box). The player will not move if the field is a wall or a box. Right direction means moving the coordinate by [1, 0]',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']

GUIDE = """
### Sokoban Puzzle Instructions

In Sokoban, your goal is to move all the boxes to the target spots on the grid. This requires careful planning and strategic moves. Here's how it works:

---

#### Symbols and Their Meaning
- **Floor (`_`)**: Open spaces where you can walk and move boxes.
- **Walls (`#`)**: These block movement. You can't move through or push anything into walls.
- **Targets (`O`)**: The spots where boxes need to go.
- **Boxes (`X`)**: These are what you need to push onto the targets.
- **Box on Target (`√`)**: A box successfully placed on a target.
- **Player (`P`)**: That's you! You'll move around the grid to push boxes.
- **Player on Target (`S`)**: You standing on a target.

---

#### Your Goal
Push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on targets, you win!

---

#### Rules to Remember
1. **You Can Only Push Boxes**: You can't pull them, so plan ahead to avoid getting stuck.
2. **No Moving Through Walls**: You can't walk through or push boxes into walls (`#`).
3. **Avoid Traps**: Don't push boxes into corners or against walls where they can't be moved again.

---

#### Controls
You can move the player by taking the function call. There are 8 actions you can take:
- Move the player up
- Move the player down
- Move the player left
- Move the player right
- Push the box up
- Push the box down
- Push the box left
- Push the box right
To actually move the player, you need to call the function with proper function name.

#### Rewards
- **Move**: Each step you take costs 0.1.
- **Push Box to Target**: Each box placed on a target gives you 1.0.
- **Achieve Goal**: When all boxes are on targets, you get a reward of 10.0.

---

#### Example Map
Here's an example of a Sokoban puzzle:

# 	 # 	 # 	 # 	 # 	 # 	 # 	 
# 	 _ 	 _ 	 # 	 # 	 # 	 # 	 
# 	 _ 	 # 	 # 	 # 	 O 	 # 	 
# 	 _ 	 _ 	 _ 	 O 	 _ 	 # 	 
# 	 _ 	 X 	 X 	 _ 	 _ 	 # 	 
# 	 _ 	 O 	 _ 	 X 	 P 	 # 	 
# 	 # 	 # 	 # 	 # 	 # 	 # 	 

Each puzzle will have a different layout, but the rules and goal remain the same.

---

#### Tips for Beginners
1. **Move Boxes Step by Step**: Push them one at a time toward the targets.
2. **Think Ahead**: Avoid pushing a box into a spot where you can't move it again.

Enjoy the challenge!
"""

if __name__ == "__main__":
    env = SokobanEnv()
    obs, info = env.reset()
    print(obs[0]["content"])
    print(info)

    action = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "move_down"}}]
    }
    obs, reward, done, truncated, info = env.step(action)
    print(obs[0]["content"])
    print(f"{reward=}")
    print(f"{done=}")
    print(f"{truncated=}")
    print(f"{info=}")
    
    action = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "move_right"}}]
    }
    obs, reward, done, truncated, info = env.step(action)
    print(obs[0]["content"])
    print(f"{reward=}")
    print(f"{done=}")
    print(f"{truncated=}")
    print(f"{info=}")
    
    action = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "move_up"}}]
    }
    obs, reward, done, truncated, info = env.step(action)
    print(obs[0]["content"])
    print(f"{reward=}")
    print(f"{done=}")
    print(f"{truncated=}")
    print(f"{info=}")
    
    action = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "move_left"}}]
    }
    obs, reward, done, truncated, info = env.step(action)
    print(obs[0]["content"])
    print(f"{reward=}")
    print(f"{done=}")
    print(f"{truncated=}")
    print(f"{info=}")