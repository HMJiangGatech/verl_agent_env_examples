import pandas as pd
import os
import json
from tqdm import tqdm
from verl_agent_env.envs.sokoban.sokoban import SokobanEnv

import ray

ray.init(num_cpus=75)

# create agent env data
num_train = 10000
num_test = 100

@ray.remote
def create_env_data(seed, offset=0):
    env = SokobanEnv(
        dim_room=(5, 5),
        num_boxes=1
    )
    env.reset(seed=seed + offset)
    room_setup = env.serialize_room()
    return {
        "env_name": 'verl_env/sokoban-v0',
        "seed": seed + offset,
        "env_kwargs": json.dumps(
            {
                'room_setup': room_setup,
                'dim_room': [5, 5],
                'num_boxes': 1
            }
        )
    }

# Parallelize train data creation
print("Creating training data...")
train_futures = [create_env_data.remote(i) for i in range(num_train)]
train_data = []
for batch in tqdm(range(0, len(train_futures), 100)):
    batch_results = ray.get(train_futures[batch:batch+100])
    train_data.extend(batch_results)
train_df = pd.DataFrame(train_data)

# Parallelize test data creation
print("Creating test data...")
test_futures = [create_env_data.remote(i, offset=num_train) for i in range(num_test)]
test_data = ray.get(test_futures)
test_df = pd.DataFrame(test_data)

os.makedirs('data', exist_ok=True)

train_df.to_parquet('data/simple_train.parquet')
test_df.to_parquet('data/simple_test.parquet')

print("Example of train data:")
print(train_df.head()) 

print("Example of test data:")
print(test_df.head()) 