import pandas as pd
import os

# create agent env data
num_train = 100000
num_test = 250

train_df = [
    {
        "env_name": 'verl_env/sokoban-v0',
        "seed": i,
        "env_kwargs": None
    } for i in range(num_train)
]

test_df = [
    {
        "env_name": 'verl_env/sokoban-v0',
        "seed": i + num_train,
        "env_kwargs": None
    } for i in range(num_test)
]

train_df = pd.DataFrame(train_df)
test_df = pd.DataFrame(test_df)

os.makedirs('data', exist_ok=True)

train_df.to_parquet('data/train.parquet')
test_df.to_parquet('data/test.parquet')

print("Example of train data:")
print(train_df.head()) 

print("Example of test data:")
print(test_df.head()) 