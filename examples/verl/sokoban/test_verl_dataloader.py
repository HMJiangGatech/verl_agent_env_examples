import os
import unittest
from verl.utils.dataset.rl_agent_dataset import RLAgentDataset
from transformers import AutoTokenizer

class TestRLAgentDataset(unittest.TestCase):
    def setUp(self):
        # Ensure data is generated
        self.train_file = 'data/train.parquet'
        self.test_file = 'data/test.parquet'
        self.environment_endpoint = 'http://localhost:8000'  # Example endpoint
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
        self.agent_prompt_style = 'qwen2_5'
        self.max_prompt_length = 4096
        
    def test_dataset_initialization(self):
        # Initialize the dataset
        dataset = RLAgentDataset(
            environment_endpoint=self.environment_endpoint,
            parquet_files=[self.train_file, self.test_file],
            tokenizer=self.tokenizer,
            agent_prompt_style=self.agent_prompt_style,
            max_prompt_length=self.max_prompt_length
        )

        # Check the length of the dataset
        self.assertEqual(len(dataset), 10100)  # 10000 train + 100 test

        # Check if the first item is loaded correctly
        for i in [0,10,20,-20]:
            print(f"Testing item {i}")
            first_item = dataset[i]
            self.assertIn('input_ids', first_item)
            self.assertIn('attention_mask', first_item)
            self.assertIn('position_ids', first_item)
            self.assertIn('raw_prompt', first_item)
            for message in first_item['raw_prompt']:
                print(f"Role: \n{message['role']}, Content: \n{message['content']}")

if __name__ == '__main__':
    unittest.main()
