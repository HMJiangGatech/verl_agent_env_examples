# How to run verl with agent env

1. Create workspace

```bash
cd /your/workspace/path
```

2. Download code
```bash
git clone https://github.com/HMJiangGatech/verl_agent_env_examples.git
git clone https://github.com/HMJiangGatech/verl.git
```

3. Launch the docker
```bash
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
-v $(pwd):/root/code \
whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6
```

4. Install packages and download models

Do the following in the docker:
```bash
cd verl
git checkout jhaoming/agent
pip install -e .
cd ../verl_agent_env_examples
pip install -e .
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/code/models/qwen2_5-7b-instruct --local-dir-use-symlinks False
```

5. Generate dataset for verl training

```bash
cd ~/code/verl_agent_env_examples/examples/verl/sokoban
python curate_data_simply.py
```

Optionally, you can test it by:
```bash
python test_verl_dataloader.py
```

6. Log-in the docker and launch environment service

Find the container id
```bash
docker ps
```

and bash into it and launch the environment service on 8000
```bash
docker exec -it 30b6354a01ee /bin/bash
cd ~/code/verl_agent_env_examples
uvicorn src.verl_agent_env.app:app --reload
```

7. Run training script

```bash
sh run_agent_qwen2_5-7b.sh
```

Note: The scirpt is using mlflow, you need to change it and install mlflow in the docker: `pip install mlflow`