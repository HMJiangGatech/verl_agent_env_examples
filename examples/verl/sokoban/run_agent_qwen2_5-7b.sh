set -x
export MLFLOW_TRACKING_URI=https://prod.us-east-1.internal.mlflow.nile.amazon.dev
python3 -m verl.trainer.main_agent_ppo \
    algorithm.adv_estimator=gae \
    env.environment_endpoint=http://localhost:8000 \
    env.max_turn=10 \
    data.agent_prompt_style=qwen2_5 \
    data.train_files=$HOME/code/verl_agent_env_examples/examples/verl/sokoban/data/simple_train.parquet \
    data.val_files=$HOME/code/verl_agent_env_examples/examples/verl/sokoban/data/simple_test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=15360 \
    data.max_response_length=128 \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HOME/code/models/qwen2_5-7b-instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$HOME/code/models/qwen2_5-7b-instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','mlflow'] \
    trainer.project_name='verl_agent_env_examples' \
    trainer.experiment_name='sokoban_qwen2_5-7b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@
