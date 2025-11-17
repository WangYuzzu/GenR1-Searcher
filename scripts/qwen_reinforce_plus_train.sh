export NUMEXPR_MAX_THREADS=128
export RAY_DEDUP_LOGS=0

# Your wandb token
wandb_token=xxx
export WANDB_API_KEY=$wandb_token

# Path of training data
DATA_PATH=/root/autodl-tmp/GenR1-Searcher/data/training_set/stage_2.jsonl

TOKENIZER_PATH=/root/autodl-tmp/Qwen-2.5-3B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
N_SAMPLES=16
EPISODE=100
LR=2e-6
MAX_LENGTH=29000
PORT=1278  # 远端奖励服务器端口（启动 reward_server.py 时用的）
TEMP=1.0
WARMUP=0.0
TBS=36
RBS=36
KL=0.0


SAVE_MODEL_NAME=qwen_reinfoce_plus
LOG_BASE=results/logs
WANDB_RUN_NAME=3b_7b-retrieve_stage3-retrieve

GROUP_METHOD=normal



mkdir -p results/ckpts/$SAVE_MODEL_NAME
mkdir -p $LOG_BASE

ray job submit --address="http://127.0.0.1:8267" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 6 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --remote_rm_url http://localhost:${PORT}/get_reward \
   --save_path /root/autodl-tmp/R1-Searcher/results/ckpts/qwen_reinfoce_plus \
   --pretrain ${TOKENIZER_PATH} \
   --load_checkpoint \
   --ckpt_path /root/autodl-tmp/R1-Searcher/ckpt/checkpoints_ppo_ray \
   --micro_train_batch_size 1 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator reinforce_baseline \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len 1024 \
   --generate_max_len $MAX_LENGTH \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate $LR \
   --critic_learning_rate 9e-6 \
   --init_kl_coef $KL \
   --prompt_data $DATA_PATH \
   --input_key question \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 25 \
   --vllm_sync_backend nccl \
   --max_ckpt_num 8 \
   --temperature $TEMP \
   --overlap_comm \
   --packing_samples \
   --wandb_run_name $WANDB_RUN_NAME \
   --use_wandb True \
   # --apply_chat_template \

