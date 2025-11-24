set -x

ts=$(date '+%Y%m%d_%H%M%S') 
model_name=Qwen2.5-3B-Instruct
n_outer=3

experiment_name=iGSM-iterative-sft-$model_name-n_outer_${n_outer}-$ts
save_path=/mnt/local3/wxy/models/$experiment_name

# 确保输出目录存在
mkdir -p $save_path

# 启动训练
# 注意：
# 1. trainer.strategy=iterative 启用迭代式训练
# 2. trainer.n_outer 控制外层循环次数
# 3. optim_q 和 optim_p 分别控制 q 模型和 p 模型的优化器参数
# 4. model.partial_pretrain 指定初始模型路径

CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 -m \
    verl.trainer.fsdp_sft_trainer \
    data.train_files=/mnt/local2/wxy/sft_project/datasets/iGSM/verl_sft/train_16.parquet \
    data.val_files=/mnt/local2/wxy/sft_project/datasets/iGSM/verl_sft/test_16.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.micro_batch_size_per_gpu=2 \
    data.micro_batch_size=32 \
    data.train_batch_size=512 \
    data.max_length=2048 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    model.partial_pretrain=/mnt/local/wxy/models/$model_name \
    model.fsdp_config.model_dtype=bf16 \
    trainer.project_name=iGSM-iterative-sft \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=5 \
    trainer.logger='["console", "swanlab"]' \
    trainer.default_local_dir=$save_path \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    model.strategy=fsdp \
    trainer.checkpoint.save_contents=["hf_model"] \
    +trainer.strategy=iterative \
    +trainer.n_outer=$n_outer \
    +optim_q.lr=2e-5 \
    +optim_p.lr=1e-6 \
    optim.lr=1e-5
