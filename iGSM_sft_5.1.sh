set -x

ts=$(date '+%Y%m%d_%H%M%S') 
model_name=Qwen2.5-3B-Instruct
# model_name=Qwen3-0.6B
n_outer_iterations=10
soft_label_alpha=0.5
anti_forgetting=true
experiment_name=iGSM-sft-$model_name-anti_forgetting_$anti_forgetting-3.0-n_outer_iterations_${n_outer_iterations}-soft_label_alpha_${soft_label_alpha}-$ts
save_path=/mnt/local3/wxy/models/$experiment_name

CUDA_VISIBLE_DEVICES=0,1 torchrun  --nproc_per_node=2 -m \
    verl.trainer.fsdp_sft_trainer \
    data.train_files=/mnt/local2/wxy/sft_project/datasets/iGSM/verl_sft/train_16.parquet \
    data.val_files=/mnt/local2/wxy/sft_project/datasets/iGSM/verl_sft/test_16.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=2048 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=4 \
    model.partial_pretrain=/mnt/local/wxy/models/$model_name \
    model.fsdp_config.model_dtype=bf16 \
    trainer.project_name=iGSM-sft \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=5 \
    trainer.logger='["console", "swanlab"]' \
    trainer.default_local_dir=$save_path \
    trainer.save_freq=20 \
    model.strategy=fsdp\
    trainer.checkpoint.save_contents=["hf_model"]\
    optim.lr=1e-6 \
    +trainer.anti_forgetting=$anti_forgetting \
    +trainer.n_outer_iterations=$n_outer_iterations \
    +trainer.soft_label_alpha=$soft_label_alpha \
    +trainer.gold_sample_ratio=0.5 \
    +trainer.use_dynamic_alpha=true