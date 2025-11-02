set -x

ts=$(date '+%Y%m%d_%H%M%S') 
model_name=Qwen2.5-3B-Instruct
n_outer_iterations=3
soft_label_alpha=0.5
use_dynamic_alpha=false
soft_label_sample_ratio=0.0
use_migiting_forget=true
experiment_name=iGSM-sft-$model_name-use_migiting_forget_$use_migiting_forget-3.0
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
    trainer.project_name=iGSM-sft \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=5 \
    trainer.logger='["console", "swanlab"]' \
    trainer.default_local_dir=$save_path \
    trainer.save_freq=20 \
    trainer.checkpoint.save_contents=["hf_model"]\
    optim.lr=1e-6 \
    +use_migiting_forget=$use_migiting_forget \
    +n_outer_iterations=$n_outer_iterations\
    +soft_label_alpha=$soft_label_alpha\
    +use_dynamic_alpha=$use_dynamic_alpha\
    +soft_label_sample_ratio=$soft_label_sample_ratio