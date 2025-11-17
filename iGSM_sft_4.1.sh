set -x

ts=$(date '+%Y%m%d_%H%M%S') 
model_name=Qwen2.5-3B-Instruct
# model_name=Qwen3-0.6B
    # model.strategy=fsdp\
n_samples=4
temperature=1.0
use_simpo=true
min_margin=0.1
beta=0.1
gamma=0.5
experiment_name=iGSM-sft4.1-$model_name-use_simpo_$use_simpo-n_samples_${n_samples}-temperature_${temperature}-min_margin_${min_margin}-beta_${beta}-gamma_${gamma}-$ts
save_path=/root/autodl-fs/models/$experiment_name

torchrun --nproc_per_node=1 -m \
    verl.trainer.fsdp_sft_trainer \
    data.train_files=/root/autodl-fs/datasets/iGSM/verl_sft/train_16.parquet \
    data.val_files=/root/autodl-fs/datasets/iGSM/verl_sft/test_16.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=2048 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=8 \
    data.train_batch_size=20 \
    model.partial_pretrain=/root/autodl-fs/models/$model_name \
    model.fsdp_config.model_dtype=bf16 \
    trainer.project_name=iGSM-sft \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=5 \
    trainer.logger='["console", "swanlab"]' \
    trainer.default_local_dir=$save_path \
    trainer.save_freq=50 \
    trainer.checkpoint.save_contents=["hf_model"]\
    optim.lr=1e-6 \
    +use_simpo=$use_simpo \
    +simpo.n_samples=$n_samples\
    +simpo.temperature=$temperature\
    +simpo.min_margin=$min_margin\
    +simpo.beta=$beta\
    +simpo.gamma=$gamma
