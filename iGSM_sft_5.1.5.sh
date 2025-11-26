set -x

ts=$(date '+%Y%m%d_%H%M%S') 
model_name=Qwen2.5-3B-Instruct
n_outer=30
fkl_weight=0.5
q_weight=0.4
hard_ce_weight=0.001
q_lr=2e-5
p_lr=5e-6
dynamic_q_weight=true

experiment_name=iGSM-iterative-sft-$model_name-n_outer_${n_outer}-fkl_weight_${fkl_weight}-q_weight_${q_weight}-hard_ce_weight_${hard_ce_weight}-p_lr_${p_lr}-q_lr_${q_lr}-dynamic_q_weight_${dynamic_q_weight}-$ts
save_path=/gemini/space/guanming/sft/models/$experiment_name

# 确保输出目录存在
mkdir -p $save_path

# 启动训练
# 注意：
# 1. trainer.strategy=iterative 启用迭代式训练
# 2. trainer.n_outer 控制外层循环次数
# 3. optim_q 和 optim_p 分别控制 q 模型和 p 模型的优化器参数
# 4. model.partial_pretrain 指定初始模型路径
    # data.micro_batch_size=64 \
    # model.strategy=fsdp \
    # model.partial_pretrain=/gemini/space/model_zoo/$model_name \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=4 -m \
    verl.trainer.fsdp_sft_trainer \
    data.train_files=/gemini/space/guanming/sft/datasets/iGSM/verl_sft/train_16.parquet \
    data.val_files=/gemini/space/guanming/sft/datasets/iGSM/verl_sft/test_16.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.micro_batch_size_per_gpu=10 \
    data.train_batch_size=400 \
    data.max_length=2048 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    model.partial_pretrain=/gemini/space/guanming/sft/models/iGSM-iterative-sft-Qwen2.5-3B-Instruct-n_outer_30-fkl_weight_0.5-q_weight_0.3-hard_ce_weight_0.0-p_lr_1e-5-q_lr_2e-5-dynamic_q_weight_true-20251126_113901/iter_4_p/global_step_60/huggingface \
    model.fsdp_config.model_dtype=bf16 \
    trainer.project_name=iGSM-iterative-sft \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=3 \
    trainer.logger='["console", "swanlab"]' \
    trainer.default_local_dir=$save_path \
    +trainer.debug_fixed_q_path=/gemini/space/guanming/sft/models/iGSM-iterative-sft-Qwen2.5-3B-Instruct-n_outer_30-20251125_235707/iter_0_q/global_step_250/huggingface \
    trainer.save_freq=20 \
    trainer.test_freq=100 \
    trainer.checkpoint.save_contents=["hf_model"] \
    +trainer.strategy=iterative \
    +trainer.fkl_weight=$fkl_weight \
    +trainer.q_weight=$q_weight \
    +trainer.hard_ce_weight=$hard_ce_weight \
    +trainer.dynamic_q_weight=$dynamic_q_weight \
    +trainer.n_outer=$n_outer \
    +optim_q.lr=$q_lr \
    +optim_p.lr=$p_lr \
    optim.lr=1e-5
