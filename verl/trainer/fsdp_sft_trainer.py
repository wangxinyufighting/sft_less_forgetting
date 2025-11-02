# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import time
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.config.optimizer import build_optimizer
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
import torch.nn.functional as F


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)

        self.lora = self.config.model.get("lora_adapter_path") is not None or self.config.model.lora_rank > 0

        # Initialize resume-related variables
        self.resume_global_step = 0

        # build model
        self._build_model_optimizer()

        # Initialize checkpoint manager
        self._init_checkpoint_manager()

        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = self.config.trainer.device

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.lora:
                self.model.enable_input_require_grads()

                lora_adapter_path = self.config.model.get("lora_adapter_path")
                if lora_adapter_path is not None:
                    from peft import PeftModel

                    print(f"Loading pre-trained LoRA adapter for sft from: {lora_adapter_path}")

                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.use_shm)

                    self.model = PeftModel.from_pretrained(self.model, local_adapter_path, is_trainable=True)
                    peft_config = self.model.peft_config["default"]
                    # Ensure task_type is TaskType enum, not string
                    if isinstance(peft_config.task_type, str):
                        peft_config.task_type = TaskType.CAUSAL_LM
                else:
                    # Convert config to regular Python types before creating PEFT model
                    lora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "bias": "none",
                    }
                    self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.lora,
        )

        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = build_optimizer(self.fsdp_model.parameters(), self.config.optim)

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.lr_warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, 1:].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            loss = loss / n_micro_batches  # normalize loss

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        start_time = time.time()

        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch, n_micro_batches=n_micro_batches)
            step_loss += loss.item()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)

        # compute time spent per step
        end_time = time.time()
        spend_time_per_step = end_time - start_time

        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
        return {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/time(s)": spend_time_per_step,
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # Use checkpoint manager to save
        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")

            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

            # Update latest checkpoint tracker (atomic write)
            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager with proper configuration"""
        # Get checkpoint configuration from config, with defaults
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # Set default values if not specified
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # Create checkpoint config dict
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)

        # Initialize checkpoint manager
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint"""
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        train_time = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                )
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)
                train_time += metric["train/time(s)"]
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                            self.device_name
                        )
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(
        config.data.train_files, config.data, tokenizer, max_samples=config.data.get("train_max_samples", -1)
    )
    val_dataset = create_sft_dataset(
        config.data.val_files, config.data, tokenizer, max_samples=config.data.get("val_max_samples", -1)
    )

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

    destroy_global_process_group()


class AntiForgettingSFTTrainer(FSDPSFTTrainer):
    """
    Extended FSDP SFT Trainer with anti-forgetting capability.
    
    Key differences from base trainer:
    - Maintains a frozen reference model (p0)
    - Iterates through outer loops of: SFT → soft label construction → soft label training
    - Uses geometric averaging of p0 and q distributions for soft labels
    """
    
    def __init__(self, config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset):
        # Anti-forgetting specific config (set before super().__init__)
        self.n_outer_iterations = getattr(config.trainer, 'n_outer_iterations', 3)
        self.soft_label_alpha = getattr(config.trainer, 'soft_label_alpha', 0.5)
        self.use_dynamic_alpha = getattr(config.trainer, 'use_dynamic_alpha', False)
        self.soft_label_sample_ratio = getattr(config.trainer, 'soft_label_sample_ratio', 0.8)
        
        # We need to save the initial state before any training
        self.save_reference_state = True
        
        super().__init__(config, device_mesh, ulysses_device_mesh, tokenizer, train_dataset, val_dataset)
        
        # After parent init, save the initial model state as p0
        self._save_initial_reference_state()
        
        if self.device_mesh.get_rank() == 0:
            print(f"Anti-Forgetting SFT Configuration:")
            print(f"  - Outer iterations: {self.n_outer_iterations}")
            print(f"  - Soft label alpha: {self.soft_label_alpha}")
            print(f"  - Use dynamic alpha: {self.use_dynamic_alpha}")
            print(f"  - Gold/Generated sample ratio: {self.soft_label_sample_ratio:.2f}")
    
    def _save_initial_reference_state(self):
        """
        Save the initial model state as reference (p0).
        We store it as a state_dict that can be loaded later for computing logits.
        """
        if self.device_mesh.get_rank() == 0:
            print("Saving initial model state as reference (p0)...")
        
        # Check if using FSDP1 or FSDP2
        if self.config.model.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            
            # Save full state dict on rank 0
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
                self.p0_state_dict = self.fsdp_model.state_dict()
            
            # On other ranks, set to None (will be broadcasted when needed)
            if self.device_mesh.get_rank() != 0:
                self.p0_state_dict = None
                
        elif self.config.model.strategy == "fsdp2":
            # For FSDP2, use the built-in state_dict methods
            if self.device_mesh.get_rank() == 0:
                self.p0_state_dict = self.fsdp_model.state_dict()
            else:
                self.p0_state_dict = None
        else:
            raise NotImplementedError(f"Strategy {self.config.model.strategy} not implemented")
        
        if self.device_mesh.get_rank() == 0:
            print("Initial reference state (p0) saved successfully.")
    
    def _load_reference_model_for_inference(self, target_model):
        """
        Load p0 state into a model for inference.
        This is called when we need to compute logits from p0.
        
        Args:
            target_model: The FSDP model to load p0 state into
        """
        if self.config.model.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            
            load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(target_model, StateDictType.FULL_STATE_DICT, load_policy):
                if self.device_mesh.get_rank() == 0:
                    target_model.load_state_dict(self.p0_state_dict)
        elif self.config.model.strategy == "fsdp2":
            # For FSDP2
            if self.device_mesh.get_rank() == 0:
                target_model.load_state_dict(self.p0_state_dict)
        else:
            raise NotImplementedError(f"Strategy {self.config.model.strategy} not implemented")
        
        # Sync across all ranks
        torch.distributed.barrier()
        target_model.eval()
    
    def _init_reference_model(self):
        """This method is replaced by _save_initial_reference_state"""
        pass
    
    def _create_temp_sft_model(self):
        """
        Create a temporary model q for SFT exploration.
        
        For FSDP models, we need to:
        1. Save current model state
        2. Create a new model instance with the same architecture
        3. Load the saved state into the new model
        4. Wrap it with FSDP
        """
        if self.device_mesh.get_rank() == 0:
            print("Creating temporary SFT model (q) from current policy (p)...")
        
        from verl.utils.fsdp_utils import get_fsdp_wrap_policy
        from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
        from verl.utils.device import get_device_id
        
        # Step 1: Get state dict from current model
        if self.config.model.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.fsdp_model.state_dict()
        elif self.config.model.strategy == "fsdp2":
            if self.device_mesh.get_rank() == 0:
                state_dict = self.fsdp_model.state_dict()
            else:
                state_dict = None
        else:
            raise NotImplementedError(f"Strategy {self.config.model.strategy} not implemented")
        
        # Step 2: Create a new model instance with same architecture
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        from verl.utils.torch_dtypes import PrecisionType
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        # Get initialization context
        from verl.utils.fsdp_utils import get_init_weight_context_manager
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.tie_word_embeddings, 
            mesh=self.device_mesh
        )
        
        from transformers import AutoModelForCausalLM
        
        with init_context():
            temp_model = AutoModelForCausalLM.from_config(
                self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
            
            # Apply same modifications as original model
            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=temp_model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)
            
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=temp_model)
            
            if self.lora:
                temp_model.enable_input_require_grads()
                # Note: LoRA setup would need special handling
                # For now, assuming we're working with full model
        
        if self.config.model.enable_gradient_checkpointing:
            temp_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # Step 3: Wrap with FSDP
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )
        
        auto_wrap_policy = get_fsdp_wrap_policy(
            temp_model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.lora,
        )
        
        cpu_offload = None
        if self.config.model.fsdp_config.cpu_offload:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)
        
        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            from verl.utils.fsdp_utils import init_fn
            
            temp_fsdp_model = FSDP(
                temp_model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
            
            # Step 4: Load state dict into new FSDP model
            if self.device_mesh.get_rank() == 0:
                print("Loading state dict into temporary model...")
            
            load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(temp_fsdp_model, StateDictType.FULL_STATE_DICT, load_policy):
                if self.device_mesh.get_rank() == 0:
                    temp_fsdp_model.load_state_dict(state_dict)
                    
        elif fsdp_strategy == "fsdp2":
            from verl.utils.fsdp_utils import apply_fsdp2, CPUOffloadPolicy, MixedPrecisionPolicy, fsdp2_load_full_state_dict
            
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )
            
            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            
            # Save full state before applying FSDP2
            full_state = temp_model.state_dict() if state_dict is None else state_dict
            
            apply_fsdp2(temp_model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(temp_model, full_state, self.device_mesh, cpu_offload)
            temp_fsdp_model = temp_model
        else:
            raise NotImplementedError(f"Strategy {fsdp_strategy} not implemented")
        
        # Sync across all ranks
        torch.distributed.barrier()
        
        # Create optimizer for temp model
        temp_optimizer = self._build_temp_optimizer(temp_fsdp_model)
        
        if self.device_mesh.get_rank() == 0:
            print("Temporary model (q) created successfully.")
        
        return temp_fsdp_model, temp_optimizer
    
    def _build_temp_optimizer(self, model):
        """Build optimizer for temporary model"""
        from verl.workers.config.optimizer import build_optimizer
        return build_optimizer(model.parameters(), self.config.optim)
    
    def _sample_sequence(self, model, input_ids, attention_mask, max_length=None):
        """Sample a sequence from the model (for exploration)"""
        model.eval()
        
        if max_length is None:
            max_length = self.config.data.max_length
        
        with torch.no_grad():
            # Simple greedy sampling (can be extended to other strategies)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=False,  # Greedy for stability
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        return generated
    
    def _get_token_logits(self, model, input_ids, attention_mask, position_ids):
        """
        Get logits for each token position from a model.
        
        Args:
            model: Can be 'p0' (string) or an FSDP model instance
            input_ids, attention_mask, position_ids: Input tensors
        
        Returns:
            logits tensor
        """
        if isinstance(model, str) and model == 'p0':
            # Special case: need to temporarily load p0 state
            # We'll reuse temp_model for this to avoid creating another FSDP instance
            # This should be called within _construct_soft_labels where temp_model is available
            raise ValueError("Use _get_p0_logits() method instead for p0 reference")
        
        model.eval()
        
        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
            logits = output.logits
        
        return logits
    
    def _get_p0_and_q_logits(self, temp_model, input_ids, attention_mask, position_ids):
        """
        Get logits from both p0 (reference) and q (temp_model).
        
        To avoid creating multiple FSDP models, we:
        1. Save current temp_model state
        2. Load p0 state into temp_model
        3. Get p0 logits
        4. Restore temp_model state
        5. Get q logits
        
        Args:
            temp_model: The temporary FSDP model (q)
            input_ids, attention_mask, position_ids: Input tensors
        
        Returns:
            (logits_p0, logits_q): Tuple of logit tensors
        """
        if self.config.model.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            
            # Step 1: Save current q state
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(temp_model, StateDictType.FULL_STATE_DICT, save_policy):
                q_state_dict = temp_model.state_dict()
            
            # Step 2: Load p0 state and get logits
            self._load_reference_model_for_inference(temp_model)
            logits_p0 = self._get_token_logits(temp_model, input_ids, attention_mask, position_ids)
            
            # Step 3: Restore q state
            load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(temp_model, StateDictType.FULL_STATE_DICT, load_policy):
                if self.device_mesh.get_rank() == 0:
                    temp_model.load_state_dict(q_state_dict)
            torch.distributed.barrier()
            
        elif self.config.model.strategy == "fsdp2":
            # Step 1: Save current q state
            if self.device_mesh.get_rank() == 0:
                q_state_dict = temp_model.state_dict()
            else:
                q_state_dict = None
            
            # Step 2: Load p0 state and get logits
            self._load_reference_model_for_inference(temp_model)
            logits_p0 = self._get_token_logits(temp_model, input_ids, attention_mask, position_ids)
            
            # Step 3: Restore q state
            if self.device_mesh.get_rank() == 0:
                temp_model.load_state_dict(q_state_dict)
            torch.distributed.barrier()
        else:
            raise NotImplementedError(f"Strategy {self.config.model.strategy} not implemented")
        
        # Step 4: Get q logits
        logits_q = self._get_token_logits(temp_model, input_ids, attention_mask, position_ids)
        
        return logits_p0, logits_q
    
    def _construct_soft_labels(self, batch, temp_model, outer_iter):
        """
        Construct soft labels by geometric averaging p0 and q distributions.
        
        Args:
            batch: Current batch with gold data
            temp_model: Temporary SFT model q
            outer_iter: Current outer iteration (for dynamic alpha)
        
        Returns:
            Batch with soft labels added
        """
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        
        # Decide whether to use gold answer or sample from q
        use_gold = torch.rand(1).item() < self.soft_label_sample_ratio
        
        if use_gold:
            # Use gold answer sequence
            sequence_ids = input_ids
        else:
            # Sample from q (exploration)
            sequence_ids = self._sample_sequence(temp_model, input_ids, attention_mask)
            # Ensure same shape as input_ids
            if sequence_ids.shape[1] > input_ids.shape[1]:
                sequence_ids = sequence_ids[:, :input_ids.shape[1]]
            elif sequence_ids.shape[1] < input_ids.shape[1]:
                # Pad if needed
                pad_length = input_ids.shape[1] - sequence_ids.shape[1]
                sequence_ids = F.pad(sequence_ids, (0, pad_length), value=self.tokenizer.pad_token_id)
        
        # Get logits from both p0 and q
        # This method handles the complexity of loading/unloading p0 state
        logits_p0, logits_q = self._get_p0_and_q_logits(
            temp_model, sequence_ids, attention_mask, position_ids
        )
        
        # Compute soft labels via geometric averaging
        # log p* = (1-alpha) * log p0 + alpha * log q
        log_p0 = F.log_softmax(logits_p0, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        
        # Dynamic alpha: gradually increase q's weight
        if self.use_dynamic_alpha:
            alpha = self.soft_label_alpha + 0.1 * outer_iter / self.n_outer_iterations
            alpha = min(alpha, 0.9)  # Cap at 0.9
        else:
            alpha = self.soft_label_alpha
        
        log_p_star = (1 - alpha) * log_p0 + alpha * log_q
        soft_labels = F.softmax(log_p_star, dim=-1)
        
        # Add soft labels to batch
        batch["soft_labels"] = soft_labels
        batch["sequence_ids"] = sequence_ids
        batch["use_gold"] = use_gold
        
        return batch
    
    def _compute_soft_label_loss(self, batch):
        """
        Compute loss using soft labels (KL divergence).
        
        This replaces the standard cross-entropy loss with KL divergence
        between model predictions and soft label distributions.
        """
        sequence_ids = batch["sequence_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        soft_labels = batch["soft_labels"].to(self.device_name)
        loss_mask = batch["loss_mask"][:, 1:].reshape(-1).to(self.device_name)
        
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = self.fsdp_model(
                input_ids=sequence_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
            logits = output.logits
            
            # Prepare for loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_soft_labels = soft_labels[..., 1:, :].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_soft_labels = shift_soft_labels.view(-1, self.model.config.vocab_size)
            
            # Compute KL divergence: KL(soft_labels || model_predictions)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            loss = F.kl_div(log_probs, shift_soft_labels, reduction='none')
            loss = loss.sum(dim=-1)  # Sum over vocabulary
            
            # Apply loss mask
            loss = loss * loss_mask
            
            valid_tokens = torch.sum(loss_mask)
            
            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_tokens)
                dp_size = self.device_mesh.size(0)
            else:
                dp_size = 1
            
            loss = torch.sum(loss) / (valid_tokens + 1e-8) * dp_size
        
        return loss
    
    def training_step_stage1(self, batch, temp_model, temp_optimizer):
        """
        Stage 1: Train on original data D to get q.
        This is similar to standard SFT but on temporary model.
        """
        temp_model.train()
        temp_optimizer.zero_grad()
        
        # Use parent's loss computation method but with temp_model
        # We need to temporarily swap models
        original_model = self.fsdp_model
        self.fsdp_model = temp_model
        
        loss = self._compute_loss_and_backward(batch, do_backward=True, n_micro_batches=1)
        
        # Restore original model
        self.fsdp_model = original_model
        
        # Clip gradients and step
        if self.config.model.strategy == "fsdp":
            temp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
            fsdp2_clip_grad_norm_(temp_model.parameters(), max_norm=self.config.optim.clip_grad)
        
        temp_optimizer.step()
        
        return loss.item()
    
    def training_step_stage3(self, batch):
        """
        Stage 3: Train p on soft-labeled data.
        """
        self.fsdp_model.train()
        self.optimizer.zero_grad()
        
        loss = self._compute_soft_label_loss(batch)
        loss.backward()
        
        # Clip gradients
        if self.config.model.strategy == "fsdp":
            self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
            fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        return loss.item()
    
    def fit_anti_forgetting(self):
        """
        Main training loop with anti-forgetting mechanism.
        
        Outer loop structure:
        for outer_iter in range(n_outer_iterations):
            1. Create temp model q from p
            2. Train q on D (Stage 1)
            3. Construct soft labels from p0 and q (Stage 2)
            4. Train p on soft labels (Stage 3)
        """
        rank = self.device_mesh.get_rank()
        
        # Initialize tracking
        if rank == 0:
            from verl.utils.tracking import Tracking
            from omegaconf import OmegaConf
            
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name + "_anti_forgetting",
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )
        
        global_step = self.resume_global_step
        
        # Outer iterations
        for outer_iter in range(self.n_outer_iterations):
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Outer Iteration {outer_iter + 1}/{self.n_outer_iterations}")
                print(f"{'='*60}")
            
            # ================== Stage 1: Create and train q ==================
            if rank == 0:
                print(f"\n[Stage 1] Training temporary model q on original data...")
            
            temp_model, temp_optimizer = self._create_temp_sft_model()
            
            stage1_losses = []
            self.train_sampler.set_epoch(epoch=outer_iter * 2)  # Different seed each outer iter
            
            for step, data in enumerate(tqdm(
                self.train_dataloader,
                desc=f"Stage 1 (Outer {outer_iter+1})",
                disable=rank != 0
            )):
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                loss = self.training_step_stage1(data, temp_model, temp_optimizer)
                stage1_losses.append(loss)
                
                global_step += 1
                
                if rank == 0 and step % 10 == 0:
                    tracking.log({"stage1/loss": loss}, step=global_step)
            
            avg_stage1_loss = sum(stage1_losses) / len(stage1_losses)
            if rank == 0:
                print(f"Stage 1 completed. Avg loss: {avg_stage1_loss:.4f}")
            
            # ================== Stage 2: Construct soft labels ==================
            if rank == 0:
                print(f"\n[Stage 2] Constructing soft labels from p0 and q...")
            
            soft_label_dataset = []
            self.train_sampler.set_epoch(epoch=outer_iter * 2 + 1)
            
            for step, data in enumerate(tqdm(
                self.train_dataloader,
                desc=f"Stage 2 (Outer {outer_iter+1})",
                disable=rank != 0
            )):
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                
                # Construct soft labels
                soft_batch = self._construct_soft_labels(data, temp_model, outer_iter)
                soft_label_dataset.append(soft_batch.cpu())
                
                if rank == 0 and step == 0:
                    gold_ratio = sum([b["use_gold"].item() for b in [soft_batch]]) / len([soft_batch])
                    print(f"Sample: Using gold answers: {gold_ratio:.2%}")
            
            if rank == 0:
                print(f"Stage 2 completed. Constructed {len(soft_label_dataset)} soft-labeled batches.")
            
            # Clean up temp model
            del temp_model, temp_optimizer
            torch.cuda.empty_cache()
            
            # ================== Stage 3: Train p on soft labels ==================
            if rank == 0:
                print(f"\n[Stage 3] Training policy model p on soft labels...")
            
            stage3_losses = []
            for step, soft_batch in enumerate(tqdm(
                soft_label_dataset,
                desc=f"Stage 3 (Outer {outer_iter+1})",
                disable=rank != 0
            )):
                soft_batch = soft_batch.to(self.device_name)
                loss = self.training_step_stage3(soft_batch)
                stage3_losses.append(loss)
                
                global_step += 1
                
                if rank == 0 and step % 10 == 0:
                    tracking.log({"stage3/loss": loss}, step=global_step)
            
            avg_stage3_loss = sum(stage3_losses) / len(stage3_losses)
            if rank == 0:
                print(f"Stage 3 completed. Avg loss: {avg_stage3_loss:.4f}")
            
            # Validation after each outer iteration
            if rank == 0:
                print(f"\n[Validation] Running validation...")
            
            val_losses = []
            for val_data in self.val_dataloader:
                val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                    self.device_name
                )
                val_loss = self.validation_step(val_data)
                val_losses.append(val_loss)
            
            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                print(f"Validation loss: {val_loss.item():.4f}")
                tracking.log({
                    "val/loss": val_loss.item(),
                    "outer_iteration": outer_iter + 1,
                    "stage1/avg_loss": avg_stage1_loss,
                    "stage3/avg_loss": avg_stage3_loss,
                }, step=global_step)
            
            # Save checkpoint after each outer iteration
            self.save_checkpoint(step=global_step)
            
            torch.distributed.barrier()
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Anti-Forgetting Training Completed!")
            print(f"Total outer iterations: {self.n_outer_iterations}")
            print(f"Final global step: {global_step}")
            print(f"{'='*60}")


def run_anti_forgetting_sft(config):
    """Main entry point for anti-forgetting SFT training"""
    from verl.utils.device import get_device_name
    from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group
    from torch.distributed.device_mesh import init_device_mesh
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer
    
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()
    
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    
    # Build tokenizer and datasets
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    
    train_dataset = create_sft_dataset(
        config.data.train_files, 
        config.data, 
        tokenizer, 
        max_samples=config.data.get("train_max_samples", -1)
    )
    val_dataset = create_sft_dataset(
        config.data.val_files, 
        config.data, 
        tokenizer, 
        max_samples=config.data.get("val_max_samples", -1)
    )
    
    # Create anti-forgetting trainer
    trainer = AntiForgettingSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Run anti-forgetting training
    trainer.fit_anti_forgetting()
    
    destroy_global_process_group()

@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    use_migiting_forget = config.get("use_migiting_forget", None)
    
    print(f"********\nUse mitigating forget: {use_migiting_forget}\n********")

    if use_migiting_forget:
        run_anti_forgetting_sft(config)
    else:
        run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config, max_samples=max_samples)
    return dataset


if __name__ == "__main__":
    main()
