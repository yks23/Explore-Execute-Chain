# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def find_special_token_position(response_seq: torch.Tensor, token_id_1: int, token_id_2: int) -> int:
    """
    Find the position of the first occurrence of special token in the response.
    
    Args:
        response_seq: Response token sequence of shape [seq_len]
        token_id_1: Primary special token ID to search for
        token_id_2: Fallback special token ID if primary not found
        
    Returns:
        int: Position of the special token, or -1 if not found
    """
    mask_pos = (response_seq == token_id_1).nonzero(as_tuple=True)[0]
    if len(mask_pos) == 0:
        mask_pos = (response_seq == token_id_2).nonzero(as_tuple=True)[0]
    
    if len(mask_pos) > 0:
        return mask_pos[0].item()
    return -1


def apply_mask_reweighting(
    base_mask: torch.Tensor,
    responses: torch.Tensor,
    token_id_1: int,
    token_id_2: int,
    coef_after: float = 1.0,
    coef_before: float = 1.0,
) -> torch.Tensor:
    """
    Apply reweighting to mask based on special token position.
    
    Splits the sequence at the special token and applies different coefficients
    to the parts before and after the token.
    
    Args:
        base_mask: Base mask tensor of shape [batch_size, seq_len]
        responses: Response token sequences of shape [batch_size, seq_len]
        token_id_1: Primary special token ID
        token_id_2: Fallback special token ID
        coef_after: Coefficient to apply after the special token
        coef_before: Coefficient to apply before the special token
        
    Returns:
        torch.Tensor: Reweighted mask of same shape as base_mask
    """
    # Ensure the mask is float type for coefficient multiplication
    reweighted_mask = base_mask.clone().float()
    
    for b in range(reweighted_mask.shape[0]):
        split_idx = find_special_token_position(responses[b], token_id_1, token_id_2)
        
        if split_idx >= 0:
            # Apply coefficients to parts before and after the split
            reweighted_mask[b, split_idx:] *= coef_after
            reweighted_mask[b, :split_idx] *= coef_before
        else:
            # If no special token found, apply before coefficient to entire sequence
            reweighted_mask[b, :] *= coef_before
    
    # Convert back to the original dtype if needed
    if base_mask.dtype != torch.float:
        reweighted_mask = reweighted_mask.to(base_mask.dtype)
    
    return reweighted_mask

class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, apply_era=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # Apply ERA activation if enabled
                    if apply_era and self.config.get("era", {}).get("enabled", False):
                        from verl.utils.era_activation import apply_era_activation
                        
                        # Get advantages from micro_batch if available
                        advantages = micro_batch.get("advantages", None)
                        response_mask = micro_batch.get("response_mask", None)
                        
                        if advantages is not None and response_mask is not None:
                            # Calculate entropy first
                            if not self.config.get("entropy_checkpointing", False):
                                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                            else:
                                entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                    self.compute_entropy_from_logits, logits_rmpad
                                )
                            
                            # Get ERA configuration
                            era_config = self.config.get("era", {})
                            
                            # Apply ERA activation with statistics
                            logits_rmpad, advantages_modified, era_stats = apply_era_activation(
                                logits=logits_rmpad,
                                entropy=entropy_rmpad,
                                advantages=advantages,
                                response_mask=response_mask,
                                era_lb=era_config.get("era_lb", 0.45),
                                era_ub=era_config.get("era_ub", 3.0),
                                era_k=era_config.get("era_k", 2.0),
                                topk_ratio=era_config.get("topk_ratio", 0.2),
                                indices=indices,
                                seqlen=seqlen,
                                return_stats=True,
                            )
                            
                            # Store ERA statistics for logging
                            if era_stats is not None:
                                micro_batch["era_stats"] = era_stats
                            
                            # Update advantages in micro_batch
                            micro_batch["advantages"] = advantages_modified

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # Initialize ERA statistics accumulator
                era_stats_accumulator = {
                    'era_total_tokens': 0,
                    'era_low_entropy_tokens': 0,
                    'era_high_entropy_tokens': 0,
                    'era_normal_tokens': 0,
                    'era_mean_entropy': 0.0,
                    'era_entropy_std': 0.0,
                    'era_batch_count': 0
                }
                
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    responses = model_inputs["responses"]

                    # Get config values with defaults
                    entropy_coeff = self.config.entropy_coeff
                    adv_coeff = self.config.get("adv_coeff", 1.0)
                    special_token_1 = self.config.get("special_token_1", 151672)
                    special_token_2 = self.config.get("special_token_2", 151673)
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # Reweight advantages for exploration phase if configured
                    if adv_coeff != 1.0:
                        advantages = apply_mask_reweighting(
                            base_mask=advantages,
                            responses=responses,
                            token_id_1=special_token_1,
                            token_id_2=special_token_2,
                            coef_after=1.0,
                            coef_before=adv_coeff,
                        )

                    # Forward pass
                    calculate_entropy = entropy_coeff != 0
                    # Check if ERA is enabled and we're in training mode
                    apply_era = (self.config.get("era", {}).get("enabled", False) and 
                               self.config.policy_loss.get("loss_mode", "vanilla") == "era")
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, 
                        temperature=temperature, 
                        calculate_entropy=calculate_entropy,
                        apply_era=apply_era
                    )

                    # Compute policy loss
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )

                    # Add entropy loss if configured
                    if entropy_coeff != 0:
                        entropy_mask_coef_after = self.config.get("entropy_mask_coef_after", 0.0)
                        entropy_mask_coef_before = self.config.get("entropy_mask_coef_before", 0.0)
                        entropy_mask_min_pos = self.config.get("entropy_mask_min_position", 5)
                        
                        # Apply entropy mask reweighting
                        entropy_mask = response_mask.clone()
                        for b in range(entropy_mask.shape[0]):
                            split_idx = find_special_token_position(
                                responses[b], special_token_1, special_token_2
                            )
                            
                            if split_idx >= 0:
                                if split_idx < entropy_mask_min_pos:
                                    # If split position is too early, zero out entire mask
                                    entropy_mask[b, :] = 0
                                else:
                                    # Apply coefficients to before and after parts
                                    entropy_mask[b, split_idx:] *= entropy_mask_coef_after
                                    entropy_mask[b, :split_idx] *= entropy_mask_coef_before
                            else:
                                # No special token found, zero out the mask
                                entropy_mask[b, :] = 0
                        
                        entropy_loss = agg_loss(
                            loss_mat=entropy, loss_mask=entropy_mask, loss_agg_mode=loss_agg_mode
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    # Add KL loss if configured
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kl_mask_coef_after = self.config.get("kl_mask_coef_after", 1.0)
                        kl_mask_coef_before = self.config.get("kl_mask_coef_before", 0.0)
                        
                        # Compute KL divergence
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        
                        # Apply KL mask reweighting
                        kl_mask = apply_mask_reweighting(
                            base_mask=response_mask,
                            responses=responses,
                            token_id_1=special_token_1,
                            token_id_2=special_token_2,
                            coef_after=kl_mask_coef_after,
                            coef_before=kl_mask_coef_before,
                        )
                        
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=kl_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    # Collect ERA statistics if available
                    if "era_stats" in model_inputs and model_inputs["era_stats"] is not None:
                        era_stats = model_inputs["era_stats"]
                        era_stats_accumulator['era_total_tokens'] += era_stats['era_total_tokens']
                        era_stats_accumulator['era_low_entropy_tokens'] += era_stats['era_low_entropy_tokens']
                        era_stats_accumulator['era_high_entropy_tokens'] += era_stats['era_high_entropy_tokens']
                        era_stats_accumulator['era_normal_tokens'] += era_stats['era_normal_tokens']
                        era_stats_accumulator['era_mean_entropy'] += era_stats['era_mean_entropy']
                        era_stats_accumulator['era_entropy_std'] += era_stats['era_entropy_std']
                        era_stats_accumulator['era_batch_count'] += 1

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                
                # Calculate and add ERA statistics to metrics
                if era_stats_accumulator['era_batch_count'] > 0:
                    era_stats_metrics = {
                        "era/total_tokens": era_stats_accumulator['era_total_tokens'],
                        "era/low_entropy_tokens": era_stats_accumulator['era_low_entropy_tokens'],
                        "era/high_entropy_tokens": era_stats_accumulator['era_high_entropy_tokens'],
                        "era/normal_tokens": era_stats_accumulator['era_normal_tokens'],
                        "era/low_entropy_ratio": era_stats_accumulator['era_low_entropy_tokens'] / max(era_stats_accumulator['era_total_tokens'], 1),
                        "era/high_entropy_ratio": era_stats_accumulator['era_high_entropy_tokens'] / max(era_stats_accumulator['era_total_tokens'], 1),
                        "era/normal_ratio": era_stats_accumulator['era_normal_tokens'] / max(era_stats_accumulator['era_total_tokens'], 1),
                        "era/mean_entropy": era_stats_accumulator['era_mean_entropy'] / era_stats_accumulator['era_batch_count'],
                        "era/entropy_std": era_stats_accumulator['era_entropy_std'] / era_stats_accumulator['era_batch_count'],
                        "era/batch_count": era_stats_accumulator['era_batch_count'],
                    }
                    mini_batch_metrics.update(era_stats_metrics)
                
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
