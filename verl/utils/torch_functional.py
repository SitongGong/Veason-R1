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
Contain small torch utilities
"""

import pdb, re
import math
from typing import List, Literal, Union

import torch
import torch.distributed
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizer


try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels):
    output = cross_entropy_loss(logits, labels)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]


def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)

        logprobs_labels = torch.stack(logprobs_labels)

    return logprobs_labels


def self_certainty_from_logits_and_labels(logits: torch.FloatTensor, labels):
    token_log_probs = logprobs_from_logits(logits, labels)    # bsz, response_length
    # pdb.set_trace()
    self_certainty = token_log_probs.mean(dim=-1)  # [batch]
    return self_certainty


def self_certainty_from_logits(logits: torch.Tensor):
    """Calculate self-certainty from logits."""    
    # pdb.set_trace()
    self_certainty = torch.logsumexp(logits, dim=-1) - logits.mean(dim=-1)
    return self_certainty.mean(dim=-1)


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def reasoning_certainty_from_logits(logits: torch.Tensor, response_ids: torch.Tensor, tokenizer=None, certainty_type="kl_panelty"):
    
    def find_think_span_by_offsets(response_ids, tokenizer):
        # 1) 解码整段文本（保留所有字符，不跳特殊符号）
        text = tokenizer.decode(response_ids, skip_special_tokens=False)

        # pdb.set_trace()

        # 2) 正则找到 <think>...</think> 的字符区间（拿内部内容的区间）
        m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if not m:
            return None, None  # 没找到

        char_start = m.start(1)  # 内部内容的起始字符位置
        char_end   = m.end(1)    # 内部内容的结束字符位置（开区间）

        # 3) 取 offsets，把字符区间映射到 token 下标
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        offsets = enc["offset_mapping"]  # [(start_char, end_char), ...]
        # 注意：enc["input_ids"] 是重新分的 ids，和 response_ids 一致（因 decode->encode）

        # 4) 找覆盖该字符区间的 token 范围
        start_tok = None
        end_tok   = None
        for ti, (s, e) in enumerate(offsets):
            if start_tok is None and s <= char_start < e:
                start_tok = ti
            if end_tok is None and s < char_end <= e:
                end_tok = ti + 1  # 右开区间
                break

        return start_tok, end_tok  # 都是在 enc["i
    
    # 获取 <think> 和 </think> 的 token ID
    think_token_ids = tokenizer.encode("<think>", add_special_tokens=False)
    endthink_token_ids = tokenizer.encode("<", add_special_tokens=False)
    
    # Stage 1: 将一个batch内的模型推理过程提取出来
    thinking_certainty = []

    # 对每个 batch 进行处理
    for num, response in enumerate(response_ids):
        # 找到 <think> 和 </think> 在 response 中的位置
        # start_idx = None
        # end_idx = None

        start_idx, end_idx = find_think_span_by_offsets(response, tokenizer)
        # # 在 response 中查找 <think> token IDs
        # for i in range(len(response) - len(think_token_ids) + 1):
        #     if torch.equal(response[i:i + len(think_token_ids)], torch.tensor(think_token_ids).to(response.device)):
        #         start_idx = i
        #         break

        # # 在 response 中查找 </think> token IDs
        # for i in range(len(response) - len(endthink_token_ids) + 1):
        #     if torch.equal(response[i:i + len(endthink_token_ids)], torch.tensor(endthink_token_ids).to(response.device)):
        #         end_idx = i + len(endthink_token_ids) # 包括 </think> 标记本身
        #         break

        # 如果找到 <think> 和 </think>，提取它们之间的推理内容
        if start_idx is not None and end_idx is not None:
            thinking_labels = response[start_idx: end_idx - len(endthink_token_ids)]      # reasoning_length
            thinking_logits = logits[num][start_idx: end_idx - len(endthink_token_ids)]      # reasoning_length, vocab_size
            if certainty_type == "logprob":
                self_certainty = self_certainty_from_logits_and_labels(thinking_logits.unsqueeze(0), thinking_labels.unsqueeze(0))
            elif certainty_type == "kl_panelty":
                self_certainty = self_certainty_from_logits(thinking_logits.unsqueeze(0))
            else:
                raise NotImplementedError(f"Unknown certainty computation method {certainty_type}")
            thinking_certainty.append(self_certainty)    
        else:
            if certainty_type == "logprob":
                self_certainty = self_certainty_from_logits_and_labels(logits[num].unsqueeze(0), response.unsqueeze(0))
            elif certainty_type == "kl_panelty":
                self_certainty = self_certainty_from_logits(logits[num].unsqueeze(0))
            else:
                raise NotImplementedError(f"Unknown certainty computation method {certainty_type}")
            thinking_certainty.append(self_certainty)

    return torch.cat(thinking_certainty, dim=0)       # bsz   


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def masked_mean(values, mask, axis=None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def get_eos_mask(response_ids: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64):
    """
    end of sentence token can be int or list: 1 or [1, 2]
    e.g. eos_token=1
    response_ids: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    if isinstance(eos_token, int):
        eos_token = [eos_token]

    eos_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    for token in eos_token:
        eos_mask |= response_ids.eq(token)

    eos_mask = eos_mask.long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def pad_2d_list_to_length(response, pad_token_id, max_length=None) -> torch.Tensor:
    """
    pad a 2D list (e.g. responses, logprobs) to a 2D tensor.
    """
    response_length = max(len(sub_list) for sub_list in response)
    if max_length is not None and max_length > response_length:
        target_length = max_length
    else:
        target_length = response_length
    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    tensor = torch.tensor(padded_response)
    return tensor


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors

    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)


def tokenize_and_postprocess_data(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    pad_token_id: int,
    left_pad: bool = True,
    truncation: Literal["left", "right", "error"] = "error",
):
    """
    input_data is the output from tokenizer.
    """
    assert truncation in ["left", "right", "error"]

    input_data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    input_ids = input_data["input_ids"][0]
    attention_mask = input_data["attention_mask"][0]
    sequence_length = len(input_ids)
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > max_length:
        if truncation == "left":
            # actually, left truncation may not be reasonable
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
        elif truncation == "right":
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        elif truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}")

    return input_ids, attention_mask


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[List[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        no_padding_batch.append((ids[len(ids) - mask.sum() :]).cpu().numpy().tolist())
    return no_padding_batch


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(0.0, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        return min(1, float(current_step) / float(max(1, num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
