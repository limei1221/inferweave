import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from lean_vllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(-1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor, all_gather: bool = False):
        """Logits on rank 0 alone, which samples, or with all_gather on every rank, as vLLM's LogitsProcessor."""
        context = get_context()
        if context.logits_indices is not None:
            # A chunk that has not finished its prompt has no token to sample.
            # [num_batch_tokens, hidden_dim] -> [num_sampling_rows, hidden_dim]
            x = x[context.logits_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1 and all_gather:
            gathered = logits.new_empty(self.tp_size * logits.size(0), logits.size(1))
            dist.all_gather_into_tensor(gathered, logits)
            # [tp * rows, part] -> [rows, tp, part] -> [rows, vocab]
            return gathered.view(self.tp_size, *logits.shape).transpose(0, 1).reshape(logits.size(0), -1)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            return torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
