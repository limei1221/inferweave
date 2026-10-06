import torch
from torch import nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor | None):
        if temperatures is None:
            return logits.argmax(dim=-1)
        return self.sample(logits, temperatures)

    @torch.compile(dynamic=True)  # warmup's batch size is not serving's, so never recompile per size
    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)
        logits = logits.div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
