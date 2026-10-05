"""A step's input tensors on the device with no per-step allocation, as vLLM's InputBatch keeps them.

On CUDA each tensor has a pinned host buffer and a device buffer, both kept and grown by doubling. A step writes the
host side from numpy and copies it over once. The copy runs asynchronously, so an event keeps the next step from
rewriting the host side before it has. Elsewhere a step's arrays become fresh tensors, as there is nothing to stage.
"""

import numpy as np
import torch

NUMPY_DTYPES = {torch.int32: np.int32, torch.int64: np.int64, torch.float32: np.float32}
MIN_CAPACITY = 256


class InputBuffers:

    def __init__(self, device: torch.device):
        self.device = device
        self.staged = device.type == "cuda"
        self._host: dict[str, np.ndarray] = {}    # numpy views of the pinned tensors below
        self._pinned: dict[str, torch.Tensor] = {}
        self._device: dict[str, torch.Tensor] = {}
        self._copied: torch.cuda.Event | None = None

    def begin(self):
        """Before a step writes: the last step's copies must have read the host buffers."""
        if self._copied is not None:
            self._copied.synchronize()    # long done in a steady state: they queue behind the step before it
            self._copied = None

    def end(self):
        """After a step's last put."""
        if self.staged:
            self._copied = torch.cuda.Event()
            self._copied.record()

    def put(self, name: str, values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        """values on the device. On CUDA a view of name's buffer, which the next step's put of name overwrites."""
        values = np.asarray(values, NUMPY_DTYPES[dtype])
        if not self.staged:
            return torch.from_numpy(np.ascontiguousarray(values)).to(self.device)
        n = values.size
        if name not in self._device or self._device[name].numel() < n:
            capacity = max(n, 2 * (self._device[name].numel() if name in self._device else 0), MIN_CAPACITY)
            self._pinned[name] = torch.empty(capacity, dtype=dtype, pin_memory=True)
            self._host[name] = self._pinned[name].numpy()
            self._device[name] = torch.empty(capacity, dtype=dtype, device=self.device)
        self._host[name][:n] = values.reshape(-1)
        out = self._device[name][:n]
        out.copy_(self._pinned[name][:n], non_blocking=True)
        return out.view(values.shape)
