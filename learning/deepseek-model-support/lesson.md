# DeepSeek model support

Working through Project 2: MLA + MoE + YaRN, as implemented for DeepSeek-V2-Lite. Each lesson is one focused increment, grounded in the repository's code and its design document, `docs/deepseek-v2.md`.

Lessons, in order:

1. The latent KV cache: what MLA changes about the cache
2. How attention runs a step over that cache: chunked context and `mla_decode`
3. MoE: routing and the fused expert GEMMs
4. YaRN and the attention scale
5. How it all fits the CUDA-graph capture modes

## Lesson 1: the latent KV cache

One concept this lesson: in DeepSeek-V2, a token's keys and values are never cached per head. The cache stores one compressed latent per token, and keys/values are derived from it on demand.

### The shape of what is cached

In the standard MHA you already know, every token writes $K$ and $V$ for each KV head into the cache, and attention reads them back later. The cost scales with `num_kv_heads * head_dim * 2`.

MLA (multi-head latent attention) replaces that with a low-rank compression. For DeepSeek-V2-Lite, the number is 576 values per token per layer against the 5120 a plain cache of 16 heads would need (192-dim keys plus 128-dim values), about 31 KB per token in bf16 across all 27 layers (see `docs/deepseek-v2.md`, "MLA: the cache holds latents"). That roughly ninefold reduction is the reason the attention half of this project exists.

The 576 splits into two parts:

- **The compressed KV**, `kv_lora_rank = 512` values. This is a shared, head-independent vector.
- **The rope key**, `qk_rope_head_dim = 64` values, one key shared by all heads (hence `_with_mqa` in the projection's name). If rope rotated keys expanded from the latent, a position-dependent rotation would sit between the query and the up projection, and the up projection could no longer be moved onto the query, the trick lesson 2's decode path relies on. So this small key part stays outside the compression, is rotated on its own, and is stored with rope already applied.

### Where the latent is produced

`DeepseekV2Attention` (`lean_vllm/models/deepseek_v2.py:17`) owns the projections:

```python
# lean_vllm/models/deepseek_v2.py:43-49
self.kv_a_proj_with_mqa = ReplicatedLinear(hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias)
self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
self.kv_b_proj = ColumnParallelLinear(
    self.kv_lora_rank,
    self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
    bias=False,
)
```

- `kv_a_proj_with_mqa` (the "a" projections) is the *down* projection: hidden state to the latent, 512 + 64 wide.
- `kv_b_proj` (the "b" projection) is the *up* projection: latent back out to per-head keys and values. It runs only when reading, never before caching. That choice is what makes the cache small.

The per-step production happens in `project()` (`lean_vllm/models/deepseek_v2.py:89`). It returns exactly two tensors: the query and the latent to cache. The lines that matter:

```python
# lean_vllm/models/deepseek_v2.py:101-105
kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
kv_c = self.kv_a_layernorm(kv_c)
q_pe, k_pe = self.rotary_emb(positions, q_pe, rearrange(k_pe, "n d -> n 1 d"))
# The latent is cached normalized and with rope applied, so a read needs only kv_b_proj.
return torch.cat([q_nope, q_pe], dim=-1), torch.cat([kv_c, rearrange(k_pe, "n 1 d -> n d")], dim=-1)
```

Two facts to hold on to:

1. The compressed part is normalized before caching (`kv_a_layernorm`), so the cached latent is the final form. Nothing downstream re-normalizes.
2. RoPE is applied to the 64-dim key part before caching. The cache therefore holds `norm(W_DV h)` concatenated with `rope(k_pe)`.

A detail from the same method: the query is also split into a nope part (128 dims per head for V2-Lite) and a rope part (64 dims); only the rope part is rotated. V2-Lite does not compress the query. `q_lora_rank` is `None`, so `q_proj` goes straight from hidden state to heads (`lean_vllm/models/deepseek_v2.py:37-38`). The code keeps a second branch with `q_a_proj`/`q_b_proj` for checkpoints that do compress the query.

### Where the cache lives and what shape it has

The cache is not owned by the model. `MLAAttention` (`lean_vllm/layers/attention.py:193`), the execution layer, is separate from `DeepseekV2Attention`, the weights layer. It declares its shape and the runner hands back a slice:

```python
# lean_vllm/layers/attention.py:219-222
def kv_cache_shape(self, num_blocks: int, block_size: int) -> tuple[int, ...]:
    return (1, num_blocks, block_size, self.latent_dim)

def bind_kv_cache(self, cache: torch.Tensor):
    self.latent_cache = cache[0]
```

Compare the base class at `lean_vllm/layers/attention.py:74-78`: a standard layer answers with a stacked key-and-value pair in the backend's layout, while MLA answers with a single tensor of one latent per slot. The runner asks layers for their shapes rather than reading head counts off the config, which is what lets these two layouts coexist (Qwen3 uses the standard one; see `lean_vllm/models/qwen3.py`).

At each step, `attend()` (`lean_vllm/layers/attention.py:231`) scatters the step's latents into their paged-cache slots:

```python
self.backend.store_latents(latent, cache, context.slot_mapping)
```

That is the whole write path: one 576-wide row per token per layer.

### Reading the cache back: expansion

When attention needs actual keys and values, they are regenerated from the latents by `expand()` (`lean_vllm/models/deepseek_v2.py:75`), passed into `MLAAttention` as a callable at construction (`lean_vllm/models/deepseek_v2.py:65-73`):

```python
# lean_vllm/models/deepseek_v2.py:75-81
def expand(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kv_c, k_pe = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv = rearrange(self.kv_b_proj(kv_c), "n (h d) -> n h d", h=self.num_heads)
    k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.cat([k_nope, repeat(k_pe, "n d -> n h d", h=self.num_heads)], dim=-1)
    return k, v
```

Note the asymmetry: the 512-dim compressed part goes through `kv_b_proj` to become per-head nope-key and value, while the 64-dim rope part is repeated across the heads, because it is already final. Keys are `[nope | rope]` = 192 dims, values are 128 dims.

One subtlety, verified in `_pad()` (`lean_vllm/layers/attention.py:287`): values are zero-padded from 128 to 192 so the backend sees a single head size. The padding leaves the scores alone, and since the output is a weighted sum of values, zero value lanes give zero output lanes, which are then cut off (`o[..., :self.v_head_dim]` in `_prefill`, `lean_vllm/layers/attention.py:257` and `:274`).

### How the pieces interact

The data flow across the two files:

```text
DeepseekV2DecoderLayer.pre_attention            (models/deepseek_v2.py:178)
  └─ DeepseekV2Attention.project                (models/deepseek_v2.py:89)
       returns (query, latent), the latent already normalized and rope-applied
DeepseekV2DecoderLayer.forward                  (models/deepseek_v2.py:201)
  └─ MLAAttention(q, latent)                    via torch.ops.lean_vllm.mla_attention
       └─ attend                                (layers/attention.py:231)
            ├─ store_latents into the paged latent cache
            └─ prefill/decode over expanded keys/values, or decode over raw latents
```

Two things to notice about the plumbing:

- `MLAAttention.forward` routes through the custom op `torch.ops.lean_vllm.mla_attention` (`lean_vllm/layers/attention.py:228-229`). The layer is looked up by name because a custom op cannot take modules (`lean_vllm/layers/attention.py:14`). Being opaque, the op is where a `torch.compile` of the model would split its graph (`lean_vllm/layers/attention.py:86`). Nothing compiles the whole model today, and lesson 5 shows the runner's piecewise capture splits around attention by hand.
- The backend is chosen with `get_attention_backend(mla=True)` (`lean_vllm/layers/attention.py:212`), so an MLA layer may land on a different backend than a standard one in the same process.

The trade-off that drives everything else is that expansion is compute. A resumed prefill step that expands the whole cached context pays for `kv_b_proj` on every cached token in every layer, chunk by chunk. The next lesson covers how the step actually runs: chunked context merged by log-sum-exp, and the `mla_decode` path that skips expansion entirely for decode rows.

### Verified vs. not yet run

Per `docs/deepseek-v2.md`, all of the above is implemented and checked against transformers on tiny random checkpoints, on the torch backend, including end-to-end greedy generation. On hardware, the 20 September benchmark (`docs/benchmark-2026-09-20.md`) served the real DeepSeek-V2-Lite-Chat weights on an H100, with FlashMLA decode and both CUDA-graph modes. That was a performance run; a numerical comparison of FlashMLA decode against the torch reference is still not recorded.

## Lesson 2: running a step over the latent cache

Lesson 1 ended at the store: each step scatters its latents into the paged cache, then attention runs. This lesson covers everything `attend()` does between that store and its return, across three paths: a cold prefill, a resumed prefill, and a decode over raw latents.

### The step's metadata

Every forward pass gets a fresh `Context` (`lean_vllm/utils/context.py:7`, built by `set_context` at `lean_vllm/utils/context.py:31`). It is a plain dataclass the engine fills before the model runs, and the layers read through `get_context()`. The fields this lesson needs:

- `cu_seqlens_q`, `cu_seqlens_k`: cumulative query and key lengths of the rows packed into the step, the varlen format. The `_host` copies (`lean_vllm/utils/context.py:18`) stay on the CPU, so the layer can plan without reading tensor contents back from the device.
- `keys_are_new`: no row carries cached keys, so the step's own keys are the whole batch (`lean_vllm/utils/context.py:13`).
- `block_tables` and `context_lens`: the paged-cache layout of each row and how many tokens each row already holds.
- `prefill_rows`: each row's phase, as the scheduler recorded it. A one-token prefill is still a prefill (`lean_vllm/utils/context.py:22`).
- `context_chunks`, `mla_partitions`, `mla_decode_metadata`: memo slots, filled on first use and shared across the layers of one step. Layers of the same step see the same lengths, so the first layer's plan serves the rest.

### The dispatch

`MLAAttention.attend()` (`lean_vllm/layers/attention.py:231`) stores the latents, then picks one of three paths:

```python
# lean_vllm/layers/attention.py:234-248, abridged
if not context.is_prefill and self.backend.supports_mla_decode():
    return self._decode_latents(q, context)                       # pure decode
if (context.is_prefill and context.prefill_rows is not None
        and not all(context.prefill_rows) and self.backend.supports_mla_decode()):
    ...                                                            # mixed step, split below
return self._prefill(q, latent, context)                           # expanded path
```

Read the conditions in order: a decode step on a backend with `mla_decode` attends the latents as they are. A prefill step whose rows mix phases, on such a backend, splits into a prefill subset and a decode subset (`mla_partitions`, covered below). Everything else takes the expanded path, `keys_are_new` or not. A backend without `mla_decode` uses the expanded path for every row, decode included.

### Path one: cold prefill, nothing to read back

When every key the step reads is one it just wrote (`keys_are_new`), the paged cache is irrelevant and so is the page table. `_prefill` (`lean_vllm/layers/attention.py:250`) expands the step's own latents and calls the backend's plain `prefill` with `block_tables=None`:

```python
# lean_vllm/layers/attention.py:252-257
k, v = self.expand(latent)
if context.keys_are_new or context.block_tables is None:
    unpaged = dataclasses.replace(context, block_tables=None, keys_are_new=True)
    o = self.backend.prefill(q, k, self._pad(v), self.k_cache, self.v_cache, unpaged)
    return o[..., :self.v_head_dim].contiguous()
```

One causal varlen call covers the whole batch. No chunking, no merging.

### Path two: resumed prefill, the chunked context

A resumed row has cached keys behind its new tokens, so the step reads the cache back. `_prefill` splits the work in two:

1. The new tokens attend each other, causally. `varlen_with_lse` (`lean_vllm/layers/attention.py:260-262`) runs over the step's own keys only, with `cu_seqlens_q` passed for both the query and the key side, since each new token contributes exactly one key.
2. Each chunk of cached keys is attended unmasked, and its result is merged into the running output by log-sum-exp.

The backend contract (`lean_vllm/attention/abstract.py:83`, `:101`) says the causal mask is bottom-right aligned: query $j$ sits at key position $\text{seqlen}_k - \text{seqlen}_q + j$. In step 1 the two lengths are equal, so the alignment makes no difference there. It matters in `backend.prefill` over paged keys, where $\text{seqlen}_k > \text{seqlen}_q$: a one-token query sits last and sees every key, which is why `_causal_mask` returns `None` for it (`lean_vllm/attention/torch_backend.py:124`).

The cached side is processed in chunks bounded by `max_context_chunk` (`lean_vllm/layers/attention.py:199`), whose value the runner sets to the step token budget. The plan is built once per step by `context_chunks()` (`lean_vllm/layers/attention.py:133`), which wraps `plan_context_chunks()` (`lean_vllm/layers/attention.py:111`). Three properties of the plan:

- It runs on the host from `cu_seqlens_q_host` and `cu_seqlens_k_host`, so nothing reads a device tensor's contents and nothing syncs.
- Consecutive rows pack into one chunk until the budget is spent, and a row longer than the budget splits across chunks. The doc's test suite exercises exactly that: a budget of 4 puts the end of one row and the start of the next in the same chunk.
- Each `ContextChunk` (`lean_vllm/layers/attention.py:101`) carries its slice of the queries, its own `cu_seqlens` pair, and `slots`: the cache slot of every key in the chunk, resolved from the block tables ahead of the attention call.

The loop over chunks (`lean_vllm/layers/attention.py:265-273`) gathers a chunk's latents by `chunk.slots`, expands them with the layer's `expand`, and attends them unmasked, since every cached key precedes every query of the step:

```python
# lean_vllm/layers/attention.py:265-273, abridged
for chunk in context_chunks(context, cache.size(1), self.max_context_chunk):
    k, v = self.expand(latents[chunk.slots])
    o_chunk, lse_chunk = self.backend.varlen_with_lse(
        q[rows], k, self._pad(v), chunk.cu_seqlens_q, chunk.cu_seqlens_k,
        chunk.max_seqlen_q, chunk.max_seqlen_k, causal=False,
    )
    o[rows], lse[rows] = merge_attention(o[rows], lse[rows], o_chunk, lse_chunk)
```

`merge_attention` (`lean_vllm/layers/attention.py:159`) combines attention outputs over two disjoint key sets from each one's output and log-sum-exp. With $s_a = e^{lse_a}$ and $s_b = e^{lse_b}$, the merged output is

$$o = \frac{s_a\, o_a + s_b\, o_b}{s_a + s_b}, \qquad lse = \log(e^{lse_a} + e^{lse_b})$$

and the code computes the weight as `torch.sigmoid(lse_b - lse_a)`, which is exactly $\frac{s_b}{s_a + s_b}$, then lerps:

```python
# lean_vllm/layers/attention.py:159-163
def merge_attention(o_a, lse_a, o_b, lse_b) -> tuple[torch.Tensor, torch.Tensor]:
    weight_b = rearrange(torch.sigmoid(lse_b - lse_a), "n h -> n h 1")
    o = torch.lerp(o_a.float(), o_b.float(), weight_b)
    return o.to(o_a.dtype), torch.logaddexp(lse_a, lse_b)
```

Every chunk merges into the running result the same way, so the split across chunks changes memory use, not the answer.

### Path three: decode over the latents, nothing expands

The expanded path has a cost you can now name: a resumed row re-runs `kv_b_proj` on its whole cached context, in every layer, every step. `_decode_latents` (`lean_vllm/layers/attention.py:276`) removes that cost for decode rows by moving the query into latent space instead of moving the keys out of it. The keys are $W_k c$ for a latent $c$, and the inner product lets the projection switch sides:

$$q_{nope} \cdot (W_k c) = (W_k^{\top} q_{nope}) \cdot c$$

So the query's nope part is projected by the key half of `kv_b_proj`, concatenated back onto the untouched rope part, and attention then runs against the raw cached latents as one key head shared by all query heads:

```python
# lean_vllm/layers/attention.py:276-285, abridged
w_k, w_v = self.latent_projections()
q_nope, q_pe = q.split([w_k.size(1), self.head_dim - w_k.size(1)], dim=-1)
q = torch.cat([torch.einsum("bhn,hnl->bhl", q_nope, w_k), q_pe], dim=-1)
o = self.backend.mla_decode(q, self.latent_cache, w_k.size(2), context)
return torch.einsum("bhl,hvl->bhv", o, w_v)
```

`latent_projections` (`lean_vllm/models/deepseek_v2.py:83`) hands over `kv_b_proj`'s weight reshaped per head, split into its key rows `w_k` and value rows `w_v`. Two facts complete the picture:

- The values are the first $kv\_lora\_rank$ entries of each latent, per the backend contract (`lean_vllm/attention/abstract.py:113`). The latent was built as `norm(kv_c)` concatenated with `rope(k_pe)`, and the first 512 entries are the compressed KV, so they are the right values.
- Attention is linear in the values, so applying $W_v$ after the weighted sum equals applying it before. The output is `[batch, heads, v_head_dim]`, the same shape the expanded path returns after cutting the padding.

For V2-Lite a decode query is 576 wide per head in latent space (512 projected + 64 rope), and the value read is 512 wide, so the whole decode touches only the 576-wide cache rows. No `kv_b_proj` on any cached token.

This is not free in arithmetic: each score is now a 576-wide dot product instead of a 192-wide one. The win is memory traffic. Decode is bound by reading the cache, and here all 16 heads read one shared 576-wide row per token rather than their own expanded keys and values.

### Mixed steps: splitting by phase

A step can carry a resumed prompt beside decode rows. When the backend supports `mla_decode`, `attend()` splits it with `mla_partitions()` (`lean_vllm/layers/attention.py:166`), which builds, once per step, a subset `Context` for each phase and the token indices of its rows. Each subset then takes its own path from above: the prefill subset runs `_prefill`, the decode subset runs `_decode_latents`. Both outputs are scattered back into the original token order with `index_copy_` before the layer returns (`lean_vllm/layers/attention.py:241-246`). Only prefill rows expand anything.

### The backends behind the contract

`AttentionBackend` (`lean_vllm/attention/abstract.py`) defines the pieces this lesson used. `supports_mla_decode` (`lean_vllm/attention/abstract.py:33`) is the switch `attend()` reads. `varlen_with_lse` (`:90`) is abstract, so every backend implements it; `store_latents` (`:64`) is not, and its default raises `NotImplementedError`, so only backends that serve MLA implement it.

`TorchAttention` (`lean_vllm/attention/torch_backend.py:9`) implements everything, as the reference. Its `mla_decode` (`lean_vllm/attention/torch_backend.py:101`) gathers each row's latents from the pages and runs SDPA with the latent serving as both the key and, truncated to `v_dim`, the value:

```python
# lean_vllm/attention/torch_backend.py:106-110, abridged
latent = self._gather_pages(rearrange(latent_cache, "n p d -> n p 1 d"), block_tables[i], seqlen_k)
kv = repeat(latent, "l 1 d -> 1 h l d", h=q.size(1))
o = F.scaled_dot_product_attention(
    rearrange(q[i:i + 1], "b h d -> 1 h b d"), kv, kv[..., :v_dim], scale=self.scale,
)
```

`FlashMLABackend` (`lean_vllm/attention/flashmla_backend.py:14`) runs the real kernel instead: FlashMLA's dense decode, with FlashAttention-3 inherited for every other call. It requires 64-token pages (`mla_block_size`, `lean_vllm/attention/flashmla_backend.py:40`), which is why the config sets the cache block size to 64 when this backend is chosen. Its `mla_decode` (`lean_vllm/attention/flashmla_backend.py:43`) calls `get_mla_metadata()`, with no arguments, on the first layer's call and stashes the result in `context.mla_decode_metadata`. That object is a holder: in this FlashMLA build the kernel itself computes the tile schedule from `context_lens` and keeps it there, and the rest of the step's layers reuse it, since they share the step's lengths.

### The cost of each path

- Cold prefill: one expansion of the step's own tokens, one causal varlen call.
- Resumed prefill: the same, plus re-expansion of the whole cached context, chunk by chunk, in every layer of the step.
- Decode over latents: no expansion at all; the query moves to latent space once per step.

A decode row is a resumed row with one query, so on the expanded path it would pay that middle cost every step. That is why the capability check exists. Chunking bounds the memory, not the compute.

### Verified vs. not yet run

The torch reference for every path above is exercised by `tests/test_deepseek_v2.py` against transformers, including the mixed-row regression that counts expanded tokens to confirm decode context is not expanded (see `docs/deepseek-v2.md`, "Testing"). FlashMLA's decode has run on an H100 in the 20 September benchmark, mixed steps included, but only as a performance run; no numerical comparison against the torch reference is recorded.

## Lesson 3: the MoE layer

One concept this lesson: how a token crosses the routed MoE layer. The router picks $k$ of $E$ experts and one weight for each pick, and that pair of tensors is the entire interface between the routing decision and the expert compute. You know the MoE idea conceptually; this lesson is the concrete version of it, in two implementations: a portable one built on `grouped_mm`, and a Triton kernel modeled on vLLM's.

### Where the MoE sits in the model

`DeepseekV2DecoderLayer.__init__` (`lean_vllm/models/deepseek_v2.py:169`) picks the layer's MLP once, at build time:

```python
# lean_vllm/models/deepseek_v2.py:169-170
is_moe = (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
          and layer_idx % (getattr(config, "moe_layer_freq", None) or 1) == 0)
```

The first `first_k_dense_replace` layers use the dense MLP; the rest switch. So "MoE model" really means "MoE from some layer onward".

`DeepseekV2MoE` (`lean_vllm/models/deepseek_v2.py:112`) holds three pieces:

- `gate` (`:127`), a replicated linear from `hidden_size` to `n_routed_experts`: the router.
- `experts` (`:128`), a `FusedMoE`: all routed experts in one module, this lesson's main subject.
- `shared_experts` (`:129-135`), a plain dense MLP (`DeepseekV2MLP`, aliased at `lean_vllm/models/deepseek_v2.py:14` to `Qwen3MLP` from `lean_vllm/models/qwen3.py:94`). Shared experts always run for every token, outside the routing, and their output is added at the end (`lean_vllm/models/deepseek_v2.py:153-157`).

So a token's feed-forward cost is `top_k` routed experts plus the shared expert, regardless of how many experts exist. The parameter cost is where all $E$ experts are paid.

### What one expert is

An expert is the same gated SiLU MLP the dense layers use, with nothing MoE-specific in it: a merged `gate_up_proj`, `SiluAndMul` (`lean_vllm/layers/activation.py:6`, which computes $\mathrm{silu}(x_1) \cdot x_2$ over the two halves of the stacked projection), then `down_proj`. `FusedMoE` (`lean_vllm/layers/moe.py:12`) just stacks $E$ copies of those weights into two 3D tensors:

```python
# lean_vllm/layers/moe.py:31-32
self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * self.intermediate_size, hidden_size))
self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, self.intermediate_size))
```

The loader (`lean_vllm/layers/moe.py:37`) maps each checkpoint weight `experts.{e}.{proj}.weight` into row `e` of the stack, splitting gate and up into the two halves of `gate_up_proj`. Tensor parallelism shards the intermediate size (`lean_vllm/layers/moe.py:30`, `divide(intermediate_size, tp_size)`), so each rank holds a slice of every expert, and `forward` ends with an `all_reduce` when the world is larger than one (`lean_vllm/layers/moe.py:65-66`).

### Routing

`route()` (`lean_vllm/models/deepseek_v2.py:137`) turns the hidden state into $k$ expert ids and $k$ weights. Three steps, matching the original V2 code:

1. Scores, in float32 for stability:

$$p = \mathrm{softmax}(W_g\, x), \qquad p \in \mathbb{R}^{E}$$

2. Selection. `greedy` is a plain top-$k$ over $p$. `group_limited_greedy` first marks the `topk_group` best groups (group score = max score in the group) as eligible and zeroes the rest (`lean_vllm/models/deepseek_v2.py:141-145`), then takes top-$k$ among the survivors. The two asserts at `:119-120` are the whole config surface this layer supports: softmax scoring and one of those two methods.
3. Weighting (`:147-150`): with `norm_topk_prob` and $k > 1$, the $k$ weights are renormalized to sum to one; otherwise they are multiplied by `routed_scaling_factor`. An epsilon of 1e-20 guards the division.

The doc's test suite runs one config with each method, so both branches are covered (`docs/deepseek-v2.md`, "Testing").

### The portable path: one row per pair

`torch_experts` (`lean_vllm/layers/moe.py:46`) is the reference implementation, and the one that runs on CPU and Apple Silicon. It reorganizes the batch so the expert GEMMs become two `F.grouped_mm` calls:

```python
# lean_vllm/layers/moe.py:48-56, abridged
expert_ids, order = rearrange(topk_ids, "n k -> (n k)").sort()
token_ids = order // self.top_k
offsets = torch.searchsorted(expert_ids, experts, right=True).to(torch.int32)
h = F.grouped_mm(x[token_ids], rearrange(self.gate_up_proj, "e o i -> e i o"), offs=offsets)
h = F.grouped_mm(self.act_fn(h), rearrange(self.down_proj, "e o i -> e i o"), offs=offsets)
h = h * rearrange(topk_weights, "n k -> (n k) 1")[order].to(h.dtype)
return torch.zeros_like(x).index_add_(0, token_ids, h)
```

Read it as four moves. Flatten the $N \times k$ expert picks into $Nk$ pairs. Sort the pairs by expert id, so each expert's pairs form one contiguous run, with `order` remembering which pair went where and `token_ids` mapping each pair back to its token. Find each expert's run boundary with `searchsorted`; the comment in the code notes that `bincount` would work but syncs on CUDA, and `searchsorted` does not. Then the two grouped GEMMs run over the sorted rows, with `offs` telling each expert group where its rows end, and the activation between them. The routing weight multiplies each pair's row, and `index_add_` sums a token's $k$ rows back into one.

### The Triton path: one block, one expert

`fused_experts` (`lean_vllm/layers/fused_moe.py:117`) does the same computation with a Triton kernel on CUDA, the way vLLM's fused MoE does it. The win it is after: when a token's activation row is multiplied against one expert's weight, that weight can be loaded once and reused for the whole block of rows that share the expert. `align_blocks` (`lean_vllm/layers/fused_moe.py:91`) arranges exactly that. It sorts the pairs by expert, then pads each expert's run out to a whole number of `BLOCK_M` rows:

```python
# lean_vllm/layers/fused_moe.py:102-112, abridged
padded = (counts + block_m - 1) // block_m * block_m
num_blocks = (num_pairs + block_m - 1) // block_m + num_experts    # an upper bound
sorted_pairs[padded_starts[expert_of_pair] + ranks] = order.to(torch.int32)
block_experts = torch.searchsorted(padded_starts // block_m, blocks, right=True) - 1
```

Three details carry the design:

- `num_blocks` is computed from shapes, an upper bound of one wasted block per expert, never from data. `num_rows`, the count of rows that survived padding, is a device tensor. No count ever reaches the host, which is what lets this layer sit inside a CUDA graph (lesson 5).
- `sorted_pairs` holds, for every padded row, which pair it is, and `num_pairs` (out of range) for the padding overhang. The kernel masks those rows on both the load and the store, and blocks the padding left empty return early by reading `num_rows` (`lean_vllm/layers/fused_moe.py:49`).
- `block_experts` gives each block its expert with a `searchsorted` over the padded starts: a block belongs to the last expert starting at or before it, so an expert with no tokens owns no block.

`fused_moe_kernel` (`lean_vllm/layers/fused_moe.py:21`) is a standard tiled GEMM on top of that layout. The grid is blocks times output-column tiles. A block reads its expert id and its `BLOCK_M` pair indices, gathers the activation rows as `sorted_pairs // top_k` (or as themselves, in the second GEMM, because its input is already one row per pair), and loops the K dimension in `BLOCK_K` tiles into a float32 accumulator. The expert's weight tile is addressed once per block via the expert id, so it is loaded once and reused across the block's rows.

Two launches, one either side of the activation (`lean_vllm/layers/fused_moe.py:146-152`): the first GEMM computes `x @ gate_up_proj` with no routed weight, `SiluAndMul` runs on the pairs, and the second GEMM folds the routing weight into its epilogue via `MUL_ROUTED_WEIGHT`. A final `reduce(..., "sum")` over the $k$ pairs per token replaces the `index_add_`.

`use_triton` (`lean_vllm/layers/fused_moe.py:75`) picks the path: Triton on CUDA when it is importable, `grouped_mm` everywhere else, with `LEAN_VLLM_MOE_BACKEND` forcing either by name. The launch configuration (`config`, `lean_vllm/layers/fused_moe.py:86`) is a guess: `BLOCK_M` 16 for small batches, 64 from 256 pairs up. vLLM ships a tuned table per shape and dtype; this one awaits a GPU run to tune against.

### How the pieces interact

The layer's `forward` (`lean_vllm/models/deepseek_v2.py:153`) ties it together: `route` produces the picks, `FusedMoE.forward` (`lean_vllm/layers/moe.py:58`) consumes them on whichever path is active, and the shared experts add their output on top. Since `route` runs inside `post_attention` (`lean_vllm/models/deepseek_v2.py:191`), the whole MoE sits in the capturable part of the layer, which is why both paths above avoid every host-side read of a tensor's contents.

### Verified vs. not yet run

The `grouped_mm` path is the correctness reference and runs in the test suite. The Triton path's blocking is checked without a GPU: `tests/test_fused_moe.py` has `blocked_moe`, a torch transcription of what `align_blocks` and the kernel do with the layout, compared against `torch_experts`, plus checks that the alignment holds every pair exactly once and gives an empty expert no block. The arithmetic inside `tl.dot` is covered by `test_the_triton_kernel_matches_grouped_mm_on_cuda`, which runs `fused_experts` in bf16 against `grouped_mm` and passed on an H100 (commit `e7891ce`). The launch configuration is still untuned.

## Lesson 4: YaRN

One concept this lesson: YaRN stretches the rope past the length it was trained on, and it does so in two places, not one. The rope frequencies change, and the attention softmax scale changes with them. Both live in this repository, split across `lean_vllm/layers/rotary_embedding.py` and the attention layer's constructor.

### The problem YaRN solves

Recall rope as you know it: each pair of head dimensions rotates by angle $t \cdot \omega_i$ at position $t$, with $\omega_i = base^{-2i/d}$ falling geometrically from pair to pair, so attention scores depend only on relative position. The frequencies were chosen for a trained length $L$ = `original_max_position_embeddings`.

To serve a context $f$ times longer, the naive fix is position interpolation: divide every angle by $f$, so $f L$ positions map back into the trained range. That warps every frequency band equally, including the high-frequency pairs that encode local token structure, and quality drops. YaRN's fix is to treat frequency bands differently. A slow pair that never completed a full turn within $L$ would, past $L$, reach angles it never saw in training, so it must be interpolated. A fast pair has already seen every angle many times over, so it can extrapolate and keep its local resolution. YaRN interpolates the slow pairs, keeps the fast ones as they are, and ramps linearly in between.

### The frequencies, in code

`yarn_inv_freq` (`lean_vllm/layers/rotary_embedding.py:34`) implements that per-pair blend. First it marks the ramp's boundaries by counting rotations: pair $i$ completes

$$r_i = \frac{L \cdot \omega_i}{2\pi}$$

rotations over the original context, and `correction_dim` inverts that to a pair index:

$$\mathrm{dim}(r) = \frac{d \log\left(L / (2\pi r)\right)}{2 \log b}$$

Pairs below the `beta_fast` (32 rotations) boundary stay untouched; pairs above the `beta_slow` (1 rotation) boundary are fully interpolated; between the two, a `clamp`ed linear ramp blends. For each of the $d/2$ pairs, with $\theta_i = base^{2i/d}$:

$$\hat\omega_i = \frac{ramp_i}{f \, \theta_i} + \frac{1 - ramp_i}{\theta_i}$$

so `ramp` is 1 for the slow pairs (divided by the factor, i.e. interpolated) and 0 for the fast ones (kept). A pair that completes at most one turn over the original context, i.e. whose wavelength is at least $L$, is fully stretched; a pair that turns 32 or more times over it is left unchanged.

### The pairing: GPT-J style

`apply_rotary_emb` (`lean_vllm/layers/rotary_embedding.py:8`) has two layouts. NeoX style rotates the first half against the second half of the vector; GPT-J style rotates adjacent elements, `x[..., ::2]` against `x[..., 1::2]`, which is what DeepSeek's checkpoints use. `DeepseekV2Attention` therefore builds the rope with `is_neox_style=False` (`lean_vllm/models/deepseek_v2.py:58`). Both layouts use the same cos/sin table; they differ only in which elements are paired, so the flag has to match the checkpoint, or every rotation lands on the wrong pairs.

### The cache

`RotaryEmbedding` (`lean_vllm/layers/rotary_embedding.py:50`) precomputes, once at construction: the outer product of positions $0 \dots L_{max}$ with the adjusted frequencies, then `cos` and `sin` tables from it. Its `forward` (`lean_vllm/layers/rotary_embedding.py:84`, under `@torch.compile`) gathers a row per position from the cache and rotates the query and key through `apply_rotary_emb`. `get_rope` (`lean_vllm/layers/rotary_embedding.py:97`) memoizes the instance with `lru_cache(1)`, so every layer shares one table and a different config replaces it; a dict cannot key the cache, so its sorted items do.

In this model the rope's `rotary_dim` is `qk_rope_head_dim` = 64, not the full 192-dim head (`lean_vllm/models/deepseek_v2.py:54`). You know from lesson 1 why: only the shared rope key carries positions, and only that slice is rotated, in `project()` (`lean_vllm/models/deepseek_v2.py:103`), before the latent is cached. YaRN therefore adjusts 64 of the 576 cached values per token.

### mscale, twice

Interpolation also changes how sharp the attention distribution is: stretched positions make the softmax scores flatter than training saw, and YaRN compensates with a temperature factor $m$:

```python
# lean_vllm/layers/rotary_embedding.py:30-31
def yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0
```

The two touchpoints:

1. The cos/sin tables. Inside `RotaryEmbedding.__init__`, the yarn branch multiplies `cos` and `sin` by `attention_factor` from the config, or by the ratio $\mathrm{mscale}(f, m) / \mathrm{mscale}(f, m_{all})$ when that is absent. Per the design doc this ratio is 1 for V2-Lite, so the tables are unmodified in practice (`docs/deepseek-v2.md`, "YaRN").
2. The softmax scale, in the attention layer:

```python
# lean_vllm/models/deepseek_v2.py:61-64
scaling = self.qk_head_dim ** -0.5
if rope_scaling.get("mscale_all_dim"):    # YaRN also sharpens the softmax
    mscale = yarn_get_mscale(rope_scaling["factor"], rope_scaling["mscale_all_dim"])
    scaling *= mscale * mscale
```

The base scale is the usual $1/\sqrt{d}$ with $d = 192$, and the YaRN factor multiplies it by $m^2$. The doc gives the number: about 1.59 at V2-Lite's factor of 40. This `scaling` is what `MLAAttention` passes to its backend as the softmax scale, so every attention call in lessons 1 and 2 uses it, including the `mla_decode` path.

That last case is worth a second look. In latent space the query is 576 wide, yet the scale stays $m^2/\sqrt{192}$ rather than being recomputed from 576. It must: $(W_k^{\top} q_{nope}) \cdot c + q_{pe} \cdot k_{pe}$ is the same number as the original 192-dim $q \cdot k$, so it takes the same scale.

The numbers come from V2-Lite's `config.json`: `factor` 40, `mscale` and `mscale_all_dim` both 0.707. The ratio in the cos/sin tables is therefore 1, and $m = 1 + 0.1 \cdot 0.707 \cdot \ln 40 \approx 1.261$, so $m^2 \approx 1.59$.

### Verified vs. not yet run

Both touchpoints are regression-covered: the doc's mutation list for `tests/test_deepseek_v2.py` includes "a dropped softmax mscale" and "a dropped cos/sin attention factor", so removing either fails the comparison against transformers. The checks run on tiny random checkpoints in fp32 on the torch backend; nothing YaRN-specific needs a GPU.

## Lesson 5: CUDA graphs

One concept this lesson: the same model runs in three modes, chosen per step, and MLA's design from lessons 1 and 2 decides what each captured mode can hold. A decode step is small, often a handful of tokens, so the CPU time spent launching kernels rivals the GPU time computing them. A CUDA graph records a launch sequence once and replays it. This repository captures it at two granularities, with eager execution as the fallback.

### Choosing the mode

At startup (`lean_vllm/engine/model_runner.py:47-50`) the runner forces eager when any of these hold: `enforce_eager` is configured, the device is not CUDA, the backend answers `supports_cuda_graph()` false (the torch reference does, `lean_vllm/attention/torch_backend.py:21-22`), or the model class does. `DeepseekV2ForCausalLM` answers true (`lean_vllm/models/deepseek_v2.py:236`). Otherwise the configured mode stands: `none`, `full`, `piecewise`, or `full_and_piecewise`, the default (`lean_vllm/config.py:13-15`, `:29`), after one MLA-specific filter, `_cudagraph_mode` (`lean_vllm/engine/model_runner.py:232`):

```python
# lean_vllm/engine/model_runner.py:236-239, abridged
full_safe = backend.supports_mla_decode() and backend.supports_full_cudagraph_mla_decode()
if mla and mode in FULL_MODES and not full_safe:
    return "piecewise" if mode in PIECEWISE_MODES else "none"
return mode
```

The second flag is the finer switch. `supports_full_cudagraph_mla_decode` (`lean_vllm/attention/abstract.py:38-41`, default true) is false when an `mla_decode` bakes per-step state a replay cannot refresh. FlashMLA answers true (`lean_vllm/attention/flashmla_backend.py:31-38`): the current build computes its tile schedule and split-KV workspace inside the kernel from `context_lens`, so the capture runs with worst-case lengths and each replay re-gates the KV loop on the tensors it refreshes. No backend currently answers false; the flag exists for builds where that schedule was computed outside the kernel, as the comments describe.

`_cudagraph_mode` never names a backend: it reads the two flags, so FlashMLA's answer alone decides whether its decode joins the full graph. The startup log line ("MLA decode cannot run in a full graph") fires only when the mode was actually downgraded.

### Which steps replay where

`_step_kind` (`lean_vllm/engine/model_runner.py:241`) picks per step. A full graph serves only pure decode, when `not is_prefill` and the batch fits a captured size. A piecewise graph serves any step whose token count lands in a bucket, 64 to 512 tokens. Everything else runs eager, recorded as "prefill" or "decode" by the step's kind; that includes small prefill and mixed steps under 64 tokens, which fall below the smallest bucket. Lesson 2's rule that a one-token prefill is still a prefill matters here: it keeps such rows off the full-graph path, whose captured attention assumes decode.

### The full graph: attention inside the replay

`capture_cudagraph` (`lean_vllm/engine/model_runner.py:389`) records the whole model, for batch sizes `[1, 2, 4, 8]` and then every 16 up to `min(max_num_seqs, 512)` (`:400`), largest first, into one shared pool. The inputs and outputs live in persistent tensors (`graph_vars`, `:420`) that `_replay_full` (`:262`) refreshes before each `graph.replay()`.

What a replay forbids is the whole MLA story of this mode. A replayed graph re-runs recorded kernel launches; it cannot run host code, and it cannot pick new shapes. Of lesson 2's three attention paths, only `_decode_latents` qualifies: its launches depend on the batch size and on tensors the replay refreshes, nothing else. The chunked-context path fails twice, since `plan_context_chunks` builds Python lists on the host and expansion allocates per-chunk shapes from that plan. The dispatch in `attend()` is a Python branch, baked at capture with `is_prefill=False`; only pure-decode steps ever replay, so the baked branch is the right one every time. That is the precise sense in which the full graph requires `mla_decode`.

The padded tail is the second constraint. A replay runs on a bucket possibly larger than the batch, so the latent store must skip the padded rows without a host-side mask, which would sync. `_replay_full` fills the slot map with -1 and copies the real slots over it (`:270-271`), and FlashMLA's Triton `store_latents_kernel` returns early on a -1 slot (`lean_vllm/attention/flash_backend.py:50`). The torch reference masks on the host instead (`lean_vllm/attention/torch_backend.py:43-50`), which is acceptable because that backend never captures.

Two capture-time details close the mode. `context_lens` is filled with `max_model_len` (`:404`), the worst case the FlashMLA schedule and workspace are sized against. And between warmup and capture the runner clears `mla_decode_metadata` (`:412`): the warmup pass scheduled the decode into the default memory pool, and the capture must reschedule into the graph's own pool, or the graph bakes pointers that are freed with the warmup context.

### Piecewise: attention stays eager

`capture_piecewise` (`lean_vllm/engine/model_runner.py:328`) captures everything except attention, in pieces: the embedding head; per layer, `pre_attention`; per layer, `post_attention`; and the final norm. The buckets run from 64 to `min(512, max_num_batched_tokens)` tokens, with no step padded more than a quarter (`_piecewise_buckets`, `:317`; constants at `:25-28`).

The buffers are shaped from the model rather than assumed. The runner calls `layers[0].pre_attention` on zero-filled buffers and keeps whatever comes back (`:343-344`): a query and a latent for MLA, three tensors for Qwen3. The attention output buffer comes from `Attention.output_shape`, where `MLAAttention` answers `v_head_dim` against the base class's `head_dim` (`lean_vllm/layers/attention.py:225-227`).

Replay (`_replay_piecewise`, `:283`) copies the real rows into the buffers and runs the pieces around attention, which executes eager, on real rows only:

```python
# lean_vllm/engine/model_runner.py:291-297, abridged
pre.replay()
attn_out = layer.self_attn.attn(*(buffer[:num_tokens] for buffer in buffers["attn_in"]))
buffers["attn_out"][:num_tokens] = attn_out
post.replay()
```

Because attention is eager, all of lesson 2 applies unchanged, chunked context and mixed steps included. The MoE sits inside `post_attention`, which both graph modes capture: the full graph records it with the rest of the model, and piecewise records it as the `post` piece. That is what pushed lesson 3's design: block counts from shapes, the surviving-row count kept on the device, the early return inside the kernel. Nothing the MoE runs reads a tensor's contents on the host.

The eager call still goes through the custom op: `layer.self_attn.attn(...)` calls the `MLAAttention` module, whose `forward` is `torch.ops.lean_vllm.mla_attention` (`lean_vllm/layers/attention.py:228-229`). The op plays no part in the capture itself: the runner never compiles the model, and it splits the graph around attention by hand, capturing `pre_attention` and `post_attention` as separate pieces. The op's opacity (`:86`) would matter only to a `torch.compile` of the whole model.

### Verified vs. not yet run

Both graph modes have run on an H100: the 20 September benchmark (`docs/benchmark-2026-09-20.md`) served V2-Lite in `full_and_piecewise` mode with FlashMLA decode inside the full graph, and no step ran as eager decode. That run measured speed, not numerical agreement with the torch reference. What is regression-checked on the torch backend is the eager path those modes fall back from, the mixed-row partition of lesson 2 included.