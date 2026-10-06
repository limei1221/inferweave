_IMPORT_ERROR: ImportError | None = None
try:
    import triton
    import triton.language as tl
except ImportError as e:  # installed by the cuda extra
    _IMPORT_ERROR = e
else:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    @triton.jit
    def store_latents_kernel(
        latent_ptr,
        latent_stride,
        cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        offsets = tl.arange(0, BLOCK)
        mask = offsets < D  # a latent is 576 wide for V2-Lite, so the block overhangs it
        latent = tl.load(latent_ptr + idx * latent_stride + offsets, mask=mask)
        tl.store(cache_ptr + slot * D + offsets, latent, mask=mask)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping) -> None:
    """Scatter keys/values [num_tokens, heads, dim] into [num_blocks, block_size, heads, dim] caches. Slot -1 skips."""
    num_tokens, num_heads, head_dim = key.shape
    dim = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == dim and v_cache.stride(1) == dim
    assert slot_mapping.numel() == num_tokens
    store_kvcache_kernel[(num_tokens,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, dim)


def store_latents(latent, latent_cache, slot_mapping) -> None:
    """Scatter latents [num_tokens, dim] into a [num_blocks, block_size, dim] cache. Slot -1 skips."""
    num_tokens, dim = latent.shape
    assert latent.stride(-1) == 1 and latent_cache.stride(-2) == dim
    assert slot_mapping.numel() == num_tokens
    store_latents_kernel[(num_tokens,)](
        latent, latent.stride(0), latent_cache, slot_mapping, dim, triton.next_power_of_2(dim)
    )
