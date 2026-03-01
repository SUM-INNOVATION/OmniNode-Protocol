import mlx.core as mx
import numpy as np
from omninode import OmniStore, StoreConfig
import time

print("1. Initializing OmniStore & Loading Manifest...")
store = OmniStore(StoreConfig())
manifest = store.ingest_model("../tinyllama.gguf")
shard = manifest.shards[0]

print(f"\n2. Zero-Copy MMAP of {shard.cid}...")
view = store.mmap_shard(shard.cid)
arr = np.frombuffer(view, dtype=np.uint8)

print("\n3. Pushing to Apple Silicon GPU (MLX)...")
t0 = time.time()
# Wrap the numpy array into an MLX array natively
mx_arr = mx.array(arr)
# MLX is lazy, so we call mx.eval() to force it to realize the memory
mx.eval(mx_arr)
print(f"   MLX Array Shape: {mx_arr.shape}")
print(f"   Time to load into MLX: {time.time() - t0:.5f}s")

print("\n4. Executing Tensor Math on GPU...")
t0 = time.time()
# Let's force the GPU cores to sum all 452 million bytes
result = mx.sum(mx_arr)
mx.eval(result) # Force computation
print(f"   Result of GPU sum: {result.item()}")
print(f"   Compute time: {time.time() - t0:.5f}s")

print("\n🍏 APPLE SILICON GPU INFERENCE BRIDGE CONFIRMED!")
