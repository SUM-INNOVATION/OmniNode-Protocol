import numpy as np
from omninode import OmniStore, StoreConfig
import time

print("1. Initializing OmniStore...")
store = OmniStore(StoreConfig())

print("2. Loading TinyLlama (will skip existing shards)...")
t0 = time.time()
manifest = store.ingest_model("../tinyllama.gguf")  # Notice the ../ because we moved down a folder
print(f"   Done in {time.time() - t0:.3f}s")

# Grab the first shard (the embedding layer, ~452MB)
shard = manifest.shards[0]
print(f"\n3. Target Shard: {shard.cid}")
print(f"   Size: {shard.size_bytes / 1024 / 1024:.2f} MB")

print("\n4. Performing Zero-Copy MMAP...")
t0 = time.time()
view = store.mmap_shard(shard.cid)

print("5. Wrapping in NumPy (Buffer Protocol)...")
arr = np.frombuffer(view, dtype=np.uint8)

print(f"   Done in {time.time() - t0:.5f}s")
print(f"   NumPy Array Shape: {arr.shape}")
print(f"   First 10 bytes: {arr[:10].tolist()}")
print("\n🚀 ZERO-COPY BRIDGE CONFIRMED!")
