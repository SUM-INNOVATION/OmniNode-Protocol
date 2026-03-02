import sys
import time
from omninode import OmniNet, NetConfig

# Grab role from command line argument (default to receiver)
node_role = sys.argv[1] if len(sys.argv) > 1 else "receiver"
print(f"🚀 Starting Node as: {node_role.upper()}")

# Initialize the Rust networking engine
net = OmniNet(NetConfig())
print("⏳ Waiting for mDNS discovery...")

target_peer = None

while True:
    ev = net.next_event(timeout_secs=0.5)
    
    # If no event, check if we are the sender and ready to fire
    if not ev:
        if target_peer and node_role == "sender":
            print(f"\n🎯 Target acquired: {target_peer}")
            print("📦 Generating 32MB dummy activation tensor (F16, seq=2048, dim=8192)...")
            
            # Create a 32MB payload (simulating a 70B model's hidden state)
            dummy_data = bytes([0x42] * (2048 * 8192 * 2))
            
            print("📤 Firing tensor over QUIC mesh...")
            t0 = time.time()
            net.request_tensor(
                peer_id=target_peer,
                session_id="demo-session-001",
                micro_batch_index=0,
                from_stage=0,
                to_stage=1,
                seq_len=2048,
                hidden_dim=8192,
                dtype=0, # 0 = F16
                data=dummy_data
            )
            # Clear target so we only fire once
            target_peer = None 
        continue

    # Handle Network Events
    if ev.kind == "peer_discovered":
        print(f"👋 Discovered peer: {ev.peer_id}")
        if node_role == "sender" and target_peer is None:
            # We found a peer! Set it as the target (the empty loop above will fire the tensor)
            target_peer = ev.peer_id

    elif ev.kind == "tensor_received":
        print(f"\n📥 Incoming Tensor Received from {ev.peer_id}!")
        print(f"   Session: {ev.session_id} | Micro-batch: {ev.micro_batch_index}")
        print(f"   Routing: Stage {ev.from_stage} -> Stage {ev.to_stage}")
        print(f"   Shape: [{ev.seq_len}, {ev.hidden_dim}] | Bytes: {len(ev.data)}")
        
        print("✅ Sending acknowledgment...")
        net.respond_tensor(
            channel_id=ev.channel_id,
            session_id=ev.session_id,
            micro_batch_index=ev.micro_batch_index,
            stage_index=ev.to_stage,
            accepted=True
        )
        print("🎉 Receiver test complete! You can press Ctrl+C.")

    elif ev.kind == "tensor_response_received":
        print(f"\n✅ Tensor Acknowledged by {ev.peer_id}!")
        print(f"   Accepted: {ev.accepted}")
        print("🎉 Sender test complete! You can press Ctrl+C.")

    elif ev.kind == "tensor_request_failed":
        print(f"\n❌ Tensor transfer failed: {ev.error}")
