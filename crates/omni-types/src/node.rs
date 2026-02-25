// Phase 1 stub — minimal NodeCapability for mDNS/Gossipsub.
// Extended in Phase 4 with layer ranges, VRAM, FLOPS, etc.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapability {
    pub peer_id:   String,
    pub ram_bytes: u64,
    pub platform:  Platform,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Platform {
    AppleSilicon,
    Linux,
    Windows,
}
