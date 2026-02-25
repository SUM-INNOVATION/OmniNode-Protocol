// Global configuration structs. Populated as each phase adds settings.
#[derive(Debug, Clone)]
pub struct NetConfig {
    /// UDP port for QUIC listener. 0 = OS-assigned.
    pub listen_port: u16,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self { listen_port: 0 }
    }
}
