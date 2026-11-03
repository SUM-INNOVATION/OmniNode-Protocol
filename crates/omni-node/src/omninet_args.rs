//! OmniNet routing controls on the command line (issue #158).
//!
//! Every command that opens OmniNet flattens [`OmniNetControlArgs`] into its
//! arguments, so `--mdns` and `--outbound-allow` mean the same thing, with the
//! same defaults, wherever they appear. The names and the entry syntax are
//! SNIP's (`sum-node --mdns`, `--outbound-allow`, SNIP #79), so one allow-list
//! reads the same in both stacks.
//!
//! Each flag can also come from an environment variable ([`MDNS_ENV`],
//! [`OUTBOUND_ALLOW_ENV`]); a flag on the command line takes precedence. The
//! names are a recommendation the owner has not ratified, so they live in
//! these two constants only. They are OmniNet's own on purpose: SNIP's
//! `SUM_MDNS` / `SUM_OUTBOUND_ALLOW` configure the `sum-node` children and
//! carry addresses meant for SNIP archives, not for OmniNet peers.

use clap::{Args, ValueEnum};
use omni_types::config::NetConfig;

/// Environment variable for `--mdns` (recommended name, not yet ratified).
pub(crate) const MDNS_ENV: &str = "OMNI_MDNS";

/// Environment variable for `--outbound-allow`, comma-separated (recommended
/// name, not yet ratified).
pub(crate) const OUTBOUND_ALLOW_ENV: &str = "OMNI_OUTBOUND_ALLOW";

/// `--mdns` and `--outbound-allow`.
#[derive(Args, Clone, Debug, PartialEq, Eq)]
pub(crate) struct OmniNetControlArgs {
    /// OmniNet mDNS LAN discovery: `on` (default) or `off`.
    ///
    /// `off` builds no mDNS at all: OmniNet neither announces itself nor
    /// discovers LAN peers by multicast, so it reaches peers only through
    /// `--peer` and Kademlia. It does not affect the `sum-node` children this
    /// command runs; they take SNIP's own `SUM_MDNS`.
    #[arg(long, env = MDNS_ENV, default_value = "on", value_enum)]
    pub mdns: MdnsMode,

    /// OmniNet outbound address allow-list (repeatable or comma-separated).
    /// Unset (default): no restriction.
    ///
    /// When set, every address OmniNet dials, whichever part of it asks
    /// (`--peer`, mDNS, Identify, Kademlia, AutoNAT, relay, DCUtR), must match
    /// an entry; anything else is refused before a packet is sent. Entries: an
    /// IP or CIDR (`127.0.0.1`, `10.0.0.0/8`, `::1`) or a multiaddr prefix
    /// (`/ip4/127.0.0.1/udp/4001/quic-v1`, optionally ending in
    /// `/p2p/<peer-id>`). Inbound connections from addresses outside the list
    /// are refused, and refused addresses are kept out of the peer stores. An
    /// application-level routing control, not an OS security boundary.
    #[arg(
        long = "outbound-allow",
        env = OUTBOUND_ALLOW_ENV,
        value_name = "ENTRY",
        value_delimiter = ',',
        value_parser = parse_outbound_allow_entry
    )]
    pub outbound_allow: Vec<String>,
}

/// `--mdns`. `true`/`false`, `1`/`0` and `yes`/`no` are accepted too, as for
/// SNIP's `--mdns`.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MdnsMode {
    #[value(alias = "true", alias = "1", alias = "yes")]
    On,
    #[value(alias = "false", alias = "0", alias = "no")]
    Off,
}

impl Default for OmniNetControlArgs {
    /// What the flags parse to when neither is given.
    fn default() -> Self {
        Self {
            mdns: MdnsMode::On,
            outbound_allow: Vec::new(),
        }
    }
}

impl OmniNetControlArgs {
    /// Carry the flags into `config`: mDNS on or off, and the allow-list.
    pub(crate) fn apply(&self, config: NetConfig) -> NetConfig {
        NetConfig {
            mdns: self.mdns == MdnsMode::On,
            outbound_allow: self.outbound_allow.clone(),
            ..config
        }
    }

    /// The default `NetConfig` with these flags applied.
    pub(crate) fn net_config(&self) -> NetConfig {
        self.apply(NetConfig::default())
    }
}

/// Validate one `--outbound-allow` entry at parse time, so a malformed list is
/// a usage error before anything networks. The string is kept as given;
/// `omni-net` parses it again when it builds the swarm. A blank item (from a
/// trailing comma) is kept and ignored there.
fn parse_outbound_allow_entry(raw: &str) -> Result<String, String> {
    if raw.trim().is_empty() {
        return Ok(String::new());
    }
    raw.parse::<omni_net::policy::AllowEntry>()
        .map(|_| raw.trim().to_string())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Root {
        #[command(flatten)]
        net: OmniNetControlArgs,
    }

    fn parse(args: &[&str]) -> Result<OmniNetControlArgs, clap::Error> {
        let mut argv = vec!["omni-node"];
        argv.extend_from_slice(args);
        Root::try_parse_from(argv).map(|r| r.net)
    }

    #[test]
    fn no_flags_keep_todays_defaults() {
        let net = parse(&[]).unwrap();
        assert_eq!(net, OmniNetControlArgs::default());
        let cfg = net.net_config();
        let default = NetConfig::default();
        assert!(cfg.mdns && default.mdns, "mDNS stays on");
        assert!(cfg.outbound_allow.is_empty() && default.outbound_allow.is_empty());
        assert_eq!(cfg.listen_port, default.listen_port);
        assert_eq!(cfg.bootstrap_peers, default.bootstrap_peers);
        assert_eq!(cfg.relay_server, default.relay_server);
    }

    #[test]
    fn mdns_takes_on_off_and_the_boolean_spellings() {
        for off in ["off", "false", "0", "no"] {
            assert_eq!(
                parse(&["--mdns", off]).unwrap().mdns,
                MdnsMode::Off,
                "{off}"
            );
        }
        for on in ["on", "true", "1", "yes"] {
            assert_eq!(parse(&["--mdns", on]).unwrap().mdns, MdnsMode::On, "{on}");
        }
        assert!(parse(&["--mdns", "maybe"]).is_err());
        assert!(!parse(&["--mdns", "off"]).unwrap().net_config().mdns);
    }

    #[test]
    fn outbound_allow_is_repeatable_and_comma_separated() {
        let net = parse(&[
            "--outbound-allow",
            "127.0.0.1,10.0.0.0/8",
            "--outbound-allow",
            "/ip4/192.0.2.7/udp/4001/quic-v1",
        ])
        .unwrap();
        assert_eq!(
            net.outbound_allow,
            vec!["127.0.0.1", "10.0.0.0/8", "/ip4/192.0.2.7/udp/4001/quic-v1"]
        );
        assert_eq!(net.net_config().outbound_allow, net.outbound_allow);
    }

    #[test]
    fn a_malformed_outbound_allow_entry_is_a_usage_error() {
        for bad in ["not-an-ip", "10.0.0.0/40", "/memory/1"] {
            let err = parse(&["--outbound-allow", bad]).unwrap_err();
            assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation, "{bad}");
            assert!(err.to_string().contains("--outbound-allow"), "{err}");
        }
    }

    #[test]
    fn apply_keeps_every_other_field() {
        let net = parse(&["--mdns", "off", "--outbound-allow", "127.0.0.1"]).unwrap();
        let cfg = net.apply(NetConfig {
            listen_port: 4242,
            bootstrap_peers: vec!["/ip4/127.0.0.1/udp/1/quic-v1".into()],
            ..NetConfig::default()
        });
        assert_eq!(cfg.listen_port, 4242);
        assert_eq!(cfg.bootstrap_peers.len(), 1);
        assert!(!cfg.mdns);
        assert_eq!(cfg.outbound_allow, vec!["127.0.0.1"]);
    }

    // ── Environment variables ──────────────────────────────────────────
    //
    // Process environment is shared by every test thread, so each case runs
    // in a child process of this test binary with its own environment:
    // `env_case` does the parsing and asserting there.

    const CASE: &str = "OMNI_NODE_ENV_TEST_CASE";

    /// Run `env_case` in a child with `case` and `vars` (and no other
    /// `OMNI_*` routing variable), and assert it passed.
    fn run_env_case(case: &str, vars: &[(&str, &str)]) {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "--exact",
            "omninet_args::tests::env_case",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CASE, case)
        .env_remove(MDNS_ENV)
        .env_remove(OUTBOUND_ALLOW_ENV);
        for (k, v) in vars {
            cmd.env(k, v);
        }
        let out = cmd.output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "env case {case} failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// The child side. Does nothing unless a case is set.
    #[test]
    fn env_case() {
        let Ok(case) = std::env::var(CASE) else {
            return;
        };
        match case.as_str() {
            "env-sets-both" => {
                let net = parse(&[]).unwrap();
                assert_eq!(net.mdns, MdnsMode::Off);
                assert_eq!(
                    net.outbound_allow,
                    vec!["127.0.0.1", "/ip4/192.0.2.7/udp/4001/quic-v1"]
                );
                let cfg = net.net_config();
                assert!(!cfg.mdns);
                assert_eq!(cfg.outbound_allow.len(), 2);
            }
            "flags-win-over-env" => {
                let net = parse(&["--mdns", "on", "--outbound-allow", "10.0.0.0/8"]).unwrap();
                assert_eq!(net.mdns, MdnsMode::On);
                assert_eq!(net.outbound_allow, vec!["10.0.0.0/8"]);
            }
            "env-boolean-spelling" => {
                assert_eq!(parse(&[]).unwrap().mdns, MdnsMode::Off);
            }
            "env-malformed-entry" => {
                let err = parse(&[]).unwrap_err();
                assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
                assert!(err.to_string().contains("--outbound-allow"), "{err}");
            }
            "env-unset" => {
                assert_eq!(parse(&[]).unwrap(), OmniNetControlArgs::default());
            }
            other => panic!("unknown case {other}"),
        }
    }

    #[test]
    fn the_environment_sets_both_controls() {
        run_env_case(
            "env-sets-both",
            &[
                (MDNS_ENV, "off"),
                (
                    OUTBOUND_ALLOW_ENV,
                    "127.0.0.1,/ip4/192.0.2.7/udp/4001/quic-v1",
                ),
            ],
        );
    }

    #[test]
    fn command_line_flags_take_precedence_over_the_environment() {
        run_env_case(
            "flags-win-over-env",
            &[(MDNS_ENV, "off"), (OUTBOUND_ALLOW_ENV, "127.0.0.1")],
        );
    }

    #[test]
    fn the_environment_takes_the_boolean_spellings() {
        run_env_case("env-boolean-spelling", &[(MDNS_ENV, "false")]);
    }

    #[test]
    fn a_malformed_entry_in_the_environment_is_a_usage_error() {
        run_env_case(
            "env-malformed-entry",
            &[(OUTBOUND_ALLOW_ENV, "not-an-address")],
        );
    }

    #[test]
    fn without_the_environment_the_defaults_hold() {
        run_env_case("env-unset", &[]);
    }

    #[test]
    fn the_variable_names_are_omninets_own() {
        assert_eq!(
            (MDNS_ENV, OUTBOUND_ALLOW_ENV),
            ("OMNI_MDNS", "OMNI_OUTBOUND_ALLOW")
        );
    }

    /// The flags reach the swarm: `--mdns off` builds no mDNS behaviour, and
    /// the default builds it, as before.
    #[tokio::test]
    async fn mdns_off_reaches_the_swarm_as_no_mdns_behaviour() {
        let off = parse(&["--mdns", "off"]).unwrap().net_config();
        assert!(
            !omni_net::swarm::OmniSwarm::build(&off)
                .unwrap()
                .mdns_enabled()
        );
        let on = parse(&[]).unwrap().net_config();
        assert!(
            omni_net::swarm::OmniSwarm::build(&on)
                .unwrap()
                .mdns_enabled()
        );
    }
}
