//! Issue #149 — end to end: `run-job` publishes its response bytes
//! idempotently, against a real SUM Chain devnet and real SNIP archives.
//!
//! Ignored by default; it needs three prebuilt upstream binaries:
//!
//! ```text
//! OMNI_E2E_SUMCHAIN=<sum-chain>/target/release/sumchain
//! OMNI_E2E_SETUP_LOCAL_TESTNET=<sum-chain>/target/release/setup-local-testnet
//! OMNI_E2E_SUM_NODE=<SNIP>/target/release/sum-node
//! cargo test -p omni-node --test snip_publication_e2e -- --ignored --nocapture
//! ```
//!
//! Synthetic and local only. `setup-local-testnet --mode smoke` generates a
//! one-validator devnet (chain id 1337, SNIP V2 from genesis) and throwaway
//! funded keys in a fresh temporary directory; the validator's RPC, P2P and
//! health endpoints are bound to 127.0.0.1. Three `sum-node listen` archives
//! are registered with stake and serve it; SNIP's processes find each other
//! by mDNS, as in SNIP's own local-mirror harness (`sum-node` has no option
//! to bind only loopback). No key is printed. Every process is killed and the
//! directory deleted at the end, pass or fail (`OMNI_E2E_KEEP=1` keeps the
//! directory for inspection).
//!
//! Scenarios, each with its own response bytes unless stated:
//!
//! 1. the same job run twice: `ingested`, then `reused_active`;
//! 2. a different job with the same response bytes: `reused_active`;
//! 3. the chain RPC lost right after the response's registration lands
//!    (issue #149's observed case): no result, the root left Pending, SNIP's
//!    reason in the error; the retry `resumed`;
//! 4. the run killed right after the registration lands: no result; the retry
//!    `resumed`;
//! 5. the result write failing after activation: the retry `reused_active`;
//! 6. conflicting metadata (a registration of the root with another size),
//!    by another account and by the configured one: refused;
//! 7. the root Pending, registered by another account: refused;
//! 8. the chain RPC unreachable from the start (the run fails at its input
//!    download), and lost after the runner, just before publication (the
//!    chain lookup fails; nothing is ingested): no result, nothing registered.
//!
//! Every produced result passes `verify-result` (`overall=ok`); the response
//! fetched back by a fresh `sum-node download` hashes to the signed
//! `response_hash`; the validator's log shows every root registered in
//! exactly one block and no refused duplicate registration anywhere; the chain's
//! record of a root keeps its `created_at` across retries; and no failing run
//! leaves a result file.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use omni_contributor::canonical::{hex_lower, job_hash_hex};
use omni_contributor::{
    BaseUnitRewardPolicy, ContributorJob, ContributorResult, ContributorSigner, JobAccounting,
    VerificationRequirement,
};
use omni_store::snip_v2_publish::{SnipV2FileInfo, public_object_root};
use omni_store::{SnipV2Cli, SnipV2CliConfig};
use omni_types::phase5::SnipV2ObjectId;

const CHAIN_ID: u64 = 1337;
/// `sum-node register-node` default stake.
const ARCHIVE_STAKE: &str = "1000000000";
/// `sum-node`'s default per-transaction fee (`--attest-fee`).
const TX_FEE: u128 = 1_000_000;
/// Token counts the stub runner reports, and the job declares.
const INPUT_TOKENS: u64 = 16;
const OUTPUT_TOKENS: u64 = 8;
/// The contributor's result-signing seed (synthetic).
const CONTRIBUTOR_SEED: [u8; 32] = [0x5a; 32];

// ── Small utilities ──────────────────────────────────────────────────────────

fn env_path(name: &str) -> PathBuf {
    let v = std::env::var_os(name)
        .unwrap_or_else(|| panic!("{name} must name the prebuilt binary (see the module docs)"));
    let p = PathBuf::from(v);
    assert!(p.is_file(), "{name}={} is not a file", p.display());
    p
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn wait_until<T>(what: &str, timeout: Duration, mut f: impl FnMut() -> Option<T>) -> T {
    let start = Instant::now();
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(
            start.elapsed() < timeout,
            "timed out after {timeout:?} waiting for {what}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for n in chars.by_ref() {
                    if ('@'..='~').contains(&n) {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

fn rpc(url: &str, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
    let agent = ureq::Agent::config_builder()
        .proxy(None)
        .timeout_global(Some(Duration::from_secs(10)))
        .build()
        .new_agent();
    let body = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        .to_string();
    let text = agent
        .post(url)
        .header("content-type", "application/json")
        .send(&body)
        .map_err(|e| e.to_string())?
        .body_mut()
        .read_to_string()
        .map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
        return Err(e.to_string());
    }
    Ok(v.get("result").cloned().unwrap_or(serde_json::Value::Null))
}

fn output_text(o: &Output) -> (String, String) {
    (
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn kv<'a>(stdout: &'a str, key: &str) -> Option<&'a str> {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

// ── Processes ────────────────────────────────────────────────────────────────

/// A long-running child, killed (with its process group) on drop.
struct Proc {
    name: String,
    child: Child,
}

impl Proc {
    fn spawn(name: &str, mut cmd: Command, log: &Path) -> Proc {
        use std::os::unix::process::CommandExt;
        let f = std::fs::File::create(log).unwrap();
        cmd.stdin(Stdio::null())
            .stdout(f.try_clone().unwrap())
            .stderr(f)
            .process_group(0);
        let child = cmd.spawn().unwrap_or_else(|e| panic!("spawn {name}: {e}"));
        Proc {
            name: name.to_string(),
            child,
        }
    }

    fn kill_group(&mut self) {
        let pgid = self.child.id();
        let _ = Command::new("kill")
            .args(["-KILL", &format!("-{pgid}")])
            .stderr(Stdio::null())
            .status();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        self.kill_group();
        eprintln!("[e2e] stopped {}", self.name);
    }
}

// ── A switchable loopback TCP proxy in front of the chain RPC ────────────────

/// Forwards connections to `upstream` while up; while down it refuses new
/// connections (accepts and closes them) and has shut every open one.
struct RpcProxy {
    url: String,
    up: Arc<AtomicBool>,
    conns: Arc<Mutex<Vec<TcpStream>>>,
}

impl RpcProxy {
    fn start(upstream: std::net::SocketAddr) -> RpcProxy {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let up = Arc::new(AtomicBool::new(true));
        let conns: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));
        let (up2, conns2) = (up.clone(), conns.clone());
        std::thread::spawn(move || {
            for client in listener.incoming() {
                let Ok(client) = client else { continue };
                if !up2.load(Ordering::SeqCst) {
                    let _ = client.shutdown(Shutdown::Both);
                    continue;
                }
                let Ok(server) = TcpStream::connect(upstream) else {
                    let _ = client.shutdown(Shutdown::Both);
                    continue;
                };
                {
                    let mut c = conns2.lock().unwrap();
                    c.push(client.try_clone().unwrap());
                    c.push(server.try_clone().unwrap());
                }
                for (mut from, mut to) in [
                    (client.try_clone().unwrap(), server.try_clone().unwrap()),
                    (server, client),
                ] {
                    std::thread::spawn(move || {
                        let mut buf = [0u8; 16384];
                        loop {
                            match from.read(&mut buf) {
                                Ok(0) | Err(_) => break,
                                Ok(n) => {
                                    if to.write_all(&buf[..n]).is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                        let _ = to.shutdown(Shutdown::Both);
                    });
                }
            }
        });
        RpcProxy { url, up, conns }
    }

    fn cut(&self) {
        self.up.store(false, Ordering::SeqCst);
        for c in self.conns.lock().unwrap().drain(..) {
            let _ = c.shutdown(Shutdown::Both);
        }
    }

    fn restore(&self) {
        self.up.store(true, Ordering::SeqCst);
    }
}

// ── The devnet ───────────────────────────────────────────────────────────────

struct Devnet {
    dir: PathBuf,
    rpc_url: String,
    rpc_addr: std::net::SocketAddr,
    validator_log: PathBuf,
    sum_node: PathBuf,
    omni_node: PathBuf,
    /// Role → `sum-node --key-file` (hex seed).
    key_files: BTreeMap<String, PathBuf>,
    /// Role → seed.
    seeds: BTreeMap<String, [u8; 32]>,
    /// Kept last so they stop before the directory goes.
    procs: Vec<Proc>,
}

impl Devnet {
    fn start(dir: &Path) -> Devnet {
        let sumchain = env_path("OMNI_E2E_SUMCHAIN");
        let setup = env_path("OMNI_E2E_SETUP_LOCAL_TESTNET");
        let sum_node = env_path("OMNI_E2E_SUM_NODE");
        let omni_node = PathBuf::from(env!("CARGO_BIN_EXE_omni-node"));

        // 1. Provision: throwaway keys and genesis, in this run's directory.
        let devnet = dir.join("devnet");
        std::fs::create_dir(&devnet).unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&devnet, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let out = Command::new(&setup)
            .args(["--mode", "smoke", "--output-dir"])
            .arg(&devnet)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "setup-local-testnet: {:?}",
            output_text(&out)
        );

        let keys = dir.join("keys");
        std::fs::create_dir(&keys).unwrap();
        let mut key_files = BTreeMap::new();
        let mut seeds = BTreeMap::new();
        for role in [
            "archive1", "archive2", "archive3", "client", "verifier", "funder",
        ] {
            let bytes: Vec<u8> = serde_json::from_slice(
                &std::fs::read(devnet.join("keys").join(format!("{role}.json"))).unwrap(),
            )
            .unwrap();
            let seed: [u8; 32] = bytes.try_into().expect("32-byte seed");
            let p = keys.join(format!("{role}.hex"));
            std::fs::write(&p, hex_lower(&seed)).unwrap();
            key_files.insert(role.to_string(), p);
            seeds.insert(role.to_string(), seed);
        }

        // 2. The validator, loopback only.
        let (rpc_port, p2p_port, health_port) = (free_port(), free_port(), free_port());
        let config = dir.join("validator.toml");
        std::fs::write(
            &config,
            format!("[health]\naddr = \"127.0.0.1:{health_port}\"\n"),
        )
        .unwrap();
        let validator_log = dir.join("validator.log");
        let mut cmd = Command::new(&sumchain);
        cmd.arg("run")
            .arg("--config")
            .arg(&config)
            .arg("--genesis")
            .arg(devnet.join("genesis.json"))
            .arg("--data-dir")
            .arg(devnet.join("data"))
            .arg("--validator-key")
            .arg(devnet.join("keys").join("validator.json"))
            .arg("--p2p-addr")
            .arg(format!("/ip4/127.0.0.1/tcp/{p2p_port}"))
            .arg("--rpc-addr")
            .arg(format!("127.0.0.1:{rpc_port}"));
        let mut procs = vec![Proc::spawn("validator", cmd, &validator_log)];
        let rpc_url = format!("http://127.0.0.1:{rpc_port}");
        let rpc_addr: std::net::SocketAddr = format!("127.0.0.1:{rpc_port}").parse().unwrap();

        let height = |url: &str| -> Option<u64> {
            rpc(url, "chain_getBlockHeight", serde_json::json!(["latest"]))
                .ok()
                .and_then(|v| v.get("height").and_then(|h| h.as_u64()))
        };
        let first = wait_until("the validator's RPC", Duration::from_secs(120), || {
            height(&rpc_url)
        });
        wait_until("block production", Duration::from_secs(120), || {
            height(&rpc_url).filter(|h| *h > first)
        });
        eprintln!("[e2e] validator producing at {rpc_url}");

        let mut dev = Devnet {
            dir: dir.to_path_buf(),
            rpc_url,
            rpc_addr,
            validator_log,
            sum_node,
            omni_node,
            key_files,
            seeds,
            procs: Vec::new(),
        };

        // 3. Three staked archives.
        for i in 1..=3 {
            let role = format!("archive{i}");
            let out = dev
                .sum_node_cmd(&role, &format!("store-{role}"), &dev.rpc_url)
                .args(["register-node", "--stake", ARCHIVE_STAKE])
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{role} register-node: {:?}",
                output_text(&out)
            );
            let mut cmd = dev.sum_node_cmd(&role, &format!("store-{role}"), &dev.rpc_url);
            cmd.arg("listen");
            procs.push(Proc::spawn(&role, cmd, &dir.join(format!("{role}.log"))));
        }
        let url = dev.rpc_url.clone();
        wait_until("three active archives", Duration::from_secs(120), || {
            let h = height(&url)?;
            let nodes = rpc(
                &url,
                "storage_getActiveNodesAtHeight",
                serde_json::json!([h]),
            )
            .ok()?;
            let n = nodes
                .as_array()?
                .iter()
                .filter(|n| n["role"] == "ArchiveNode" && n["status"] == "Active")
                .count();
            (n >= 3).then_some(())
        });
        eprintln!("[e2e] three archives registered and listening");
        dev.procs = procs;
        dev
    }

    /// `sum-node` as `role`, with its own store, against `rpc_url`.
    fn sum_node_cmd(&self, role: &str, store: &str, rpc_url: &str) -> Command {
        let mut cmd = Command::new(&self.sum_node);
        cmd.env("RUST_LOG", "info")
            .arg("--key-file")
            .arg(&self.key_files[role])
            .arg("--store-dir")
            .arg(self.dir.join(store))
            .arg("--rpc-url")
            .arg(rpc_url)
            .arg("--chain-id")
            .arg(CHAIN_ID.to_string());
        cmd
    }

    fn address(&self, role: &str) -> String {
        omni_store::snip_v2::snip_account_from_seed(&self.seeds[role]).unwrap()
    }

    /// Publish `bytes` as `role` with `sum-node ingest-v2`; their root.
    fn ingest(&self, role: &str, bytes: &[u8], name: &str) -> SnipV2ObjectId {
        let path = self.dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        let out = self
            .sum_node_cmd(role, &format!("store-{role}"), &self.rpc_url)
            .arg("ingest-v2")
            .arg(&path)
            .args(["--visibility", "public"])
            .output()
            .unwrap();
        let (so, se) = output_text(&out);
        assert!(out.status.success(), "ingest {name}: {so}\n{se}");
        let root = kv(&so.replace(": ", "="), "merkle_root")
            .unwrap_or_else(|| panic!("ingest {name}: no root: {so}\n{se}"))
            .to_string();
        assert!(so.contains("lifecycle: Active"), "ingest {name}: {so}");
        let root = SnipV2ObjectId::from_hex(&root).unwrap();
        assert_eq!(
            root,
            public_object_root(bytes),
            "SNIP's root = ours ({name})"
        );
        root
    }

    fn record(&self, root: &SnipV2ObjectId) -> Option<SnipV2FileInfo> {
        SnipV2Cli::new(SnipV2CliConfig {
            rpc_url: Some(self.rpc_url.clone()),
            ..SnipV2CliConfig::default()
        })
        .file_info_v2(root)
        .expect("chain lookup")
    }

    /// Register `root` Pending as `role` with the given metadata, by a
    /// hand-built `RegisterFilePendingV2` (what any account can send; the
    /// chain cannot check sizes against bytes it never sees). Nothing is
    /// pushed, so it stays Pending.
    fn register_pending_raw(&self, role: &str, root: &SnipV2ObjectId, size: u64, chunks: u32) {
        use sumchain_wire::storage_metadata::{
            StorageMetadataOperationV2, StorageMetadataV2TxData,
        };
        use sumchain_wire::transaction::{SignedTransaction, TransactionV2, TxPayload};
        use sumchain_wire::{Address, Hash};

        let signer = ContributorSigner::from_seed_bytes(&self.seeds[role]).unwrap();
        let pk = signer.pubkey_bytes();
        let from = Address::from_public_key(&pk);
        let nonce = rpc(
            &self.rpc_url,
            "get_nonce",
            serde_json::json!([from.to_base58()]),
        )
        .unwrap()
        .as_u64()
        .expect("nonce");
        let tx = TransactionV2 {
            chain_id: CHAIN_ID,
            from,
            fee: TX_FEE,
            nonce,
            payload: TxPayload::StorageMetadataV2(StorageMetadataV2TxData {
                operation: StorageMetadataOperationV2::RegisterFilePendingV2 {
                    merkle_root: Hash::new(*root.as_bytes()),
                    plaintext_size_bytes: size,
                    stored_size_bytes: size,
                    chunk_count: chunks,
                    fee_deposit: 0,
                    visibility: 0,
                    initial_access: vec![],
                },
            }),
        };
        let sig = signer.sign(tx.signing_hash().as_bytes());
        let hex = SignedTransaction::new_v2(tx, sig, pk).to_hex();
        rpc(
            &self.rpc_url,
            "send_raw_transaction",
            serde_json::json!([hex]),
        )
        .expect("send_raw_transaction");
        wait_until(
            "the hand-built registration",
            Duration::from_secs(60),
            || self.record(root),
        );
    }

    /// Lines of the validator's log containing every needle.
    fn validator_lines(&self, needles: &[&str]) -> usize {
        let log = strip_ansi(&std::fs::read_to_string(&self.validator_log).unwrap_or_default());
        log.lines()
            .filter(|l| needles.iter().all(|n| l.contains(n)))
            .count()
    }

    /// The blocks whose execution logged `event` for `root`. The proposer
    /// executes a block it proposes twice (it screens the proposal, then
    /// executes the block), logging each transaction's effect twice under
    /// the same block number; the set of blocks is what counts.
    fn blocks_logging(&self, event: &str, root: &SnipV2ObjectId) -> BTreeSet<u64> {
        let log = strip_ansi(&std::fs::read_to_string(&self.validator_log).unwrap_or_default());
        let needle = format!("merkle_root={}", root.to_hex());
        let mut block = None;
        let mut blocks = BTreeSet::new();
        for line in log.lines() {
            if let Some(rest) = line.split("Executing block ").nth(1) {
                block = rest
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse::<u64>().ok());
            } else if line.contains(event) && line.contains(&needle) {
                blocks.insert(block.expect("an execution line precedes every effect"));
            }
        }
        blocks
    }

    /// Blocks that registered `root`.
    fn registrations_of(&self, root: &SnipV2ObjectId) -> usize {
        self.blocks_logging("V2 file registered Pending", root)
            .len()
    }

    /// Blocks that activated `root`.
    fn activations_of(&self, root: &SnipV2ObjectId) -> usize {
        self.blocks_logging("V2 file activated", root).len()
    }
}

impl Drop for Devnet {
    fn drop(&mut self) {
        // Stop archives first, then the validator.
        while let Some(p) = self.procs.pop() {
            drop(p);
        }
    }
}

// ── Jobs and runs ────────────────────────────────────────────────────────────

struct JobFiles {
    job: ContributorJob,
    job_path: PathBuf,
}

fn make_job(
    dev: &Devnet,
    manifest: &[u8],
    manifest_root: &SnipV2ObjectId,
    input: &[u8],
    name: &str,
) -> JobFiles {
    let input_root = dev.ingest("verifier", input, &format!("{name}.input"));
    let mut job = ContributorJob {
        schema_version: 1,
        job_id: String::new(),
        model_hash: hex_lower(blake3::hash(manifest).as_bytes()),
        manifest_snip_root: manifest_root.to_hex(),
        input_snip_root: input_root.to_hex(),
        input_hash: hex_lower(blake3::hash(input).as_bytes()),
        verification_requirement: VerificationRequirement::AttestationOnly,
        accounting: JobAccounting {
            tokenizer_hash: hex_lower(blake3::hash(b"e2e-tokenizer").as_bytes()),
            tokenizer_id: "e2e/tokenizer".into(),
            input_token_count: INPUT_TOKENS,
            max_output_token_count: OUTPUT_TOKENS,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-10-28T00:00:00Z".into(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: None,
        dispatcher_signature_hex: None,
        notes: Some(format!("issue-149 e2e {name}")),
    };
    job.job_id = job_hash_hex(&job).unwrap();
    let job_path = dev.dir.join(format!("{name}.job.json"));
    std::fs::write(&job_path, serde_json::to_vec_pretty(&job).unwrap()).unwrap();
    JobFiles { job, job_path }
}

struct RunOutcome {
    ok: bool,
    stdout: String,
    stderr: String,
}

impl RunOutcome {
    fn publication(&self) -> Option<&str> {
        kv(&self.stdout, "response_publication")
    }
}

/// `omni-node operator contributor run-job` as the `client` account.
/// `omni-node operator contributor run-job` as the `client` account, without
/// its runner flags.
fn run_job_base(dev: &Devnet, job: &JobFiles, out: &Path, rpc_url: &str) -> Command {
    let seed = dev.dir.join("contributor.seed");
    if !seed.exists() {
        std::fs::write(&seed, CONTRIBUTOR_SEED).unwrap();
    }
    let mut cmd = Command::new(&dev.omni_node);
    cmd.env("SUM_RPC_URL", rpc_url)
        .env("SUM_KEY_FILE", &dev.key_files["client"])
        .env("SUM_STORE_DIR", dev.dir.join("store-contributor"))
        .env("SUM_CHAIN_ID", CHAIN_ID.to_string())
        .env("RUST_LOG", "info")
        .args(["operator", "contributor", "run-job", "--job"])
        .arg(&job.job_path)
        .arg("--out")
        .arg(out)
        .arg("--seed-file")
        .arg(&seed)
        .arg("--snip-binary")
        .arg(&dev.sum_node);
    cmd
}

/// [`run_job_base`] with the stub runner answering `response`.
fn run_job_cmd(
    dev: &Devnet,
    job: &JobFiles,
    response: &Path,
    out: &Path,
    rpc_url: &str,
) -> Command {
    let mut cmd = run_job_base(dev, job, out, rpc_url);
    cmd.args(["--runner", "stub", "--stub-response"])
        .arg(response)
        .args([
            "--stub-input-tokens",
            &INPUT_TOKENS.to_string(),
            "--stub-output-tokens",
            &OUTPUT_TOKENS.to_string(),
        ]);
    cmd
}

/// Standard base64 with padding.
fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                out.push(T[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// An external runner that marks `ran` when invoked, waits for `go`, then
/// answers `response`: it holds `run-job` between the downloads and the
/// publication.
fn gated_runner(dev: &Devnet, response: &[u8], ran: &Path, go: &Path) -> PathBuf {
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let envelope = serde_json::json!({
        "response_b64": base64(response),
        "measured_input_tokens": INPUT_TOKENS,
        "measured_output_tokens": OUTPUT_TOKENS,
        "stage_contributions": [{
            "contributor_pubkey_hex": signer.pubkey_hex(),
            "stage_label": "e2e",
            "work_unit_kind": "tokens",
            "work_units": INPUT_TOKENS + OUTPUT_TOKENS,
        }],
    });
    let script = dev.dir.join("gated-runner.sh");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\n: > '{ran}'\nwhile [ ! -f '{go}' ]; do /bin/sleep 0.1; done\nprintf '%s' '{envelope}'\n",
            ran = ran.display(),
            go = go.display(),
        ),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script
}

fn run_job(dev: &Devnet, job: &JobFiles, response: &Path, out: &Path, rpc_url: &str) -> RunOutcome {
    let o = run_job_cmd(dev, job, response, out, rpc_url)
        .output()
        .unwrap();
    let (stdout, stderr) = output_text(&o);
    RunOutcome {
        ok: o.status.success(),
        stdout,
        stderr,
    }
}

fn write_response(dev: &Devnet, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dev.dir.join(format!("{name}.response"));
    std::fs::write(&p, bytes).unwrap();
    p
}

/// `verify-result` says `overall=ok`, and a fresh download of the response
/// hashes to the signed `response_hash`.
fn assert_result_verifies(dev: &Devnet, job: &JobFiles, out: &Path, expected_bytes: &[u8]) {
    let result: ContributorResult =
        serde_json::from_slice(&std::fs::read(out).expect("result written")).unwrap();
    let o = Command::new(&dev.omni_node)
        .env("SUM_RPC_URL", &dev.rpc_url)
        .env("SUM_KEY_FILE", &dev.key_files["funder"])
        .env("SUM_STORE_DIR", dev.dir.join("store-verifier"))
        .env("SUM_CHAIN_ID", CHAIN_ID.to_string())
        .args(["operator", "contributor", "verify-result", "--job"])
        .arg(&job.job_path)
        .arg("--result")
        .arg(out)
        .arg("--snip-binary")
        .arg(&dev.sum_node)
        .output()
        .unwrap();
    let (so, se) = output_text(&o);
    assert!(o.status.success(), "verify-result: {so}\n{se}");
    assert_eq!(kv(&so, "overall"), Some("ok"), "verify-result: {so}\n{se}");
    assert_eq!(kv(&so, "response_hash_ok"), Some("true"), "{so}");

    let fetched = dev
        .dir
        .join(format!("fetched-{}", &result.response_hash[..16]));
    let store = format!("store-download-{}", &result.response_hash[..16]);
    let o = dev
        .sum_node_cmd("funder", &store, &dev.rpc_url)
        .arg("download")
        .arg(&result.response_snip_root)
        .arg("--output")
        .arg(&fetched)
        .output()
        .unwrap();
    assert!(o.status.success(), "download: {:?}", output_text(&o));
    let bytes = std::fs::read(&fetched).unwrap();
    assert_eq!(bytes, expected_bytes, "fetched response bytes");
    assert_eq!(
        hex_lower(blake3::hash(&bytes).as_bytes()),
        result.response_hash,
        "fetched bytes hash to the signed response_hash"
    );
    assert_eq!(result.job_id, job.job.job_id);
}

fn response_bytes(tag: u8) -> Vec<u8> {
    // 8 little-endian i16 outputs, as the canonical MLP produces.
    (0..16u8).map(|i| tag.wrapping_mul(17) ^ i).collect()
}

// ── The test ─────────────────────────────────────────────────────────────────

#[test]
#[ignore = "needs a local SUM Chain devnet and SNIP binaries; see the module docs"]
fn run_job_publishes_its_response_idempotently_against_a_real_devnet() {
    let tmp = tempfile::Builder::new()
        .prefix("omni-149-e2e-")
        .tempdir()
        .unwrap();
    // `OMNI_E2E_KEEP` keeps the directory, pass or fail.
    let (_tmp_guard, run_dir) = if std::env::var_os("OMNI_E2E_KEEP").is_some() {
        let p = tmp.keep();
        eprintln!("[e2e] keeping {}", p.display());
        (None, p)
    } else {
        let p = tmp.path().to_path_buf();
        (Some(tmp), p)
    };
    let dev = Devnet::start(&run_dir);
    let client = dev.address("client");
    let funder = dev.address("funder");

    let manifest = b"issue-149 e2e manifest (stand-in for a model manifest)".to_vec();
    let manifest_root = dev.ingest("verifier", &manifest, "manifest");
    // The log reading itself: one ingest is one registration, one activation.
    assert_eq!(dev.registrations_of(&manifest_root), 1);
    assert_eq!(dev.activations_of(&manifest_root), 1);

    // ── 1. The same job, twice ────────────────────────────────────────────
    let r1 = response_bytes(1);
    let root1 = public_object_root(&r1);
    let resp1 = write_response(&dev, "r1", &r1);
    let job_a = make_job(&dev, &manifest, &manifest_root, b"input-a", "job-a");
    let out_a1 = dev.dir.join("job-a.result.1.json");
    let run = run_job(&dev, &job_a, &resp1, &out_a1, &dev.rpc_url);
    assert!(run.ok, "1st run: {}\n{}", run.stdout, run.stderr);
    assert_eq!(run.publication(), Some("ingested"));
    assert_eq!(
        kv(&run.stdout, "response_snip_root"),
        Some(root1.to_hex().as_str())
    );
    let created1 = dev.record(&root1).unwrap().created_at;
    assert_result_verifies(&dev, &job_a, &out_a1, &r1);

    let out_a2 = dev.dir.join("job-a.result.2.json");
    let run = run_job(&dev, &job_a, &resp1, &out_a2, &dev.rpc_url);
    assert!(run.ok, "2nd run: {}\n{}", run.stdout, run.stderr);
    assert_eq!(run.publication(), Some("reused_active"));
    assert_result_verifies(&dev, &job_a, &out_a2, &r1);
    eprintln!("[e2e] 1. same job twice: ingested, then reused_active ({root1})");

    // ── 2. A different job, identical response bytes ─────────────────────
    let job_b = make_job(&dev, &manifest, &manifest_root, b"input-b", "job-b");
    assert_ne!(job_a.job.job_id, job_b.job.job_id);
    let out_b = dev.dir.join("job-b.result.json");
    let run = run_job(&dev, &job_b, &resp1, &out_b, &dev.rpc_url);
    assert!(run.ok, "job b: {}\n{}", run.stdout, run.stderr);
    assert_eq!(run.publication(), Some("reused_active"));
    assert_result_verifies(&dev, &job_b, &out_b, &r1);
    assert_eq!(dev.registrations_of(&root1), 1, "root1 registered once");
    let rec1 = dev.record(&root1).unwrap();
    assert_eq!((rec1.lifecycle, rec1.created_at), (1, created1));
    assert_eq!(rec1.owner, client);
    eprintln!("[e2e] 2. different job, same bytes: reused_active");

    // ── 3. RPC lost right after the registration lands (issue #149) ───────
    let r3 = response_bytes(3);
    let root3 = public_object_root(&r3);
    let resp3 = write_response(&dev, "r3", &r3);
    let job_c = make_job(&dev, &manifest, &manifest_root, b"input-c", "job-c");
    let proxy = RpcProxy::start(dev.rpc_addr);
    let out_c = dev.dir.join("job-c.result.json");
    let child = run_job_cmd(&dev, &job_c, &resp3, &out_c, &proxy.url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    wait_until("root3's registration", Duration::from_secs(120), || {
        dev.record(&root3).filter(|r| r.lifecycle == 0)
    });
    proxy.cut();
    let o = child.wait_with_output().unwrap();
    let (so, se) = output_text(&o);
    assert!(!o.status.success(), "attempt 1 must fail: {so}\n{se}");
    assert!(!out_c.exists(), "no result after a failed run");
    assert!(
        se.contains("not confirmed") && se.contains("re-reading the chain record then failed"),
        "the error names the unconfirmed publication and the failed re-read: {se}"
    );
    assert!(
        se.contains("sum-node ingest-v2") && (se.contains("log: ") || se.contains("stderr: ")),
        "what sum-node reported is in the error, not only a missing line: {se}"
    );
    std::thread::sleep(Duration::from_secs(5));
    let rec = dev.record(&root3).unwrap();
    assert_eq!(
        (rec.lifecycle, rec.owner.as_str()),
        (0, client.as_str()),
        "left Pending"
    );
    let created3 = rec.created_at;
    eprintln!(
        "[e2e] 3. RPC cut after registration: failed with: {}",
        se.lines().find(|l| l.starts_with("Error")).unwrap_or("")
    );

    proxy.restore();
    let run = run_job(&dev, &job_c, &resp3, &out_c, &proxy.url);
    assert!(run.ok, "retry: {}\n{}", run.stdout, run.stderr);
    assert_eq!(run.publication(), Some("resumed"));
    assert_result_verifies(&dev, &job_c, &out_c, &r3);
    let rec = dev.record(&root3).unwrap();
    assert_eq!((rec.lifecycle, rec.created_at), (1, created3));
    assert_eq!(dev.registrations_of(&root3), 1, "root3 registered once");
    assert_eq!(dev.activations_of(&root3), 1);
    eprintln!("[e2e] 3. retry: resumed");

    // ── 4. The run killed right after the registration lands ──────────────
    let r4 = response_bytes(4);
    let root4 = public_object_root(&r4);
    let resp4 = write_response(&dev, "r4", &r4);
    let job_d = make_job(&dev, &manifest, &manifest_root, b"input-d", "job-d");
    let out_d = dev.dir.join("job-d.result.json");
    let mut run_proc = Proc::spawn(
        "run-job (to be killed)",
        run_job_cmd(&dev, &job_d, &resp4, &out_d, &dev.rpc_url),
        &dev.dir.join("job-d.killed.log"),
    );
    wait_until("root4's registration", Duration::from_secs(120), || {
        dev.record(&root4).filter(|r| r.lifecycle == 0)
    });
    run_proc.kill_group();
    drop(run_proc);
    std::thread::sleep(Duration::from_secs(5));
    assert!(!out_d.exists(), "no result after a killed run");
    let rec = dev.record(&root4).unwrap();
    assert_eq!(rec.lifecycle, 0, "killed before activation");
    let created4 = rec.created_at;
    let run = run_job(&dev, &job_d, &resp4, &out_d, &dev.rpc_url);
    assert!(run.ok, "retry: {}\n{}", run.stdout, run.stderr);
    assert_eq!(run.publication(), Some("resumed"));
    assert_result_verifies(&dev, &job_d, &out_d, &r4);
    assert_eq!(dev.record(&root4).unwrap().created_at, created4);
    assert_eq!(dev.registrations_of(&root4), 1, "root4 registered once");
    eprintln!("[e2e] 4. killed after registration; retry: resumed");

    // ── 5. Activated, then the result write fails ─────────────────────────
    let r5 = response_bytes(5);
    let root5 = public_object_root(&r5);
    let resp5 = write_response(&dev, "r5", &r5);
    let job_e = make_job(&dev, &manifest, &manifest_root, b"input-e", "job-e");
    let bad_out = dev.dir.join("no-such-dir").join("job-e.result.json");
    let run = run_job(&dev, &job_e, &resp5, &bad_out, &dev.rpc_url);
    assert!(!run.ok, "the result write must fail: {}", run.stdout);
    assert!(run.stderr.contains("write result"), "{}", run.stderr);
    assert!(!bad_out.exists());
    assert_eq!(
        dev.record(&root5).unwrap().lifecycle,
        1,
        "Active before the write"
    );
    let out_e = dev.dir.join("job-e.result.json");
    let run = run_job(&dev, &job_e, &resp5, &out_e, &dev.rpc_url);
    assert!(run.ok, "retry: {}\n{}", run.stdout, run.stderr);
    assert_eq!(run.publication(), Some("reused_active"));
    assert_result_verifies(&dev, &job_e, &out_e, &r5);
    assert_eq!(dev.registrations_of(&root5), 1, "root5 registered once");
    eprintln!("[e2e] 5. result write failed after activation; retry: reused_active");

    // ── 6. Conflicting metadata ───────────────────────────────────────────
    for (tag, role) in [(6u8, "funder"), (7u8, "client")] {
        let r = response_bytes(tag);
        let root = public_object_root(&r);
        dev.register_pending_raw(role, &root, r.len() as u64 + 1, 1);
        let resp = write_response(&dev, &format!("r{tag}"), &r);
        let job = make_job(
            &dev,
            &manifest,
            &manifest_root,
            &[b'i', tag],
            &format!("job-{tag}"),
        );
        let out = dev.dir.join(format!("job-{tag}.result.json"));
        let run = run_job(&dev, &job, &resp, &out, &dev.rpc_url);
        assert!(!run.ok, "conflict ({role}) must fail: {}", run.stdout);
        assert!(!out.exists());
        assert!(
            run.stderr.contains("plaintext_size_bytes") && run.stderr.contains("refusing"),
            "{}",
            run.stderr
        );
        let rec = dev.record(&root).unwrap();
        assert_eq!(
            (rec.lifecycle, rec.plaintext_size_bytes),
            (0, 17),
            "untouched"
        );
        assert_eq!(dev.registrations_of(&root), 1);
        eprintln!("[e2e] 6. conflicting metadata registered by {role}: refused");
    }

    // ── 7. Pending, registered by another account ─────────────────────────
    let r8 = response_bytes(8);
    let root8 = public_object_root(&r8);
    dev.register_pending_raw("funder", &root8, r8.len() as u64, 1);
    let resp8 = write_response(&dev, "r8", &r8);
    let job_h = make_job(&dev, &manifest, &manifest_root, b"input-h", "job-h");
    let out_h = dev.dir.join("job-h.result.json");
    let run = run_job(&dev, &job_h, &resp8, &out_h, &dev.rpc_url);
    assert!(!run.ok, "must fail: {}", run.stdout);
    assert!(!out_h.exists());
    assert!(
        run.stderr.contains(&funder) && run.stderr.contains(&client),
        "{}",
        run.stderr
    );
    let rec = dev.record(&root8).unwrap();
    assert_eq!((rec.lifecycle, rec.owner.as_str()), (0, funder.as_str()));
    assert_eq!(dev.registrations_of(&root8), 1);
    eprintln!("[e2e] 7. Pending registered by another account: refused");

    // ── 8. Chain RPC unreachable ──────────────────────────────────────────
    // 8a. From the start: the run fails at its first SNIP call (the input
    //     download), long before publication.
    let r9 = response_bytes(9);
    let root9 = public_object_root(&r9);
    let resp9 = write_response(&dev, "r9", &r9);
    let job_i = make_job(&dev, &manifest, &manifest_root, b"input-i", "job-i");
    let out_i = dev.dir.join("job-i.result.json");
    let dead = format!("http://127.0.0.1:{}", free_port());
    let run = run_job(&dev, &job_i, &resp9, &out_i, &dead);
    assert!(!run.ok, "must fail: {}", run.stdout);
    assert!(!out_i.exists());
    assert!(run.stderr.contains("RPC"), "{}", run.stderr);
    assert!(dev.record(&root9).is_none(), "nothing registered");
    eprintln!("[e2e] 8a. RPC unreachable from the start: failed, nothing registered");

    // 8b. Lost after the downloads and the runner, before publication: the
    //     chain lookup fails and nothing is ingested.
    let r10 = response_bytes(10);
    let root10 = public_object_root(&r10);
    let job_j = make_job(&dev, &manifest, &manifest_root, b"input-j", "job-j");
    let out_j = dev.dir.join("job-j.result.json");
    let (ran, go) = (dev.dir.join("runner.ran"), dev.dir.join("runner.go"));
    let runner = gated_runner(&dev, &r10, &ran, &go);
    let proxy8 = RpcProxy::start(dev.rpc_addr);
    let child = run_job_base(&dev, &job_j, &out_j, &proxy8.url)
        .args(["--runner", "external", "--external-command"])
        .arg(&runner)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    wait_until("the runner", Duration::from_secs(180), || {
        ran.exists().then_some(())
    });
    proxy8.cut();
    std::fs::write(&go, b"").unwrap();
    let o = child.wait_with_output().unwrap();
    let (so, se) = output_text(&o);
    assert!(!o.status.success(), "must fail: {so}\n{se}");
    assert!(!out_j.exists());
    assert!(
        se.contains("storage_getFileInfoV2") && !se.contains("sum-node ingest-v2"),
        "the lookup failed and no ingest ran: {se}"
    );
    assert!(dev.record(&root10).is_none(), "nothing registered");
    eprintln!(
        "[e2e] 8b. RPC lost before publication: {}",
        se.lines().find(|l| l.starts_with("Error")).unwrap_or("")
    );

    // ── No duplicate registration, anywhere ───────────────────────────────
    assert_eq!(
        dev.validator_lines(&["already registered as V2 file"]),
        0,
        "the chain refused no duplicate registration in the whole run"
    );
    for (n, root) in [(1, &root1), (3, &root3), (4, &root4), (5, &root5)] {
        assert_eq!(dev.registrations_of(root), 1, "root{n}");
        assert_eq!(dev.activations_of(root), 1, "root{n}");
    }
    eprintln!("[e2e] every root registered once; no refused duplicate registration");

    drop(dev);
}

#[test]
fn base64_matches_rfc_4648_vectors() {
    for (raw, enc) in [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ] {
        assert_eq!(base64(raw.as_bytes()), enc);
    }
}
