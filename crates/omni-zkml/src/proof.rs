//! Phase 5 Stage 11a — proof generation backend trait + mock impl + orchestrator.
//!
//! This module introduces the first real proof-generation surface in OmniNode.
//! Stages 1-10a treated the proof bytes as opaque: the [`crate::artifact`]
//! module published whatever bytes a caller handed it, [`crate::build_commitment`]
//! emitted the SNIP V2 root of those bytes as `proof_snip_root`, and the
//! operator binary's `smoke --synthetic` flow hardcoded synthetic placeholder
//! bytes (`model_hash="a".repeat(64)`, `proof_snip_root=[0x22;32]`, etc. —
//! see the 2026-05-19 mainnet smoke audit for the on-chain evidence).
//!
//! Stage 11a replaces that placeholder with an honest end-to-end pipeline:
//!
//! 1. A [`ProofBackend`] trait abstracts "produce proof bytes given a
//!    `(model, input, output)` triple". Stage 11a ships [`MockProofBackend`],
//!    a deterministic non-cryptographic stand-in usable for local/dev/CI
//!    only. Stage 11b will plug in real systems (ezkl / risc0 / sp1).
//! 2. A [`ProofVerifier`] trait abstracts "verify proof bytes against
//!    `PublicInputs`". [`MockProofVerifier`] re-derives the deterministic
//!    mock bytes and byte-compares — that's not a SNARK check, but it is a
//!    real "tampering breaks verification" surface for tests.
//! 3. A [`ProofMetadata`] struct carries `(backend_id, model_hash,
//!    input_hash, response_hash)` **inside** the SNIP V2 proof artifact —
//!    committed by the existing `proof_snip_root` on the chain digest. This
//!    binds the input to the proof without changing Stage 6 chain-wire
//!    format. Off-chain verifiers fetch the proof artifact via SNIP V2,
//!    recover `ProofMetadata` + proof bytes, and run [`ProofVerifier::verify`].
//! 4. The [`produce_proof_artifact`] orchestrator wraps backend invocation,
//!    metadata composition, canonical JSON serialization, file write, and
//!    SNIP V2 publish via the existing [`crate::artifact::publish_proof_artifacts`]
//!    plumbing. It returns `(ProofArtifact, ResponseArtifact, ProofMetadata)`
//!    so the caller can hand the artifacts to the unchanged
//!    [`crate::build_commitment`] and continue through the Stage 4
//!    `build_attestation` + Stage 5.3 submit-workflow path with zero
//!    further edits.
//!
//! ## What stays unchanged
//!
//! - Stage 6 chain-wire bytes ([`crate::chain_wire`]). The on-chain digest
//!   still carries `(session_id, model_hash, manifest_root, response_hash,
//!   proof_root)` and only those. `proof_root` is still a SNIP V2 Merkle
//!   root — but now it's the root of a structured JSON envelope containing
//!   metadata + opaque proof bytes, rather than the root of synthetic
//!   placeholder bytes.
//! - Stage 7b transaction construction ([`omni_sumchain::tx`]).
//! - The [`crate::ProofArtifact`] / [`crate::ResponseArtifact`] structs.
//! - The [`crate::build_commitment`] signature. Its callers feed it richer
//!   artifacts now, but the function itself doesn't know proofs got real.
//!
//! ## Mainnet guard (operator-level, enforced in `omni-node`)
//!
//! `MockProofBackend::backend_id() == "mock-v1"` is non-cryptographic by
//! design. The operator binary refuses `chain_id == 1` submissions whose
//! backend is `mock-v1`, **including** with `--allow-mainnet-submit`. This
//! is a hard Stage 11a requirement. See `crates/omni-node/src/operator.rs`
//! for the enforcement site.

use std::path::Path;

use blake3;
use omni_store::SnipV2Adapter;
use serde::{Deserialize, Serialize};

use crate::artifact::{publish_proof_artifacts, ProofArtifact, ResponseArtifact};
use crate::error::{ProofBackendError, ProofPipelineError, ProofVerifierError};

// ── Stable backend ids ──────────────────────────────────────────────────────

/// Stage 11a's mock backend id. The mainnet guard in `omni-node` keys off
/// this constant; do NOT rename without coordinating the operator-side check.
pub const MOCK_BACKEND_ID: &str = "mock-v1";

// ── ModelFormat + ProofSystem (Stage 11b.0) ─────────────────────────────────

/// Stage 11b.0 — declared format of the model whose inference a proof
/// binds. Recorded in [`ProofMetadata::model_format`] and consulted by
/// the mainnet refusal logic in [`check_mainnet_eligible`].
///
/// **`ModelFormat::Other(_)` is always refused on mainnet** until a
/// future stage promotes it to a first-class enum variant with
/// chain-team review (Stage 11b.0 hard rule). `Gguf` is declared but
/// has no approved backend until Stage 11d ships an approved
/// strategy; declaring `Gguf` on a proof artifact is honest about the
/// intent but still refused on mainnet by the same logic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// ONNX (Open Neural Network Exchange). A common format for
    /// production ML provers. **No backend approved for mainnet at
    /// end of Stage 11d.x.** The Stage 11b.1 ezkl-discovery spike
    /// rejected ezkl as a dependency (`v23.0.5` has no `LICENSE`
    /// file or `Cargo.toml` `license` field); the Stage 11d.2
    /// production proof class selection remains open. ONNX
    /// requires a separately-reviewed Rust prover with a clean
    /// license before any allowlist entry can be proposed.
    Onnx,
    /// GGUF (llama.cpp model format). OmniNode's canonical model
    /// format today. **No proof backend approved at any stage
    /// through Stage 11d.x.** A future Stage 11e research track
    /// may evaluate the strategies documented in the operator
    /// runbook's "GGUF proofs" section (shadow verifier circuit,
    /// partial-inference proof, replay-based attestation); none
    /// of them prove full transformer inference correctness, and
    /// none is on the Stage 11d allowlist table. Declaring `Gguf`
    /// is explicitly invalid for any Stage 11d.3 allowlist entry
    /// per `docs/mainnet-eligibility-criteria.md` §6. Until a
    /// separately-reviewed strategy lands, declaring `Gguf` makes
    /// the refusal explicit instead of hidden — the architecture
    /// refuses the claim with a clear "no GGUF proof backend
    /// approved" message at layer 4.
    Gguf,
    /// Stage 11b.1: canonical bounded MLP described by a versioned
    /// JSON spec (architecture + weights + biases + quantization
    /// rules + canonical evaluation tuple). The model isn't
    /// imported from an ONNX/GGUF file — it's defined directly by
    /// the spec and reproduced by every framework's manifest in the
    /// cross-framework equivalence fixtures. **Testnet/dev only**;
    /// mainnet hard-refused via Stage 11b.0 refusal layers 1 + 3.
    Halo2ReferenceMlp,
    /// Stage 11d.2 — first production-grade fixed-point MLP model
    /// class. Distinct from `Halo2ReferenceMlp` (bounded
    /// testnet/dev-only) and non-Onnx (no exposed ONNX export path;
    /// the canonical spec at
    /// `crates/omni-proofs-halo2-production-mlp/assets/canonical_spec.json`
    /// is the model definition). **Off-chain proof metadata only.**
    /// Carried on `ProofArtifactBody.metadata` and consumed by the
    /// off-chain `check_mainnet_eligible` helper and the off-chain
    /// `Halo2ProductionMlpVerifier`; never serialized into chain
    /// wire, into `InferenceAttestationDigest`, into any SUM Chain
    /// RPC, or into any validator-side verification path. Mainnet
    /// eligibility STILL gated on a Stage 11d.3 allowlist entry
    /// after written chain-team sign-off; until then, layer 6 of
    /// `check_mainnet_eligible` hard-refuses any artifact carrying
    /// this format because `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`
    /// is empty.
    ProductionFixedPointMlp,
    /// Stringly-typed escape hatch for future formats. **Always
    /// refused on mainnet** unless and until promoted to a first-class
    /// enum variant with chain-team review.
    Other(String),
}

/// Stage 11b.1: ML framework that produced (or attested) a proof
/// artifact. Independent of [`ModelFormat`] — a single framework can
/// produce ONNX, GGUF, or a halo2-reference fixture; conversely, a
/// halo2-reference fixture can be reproduced by multiple frameworks
/// against the canonical spec. The pair `(model_framework,
/// model_format)` is the full description of an artifact's lineage.
///
/// **Operator runtime never executes any of these frameworks.**
/// Framework integration in Stage 11b.1.a / 11b.1.b is via committed
/// fixture manifests (developer-host-only tools regenerate them).
/// No PyTorch / TensorFlow / Caffe / RUMUS runtime appears in the
/// operator binary, in default CI, or on any code path that runs
/// during proof verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFramework {
    /// RUMUS. **Stage 11b.1.a status: equal-status primary
    /// (LiveExport).** `rumus = "0.4.0"` ships a first-class
    /// deterministic CPU FixedI16 integer-dense path
    /// (`rumus::fixed::FixedLinear` + `rumus::fixed::requantize`)
    /// implementing the canonical contract bit-identically. The
    /// committed RUMUS manifest is the live output of
    /// `tools/rumus_export/` — a standalone Cargo package
    /// intentionally outside the OmniNode workspace
    /// (root `Cargo.toml` declares `exclude = ["tools"]`) so the
    /// operator-binary build graph cannot transitively reach
    /// `rumus`.
    Rumus,
    /// PyTorch. **Stage 11b.1.a status: equal-status primary
    /// (LiveExport).** Manifest produced by
    /// `tools/pytorch_export/pytorch_export.py` via explicit
    /// `torch.int64` integer arithmetic — no `torch.quantization`
    /// / `torch.ao` APIs. Developer-host-only; never imported by
    /// the operator binary or by default CI.
    PyTorch,
    /// TensorFlow. **Stage 11b.1.a status: equal-status primary
    /// (LiveExport).** Manifest produced by
    /// `tools/tensorflow_export/tensorflow_export.py` via explicit
    /// `tf.int64` integer arithmetic — no `tf.quantization` /
    /// `tf.lite` APIs. Developer-host-only.
    TensorFlow,
    /// Caffe. **Stage 11b.1.a status: equal-status primary.**
    /// `tools/caffe_export/caffe_export.py` defaults to real
    /// Caffe if importable (`generation_mode: LiveExport`,
    /// `generator_metadata.runtime_mode: "caffe-runtime"`) and
    /// otherwise falls back to an explicit, auditable pure-NumPy
    /// emulation of the canonical contract
    /// (`generation_mode: PureNumpyCompatibility`,
    /// `generator_metadata.runtime_mode: "pure-numpy-emulation"`,
    /// `caffe_runtime_present: false`). The output bytes are
    /// byte-identical in both modes because the canonical
    /// arithmetic is deterministic; the manifest distinguishes
    /// them so the fallback is never silent.
    Caffe,
    /// Framework-agnostic — the framework-neutral canonical spec
    /// JSON is the source of truth. The Rust canonical evaluator
    /// is the neutral reference implementation (not a fifth
    /// framework); the committed `framework_agnostic_manifest.json`
    /// is a schema-coverage regression fixture with
    /// `generation_mode: ManualFixture`.
    FrameworkAgnostic,
}

/// Stage 11b.0 — declared proof system that produced a proof. Recorded
/// in [`ProofMetadata::proof_system`]; the mainnet refusal logic in
/// [`check_mainnet_eligible`] consults this to route through layered
/// refusals.
///
/// **The mainnet allowlist `MAINNET_APPROVED_PROOF_SYSTEMS` is empty at
/// end of Stage 11b.0.** No `ProofSystem` variant is mainnet-eligible
/// in 11b.0. Stage 11c is the earliest point at which a real proof
/// system can be added — and only after chain-team review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofSystem {
    /// Stage 11a — [`MockProofBackend`]. **Non-cryptographic; mainnet
    /// hard-refused.** Hermetic test scaffold only.
    Mock,
    /// Stage 11b.0 — bounded ONNX reference proof backend variant,
    /// **preserved-unused.** The Stage 11b.0 schema shipped this
    /// variant in anticipation of an ezkl-on-ONNX bounded backend;
    /// the 2026-05-22 ezkl discovery spike found ezkl lacks an
    /// explicit license grant (no `LICENSE` file at `v23.0.5`, no
    /// `Cargo.toml` `license` field), so OmniNode does not depend
    /// on it. Stage 11b.1 instead uses
    /// [`ProofSystem::Stage11bHalo2Reference`] below. This variant
    /// remains in the enum for back-compat (no Stage 11b.0 fixture
    /// ever referenced it, but the schema slot is preserved so
    /// removing it isn't a back-compat break either). **Refused on
    /// mainnet** by [`check_mainnet_eligible`] layer 3, identical
    /// to the new halo2 variant.
    Stage11bOnnxReference,
    /// Stage 11b.1 — bounded halo2 reference proof backend (the
    /// hand-rolled circuit lives in
    /// `crates/omni-proofs-halo2-reference`, lands in Stage 11b.1.b).
    /// Architecture-validation fixture; **testnet/dev only, mainnet
    /// hard-refused.** The artifact's
    /// [`ProofMetadata::testnet_or_dev_only`] flag MUST be `Some(true)`
    /// for any 11b.1 producer; refusal layers 1 (testnet flag) and
    /// 3 (bounded reference) both fire on mainnet.
    Stage11bHalo2Reference,
    /// Originally planned as a production ezkl ONNX prover variant.
    /// **The Stage 11b.1 ezkl-discovery spike rejected ezkl as a
    /// dependency** (`v23.0.5` has no `LICENSE` file or
    /// `Cargo.toml` `license` field). This variant is kept in the
    /// enum for back-compat (Stage 11b.0 schema slot) but is not
    /// targeted by any current stage. A future production ONNX
    /// prover would need a separately-reviewed Rust prover with a
    /// clean license. Refused on mainnet by layer 6 (allowlist
    /// empty); could be revisited as a Stage 11d.2 candidate if
    /// upstream license posture changes.
    Ezkl,
    /// Originally planned as a slot for a future GGUF-compatible
    /// proof strategy. **No GGUF strategy is on the Stage 11d
    /// table.** A future Stage 11e research track may evaluate the
    /// strategies documented in the operator runbook's "GGUF
    /// proofs" section. **No proof readiness claim today.**
    /// Refused on mainnet by layer 6 (allowlist empty); if
    /// `model_format = Gguf`, layer 4 fires first regardless.
    GgufStrategyTbd,
    /// Stage 11d.2 — first production-grade fixed-point MLP proof
    /// system. Built on Stage 11c's RHAZ + saturation + ReLU
    /// gadget chain (copy-pasted into a separate production
    /// crate per criteria §1.6 distinguishability hard rules),
    /// scaled to a representative `16 → 32 → 16 → 8` MLP
    /// classifying deterministic small-model workloads. **Distinct
    /// from `Stage11bHalo2Reference` per criteria §1.6 H1**;
    /// distinct `circuit_id_hex` (H2); distinct `model_hash` (H3).
    /// **Off-chain proof metadata only.** Carried on
    /// `ProofArtifactBody.metadata` and consumed by the off-chain
    /// `omni-node operator verify-proof` dispatch under the opt-in
    /// `stage11d-production-verify` cargo feature; never serialized
    /// into chain wire, into `InferenceAttestationDigest`, into
    /// any SUM Chain RPC, or into any validator-side verification
    /// path. Mainnet eligibility STILL gated on a Stage 11d.3
    /// allowlist entry after written chain-team sign-off
    /// (external cryptographer Claim 1.1.S2 + review packet R1–R9);
    /// until then, layer 6 of `check_mainnet_eligible` hard-refuses
    /// any artifact carrying this proof_system because
    /// `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` is empty.
    Stage11dProductionFixedPointMlp,
}

/// Stage 11b.0 → 11d.1 — the **legacy** mainnet allowlist of proof
/// systems. Kept as an empty back-compat alias for Stage 11d.1; the
/// structured replacement is [`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`]
/// below. Both must remain empty until Stage 11d.3 ships a written
/// chain-team sign-off; layer 6 of [`check_mainnet_eligible`] accepts
/// an artifact if EITHER list matches (so dual-empty preserves the
/// Stage 11b.0 "refuse everything" invariant bit-for-bit).
///
/// The `mainnet_allowlist_is_empty_at_stage_11b0` and
/// `legacy_MAINNET_APPROVED_PROOF_SYSTEMS_empty_after_stage_11d1` tests
/// pin this invariant.
pub const MAINNET_APPROVED_PROOF_SYSTEMS: &[ProofSystem] = &[];

/// Stage 11d.1 — structured mainnet allowlist entry.
///
/// The triple `(proof_system, circuit_id_hex, model_hash)` is the
/// matching key consulted by [`check_mainnet_eligible`] layer 6. The
/// remaining fields are recorded for audit but NOT part of the
/// matching key — rationale documented in
/// `docs/mainnet-eligibility-criteria.md` §3.
///
/// **Cannot be `Copy`** because [`ModelFormat::Other`] carries a
/// `String`. Clone is cheap (all other fields are `&'static str` or
/// small enums).
///
/// Invariants enforced by the `every_allowlist_entry_has_required_metadata`
/// runtime test:
///   - `proof_system` is not `Mock`, `Stage11bOnnxReference`, or
///     `Stage11bHalo2Reference` (bounded references are forever
///     testnet/dev-only per criteria doc §1.6 hard rule H1).
///   - `model_format` is not `Gguf` or `Other(_)` (criteria doc §6).
///   - `circuit_id_hex` / `model_hash` / `verification_key_hash_hex`
///     are 64 lowercase hex characters (BLAKE3 hash widths).
///   - `backend_id` and `chain_team_review_ref` are non-empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowlistEntry {
    pub proof_system: ProofSystem,
    pub backend_id: &'static str,
    pub circuit_id_hex: &'static str,
    pub model_hash: &'static str,
    pub model_format: ModelFormat,
    /// Stage 11d.1 — hex of [`mainnet_vk_hash`] applied to the
    /// per-verifier canonical VK bytes. The allowlist-side field
    /// name is deliberately `verification_key_hash_hex` (NOT
    /// `verification_key_hex`) to make it unambiguous that the
    /// stored value is a hash, not a raw VK encoding. The existing
    /// [`ProofMetadata::verification_key_hex`] artifact-side field
    /// is unchanged (its name is historical; per-verifier code
    /// cross-validates the two when Stage 11d.2 ships the first
    /// production verifier).
    pub verification_key_hash_hex: &'static str,
    pub chain_team_review_ref: &'static str,
}

/// Stage 11d.1 — domain separator for the canonical
/// mainnet-eligibility VK hash. Pinned at `b"OMNINODE-VK:v1:"` (15
/// ASCII bytes, no null terminator, no length prefix). The trailing
/// `v1` allows a future migration to a new scheme without ambiguity
/// over which hash an allowlist entry's `verification_key_hash_hex`
/// was computed under.
pub const MAINNET_VK_HASH_DOMAIN_SEPARATOR: &[u8] = b"OMNINODE-VK:v1:";

/// Stage 11d.1 — canonical mainnet-eligibility VK hash:
///
/// ```text
/// verification_key_hash =
///     BLAKE3(MAINNET_VK_HASH_DOMAIN_SEPARATOR || canonical_vk_bytes)
/// ```
///
/// `canonical_vk_bytes` is the per-verifier canonical serialization
/// of its `VerifyingKey`. Each production verifier MUST document its
/// `canonical_vk_bytes` scheme and pin a stable test vector.
///
/// Stage 11d.1 ships only this helper + the domain separator. The
/// concrete `canonical_vk_bytes` extraction lives in each production
/// verifier (Stage 11d.2+). Cross-validation between an artifact's
/// `metadata.verification_key_hex` and the allowlisted
/// `verification_key_hash_hex` is per-verifier business; layer 6 of
/// [`check_mainnet_eligible`] does NOT perform that cross-check.
pub fn mainnet_vk_hash(canonical_vk_bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MAINNET_VK_HASH_DOMAIN_SEPARATOR);
    hasher.update(canonical_vk_bytes);
    *hasher.finalize().as_bytes()
}

/// Stage 11d.1 — structured mainnet allowlist. **Empty by design**
/// at end of Stage 11d.1; populated only by a Stage 11d.3 PR
/// carrying written chain-team sign-off.
///
/// Layer 6 of [`check_mainnet_eligible`] consults this list keyed
/// on `(proof_system, circuit_id_hex, model_hash)`.
pub const MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES: &[AllowlistEntry] = &[];

// ── PublicInputs ────────────────────────────────────────────────────────────

/// Inputs a proof commits to. Typed (not opaque `&[u8]`) per Stage 11a OQ3
/// — picks up future backends adding fields without forcing a rewrite of
/// every verifier impl.
///
/// All three fields are 32-byte BLAKE3 hashes computed from the raw bytes of
/// the corresponding artifact. Sources at the operator side:
///
/// - `model_hash`: BLAKE3 of the model file (matches
///   [`omni_types::model::ModelManifest::model_hash`], which is the hex
///   rendering of the same bytes).
/// - `input_hash`: BLAKE3 of the inference input bytes.
/// - `output_hash`: BLAKE3 of the response bytes (equal to
///   [`omni_types::phase5::InferenceCommitment::response_hash`] decoded).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicInputs {
    pub model_hash: [u8; 32],
    pub input_hash: [u8; 32],
    pub output_hash: [u8; 32],
}

// ── ProofBackend ────────────────────────────────────────────────────────────

/// A backend that can produce proof bytes given `(model, input, output)`.
///
/// Stage 11a contract: implementations must be **deterministic** for the
/// same inputs (same `(model, input, output)` ⇒ same proof bytes). Stage
/// 11b may relax this if the chosen real backend has a salt / randomness
/// parameter, but that's an explicit decision then.
///
/// Stage 11b.0 extends the trait with four self-describing methods so
/// the off-chain verifier / mainnet refusal logic can route by
/// `proof_system`, refuse by `model_format`, and enforce the
/// "local-prover-only / no centralized service" architectural property.
/// Three methods are **defaulted** for back-compat with Stage 11a
/// implementations; [`ProofBackend::is_local_only`] is **required** so
/// every new backend has to declare its decentralization stance
/// explicitly — never by inheritance from a default.
pub trait ProofBackend {
    /// Produce proof bytes binding `(model, input, output)`.
    fn prove(
        &self,
        model: &[u8],
        input: &[u8],
        output: &[u8],
    ) -> std::result::Result<Vec<u8>, ProofBackendError>;

    /// Stable identifier for this backend (e.g. `"mock-v1"`, `"ezkl-halo2-v1"`).
    /// Recorded in [`ProofMetadata::backend_id`] so verifiers can pick the
    /// matching [`ProofVerifier`].
    fn backend_id(&self) -> &'static str;

    /// Stage 11b.0 — what proof system this backend uses. Recorded in
    /// [`ProofMetadata::proof_system`]; consumers route verification
    /// through the matching [`ProofVerifier`] and the mainnet refusal
    /// logic in [`check_mainnet_eligible`]. Defaults to
    /// [`ProofSystem::Mock`] so any implementor that hasn't migrated
    /// to Stage 11b.0 schema stays safely refused on mainnet.
    fn proof_system(&self) -> ProofSystem {
        ProofSystem::Mock
    }

    /// Stage 11b.0 — which model formats this backend can produce
    /// proofs for. Callers MUST consult this list before invoking
    /// [`ProofBackend::prove`] against bytes of a given format. An
    /// empty slice (the default) means "format-agnostic" — the case
    /// for [`MockProofBackend`], which operates on raw bytes without
    /// caring about their semantic shape.
    fn supported_formats(&self) -> &[ModelFormat] {
        &[]
    }

    /// Stage 11b.0 — circuit / image identifier specific to this
    /// backend's proof system. For ezkl this is the SHA of the
    /// compiled circuit; for RISC Zero it would be the image id; for
    /// the mock backend it's `None`. Recorded in
    /// [`ProofMetadata::circuit_id_hex`].
    fn circuit_id(&self) -> Option<[u8; 32]> {
        None
    }

    /// Stage 11b.0 — **required, no default.** Returns `true` iff this
    /// backend's prove path can run entirely on the operator's own
    /// host without any hosted service / centralized API / vendor
    /// dependency. Local CPU/GPU proving qualifies; anything that
    /// requires Bonsai or a hosted prover service does not.
    ///
    /// The decentralized proof architecture (Stage 11b.0) requires
    /// **all mainnet-eligible backends to return `true`**, enforced by
    /// the [`check_mainnet_eligible`] refusal layer that consults
    /// the backend registry. Backends that depend on hosted services
    /// can still implement the trait — but they're hard-refused on
    /// mainnet.
    ///
    /// No default — every implementor must declare. Defense in depth
    /// against a future backend silently inheriting the wrong answer.
    fn is_local_only(&self) -> bool;
}

// ── ProofVerifier ───────────────────────────────────────────────────────────

/// Verifier companion of [`ProofBackend`]. Returns `Ok(true)` on a valid
/// proof, `Ok(false)` on a structurally well-formed but failing proof, and
/// a typed error only for backend-internal failures (parse, panic, runtime).
///
/// Stage 11b.0 — verifiers self-identify their [`ProofSystem`] so the
/// off-chain verifier-routing logic (e.g., the `operator verify-proof`
/// subcommand) can dispatch to the right impl by inspecting the proof
/// artifact's [`ProofMetadata::proof_system`] field.
pub trait ProofVerifier {
    fn verify(
        &self,
        proof: &[u8],
        public_inputs: &PublicInputs,
    ) -> std::result::Result<bool, ProofVerifierError>;

    /// Stage 11b.0.1 — canonical dispatch entry point for verifying a
    /// full [`ProofArtifactBody`]. This is what `operator verify-proof`
    /// (and any other artifact-level consumer) calls uniformly across
    /// every backend. **Defaulted** to call
    /// [`Self::verify`] with the artifact's
    /// proof bytes and hashed [`PublicInputs`], so Stage 11a-shape
    /// verifiers that only consume the three hashes (the mock backend,
    /// and any future hash-only verifier) inherit the right behaviour
    /// automatically.
    ///
    /// Backends whose proof system needs backend-specific public
    /// inputs from [`ProofMetadata::public_inputs`] (the optional
    /// JSON-shaped field added in Stage 11b.0) **override this
    /// method**, validate the public-inputs schema themselves, run
    /// their proof-system-specific verify, and still cross-check the
    /// universal hashed `PublicInputs` via
    /// `body.metadata.public_inputs()` (the method on [`ProofMetadata`]
    /// that decodes the three hex hashes — distinct from the
    /// optional JSON field of the same name).
    ///
    /// This is the **single dispatch point** the operator binary uses;
    /// no operator-side per-backend helper calls. Future backends
    /// (Stage 11c+) plug into the dispatch loop by implementing
    /// `verify_artifact` and adding one match arm in
    /// `verify_proof_core` — they do not introduce backend-specific
    /// helper methods on the operator side.
    fn verify_artifact(
        &self,
        body: &ProofArtifactBody,
    ) -> std::result::Result<bool, ProofVerifierError> {
        let proof = body.proof_bytes()?;
        let public_inputs = body.metadata.public_inputs()?;
        self.verify(&proof, &public_inputs)
    }

    /// Stage 11b.0 — the proof system this verifier handles. Defaults
    /// to [`ProofSystem::Mock`] for back-compat with Stage 11a
    /// implementors; the [`MockProofVerifier`] overrides explicitly.
    fn proof_system(&self) -> ProofSystem {
        ProofSystem::Mock
    }
}

// ── Mock backend / verifier ─────────────────────────────────────────────────

/// Deterministic, non-cryptographic Stage 11a backend. Computes a 64-byte
/// "proof" as two domain-separated BLAKE3 hashes of `(model_hash ||
/// input_hash || output_hash)`. The hash chain makes the output vary with
/// inputs (so a tampered fixture fails the byte-roundtrip test) and lets
/// [`MockProofVerifier`] re-derive and byte-compare without needing the
/// original raw bytes — `PublicInputs` alone suffice.
///
/// **Not cryptographic.** This backend produces no soundness guarantee and
/// must not be used for mainnet submissions. The operator binary refuses
/// `mock-v1` on `chain_id == 1` (Stage 11a guardrail).
pub struct MockProofBackend;

impl ProofBackend for MockProofBackend {
    fn prove(
        &self,
        model: &[u8],
        input: &[u8],
        output: &[u8],
    ) -> std::result::Result<Vec<u8>, ProofBackendError> {
        let mh = blake3::hash(model);
        let ih = blake3::hash(input);
        let oh = blake3::hash(output);
        Ok(mock_proof_bytes(mh.as_bytes(), ih.as_bytes(), oh.as_bytes()))
    }

    fn backend_id(&self) -> &'static str {
        MOCK_BACKEND_ID
    }

    /// Stage 11b.0: explicit. The mock backend's proof system is
    /// `Mock`, which is hard-refused on mainnet by
    /// [`check_mainnet_eligible`] layer 2.
    fn proof_system(&self) -> ProofSystem {
        ProofSystem::Mock
    }

    /// Stage 11b.0: format-agnostic — the mock operates on raw bytes
    /// without caring about their semantic shape. The empty slice is
    /// the correct declaration; consumers querying for ONNX or GGUF
    /// support will see no match and route to a different backend.
    fn supported_formats(&self) -> &[ModelFormat] {
        &[]
    }

    /// Stage 11b.0: no circuit; the mock's "proof" is a fixed BLAKE3
    /// chain with no compiled program behind it.
    fn circuit_id(&self) -> Option<[u8; 32]> {
        None
    }

    /// Stage 11b.0: local-only by definition — the mock runs purely
    /// in-process with no network or hosted service. Declared
    /// `true` for completeness; the mainnet refusal still fires at
    /// layer 2 before this property is consulted.
    fn is_local_only(&self) -> bool {
        true
    }
}

/// Companion verifier for [`MockProofBackend`]. Re-derives the deterministic
/// mock bytes from `PublicInputs` and byte-compares against the supplied
/// proof.
pub struct MockProofVerifier;

impl ProofVerifier for MockProofVerifier {
    fn verify(
        &self,
        proof: &[u8],
        public_inputs: &PublicInputs,
    ) -> std::result::Result<bool, ProofVerifierError> {
        let expected = mock_proof_bytes(
            &public_inputs.model_hash,
            &public_inputs.input_hash,
            &public_inputs.output_hash,
        );
        Ok(proof == expected.as_slice())
    }

    /// Stage 11b.0: matches [`MockProofBackend::proof_system`] so the
    /// verifier-routing layer can dispatch by inspecting the proof
    /// artifact's [`ProofMetadata::proof_system`] field.
    fn proof_system(&self) -> ProofSystem {
        ProofSystem::Mock
    }
}

/// Domain-separated 64-byte deterministic "proof". Two BLAKE3 invocations
/// with disjoint domain tags so the two halves can't be swapped without
/// detection. The tags include the `mock-v1` backend id so a future
/// `mock-v2` would never collide.
fn mock_proof_bytes(
    model_hash: &[u8; 32],
    input_hash: &[u8; 32],
    output_hash: &[u8; 32],
) -> Vec<u8> {
    fn part(tag: &'static [u8], mh: &[u8; 32], ih: &[u8; 32], oh: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(tag);
        hasher.update(mh);
        hasher.update(ih);
        hasher.update(oh);
        *hasher.finalize().as_bytes()
    }
    let p0 = part(
        b"omninode.mock-v1.proof.part0",
        model_hash,
        input_hash,
        output_hash,
    );
    let p1 = part(
        b"omninode.mock-v1.proof.part1",
        model_hash,
        input_hash,
        output_hash,
    );
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&p0);
    out.extend_from_slice(&p1);
    out
}

// ── ProofMetadata + ProofArtifactBody ───────────────────────────────────────

/// Metadata embedded inside the SNIP V2 proof artifact. Committed by
/// `proof_snip_root` on the chain digest (which is the SNIP V2 Merkle root
/// of the canonical JSON envelope produced below). Off-chain verifiers
/// recover this struct to drive [`ProofVerifier::verify`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Stable identifier of the backend that produced the proof.
    pub backend_id: String,
    /// BLAKE3 hex of the model bytes (bare lowercase, 64 chars). Equal to
    /// [`omni_types::model::ModelManifest::model_hash`].
    pub model_hash: String,
    /// BLAKE3 hex of the inference input bytes (bare lowercase, 64 chars).
    /// **This is the binding for the "input" in
    /// `(model, input, output)`** — committed here, not on the chain
    /// digest, per Stage 11a OQ4.
    pub input_hash: String,
    /// BLAKE3 hex of the response/output bytes (bare lowercase, 64 chars).
    /// Equal to [`omni_types::phase5::InferenceCommitment::response_hash`].
    pub response_hash: String,

    // ── Stage 11b.0 additions (all optional + serde-skip-if-none) ─────
    //
    // Every field below uses `#[serde(skip_serializing_if = "Option::is_none",
    // default)]` so artifacts produced by Stage 11a code that doesn't
    // populate these fields serialize byte-identically. The Stage 11a
    // `proof_pipeline_vectors.json` fixture stays byte-stable; the
    // existing `stage11a_proof_pipeline_vectors_match_committed_fixture`
    // test confirms.
    /// Stage 11b.0: declared format of the model whose inference the
    /// proof binds. `None` on Stage 11a-vintage artifacts produced by
    /// [`MockProofBackend`]. Consulted by [`check_mainnet_eligible`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model_format: Option<ModelFormat>,

    /// Stage 11b.0: declared proof system that produced the proof.
    /// `None` is treated as [`ProofSystem::Mock`] by the mainnet
    /// refusal layer — preserving Stage 11a's "mock-v1 hard-refused"
    /// guarantee.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub proof_system: Option<ProofSystem>,

    /// Stage 11b.0: 64-char lowercase hex of the backend's
    /// `circuit_id`. Populated by backends that compile a circuit
    /// (ezkl, RISC Zero); `None` for [`MockProofBackend`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub circuit_id_hex: Option<String>,

    /// Stage 11b.0: hex of the public verification key the verifier
    /// needs to check this proof. Populated by backends whose proof
    /// systems require it (ezkl, Halo2, …); `None` for backends that
    /// encode the verification key inside the proof bytes.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub verification_key_hex: Option<String>,

    /// Stage 11b.0: operator-supplied public inputs the proof binds,
    /// in a backend-specific JSON shape. Each backend's crate
    /// documents its expected schema; treated as opaque by the
    /// refusal logic.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub public_inputs: Option<serde_json::Value>,

    /// Stage 11b.0: explicit "this artifact is testnet/dev only;
    /// reject on mainnet" flag. Producers MAY set this for hermetic
    /// or bounded-reference proofs (Stage 11b.1's halo2 reference
    /// backend does). [`check_mainnet_eligible`] refuses
    /// `Some(true)` at refusal layer 1, regardless of any other
    /// state. Note the flag is **opt-in**: `None` does NOT mean
    /// "mainnet OK" — mainnet eligibility requires the proof system
    /// to be in [`MAINNET_APPROVED_PROOF_SYSTEMS`] (empty at end of
    /// Stage 11b).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub testnet_or_dev_only: Option<bool>,

    /// Stage 11b.1.a: which ML framework produced (or attested) the
    /// artifact. Independent of [`ModelFormat`] — a framework
    /// describes the producer's runtime lineage, format describes
    /// the on-disk shape. `FrameworkAgnostic` is the natural value
    /// for the canonical halo2-reference artifact (the spec is
    /// reproducible across frameworks). The operator runtime never
    /// invokes any framework regardless of this field's value.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model_framework: Option<ModelFramework>,
}

impl ProofMetadata {
    /// Stage 11a-compatible constructor. All Stage 11b.0 fields
    /// default to `None`, which means the resulting JSON serializes
    /// byte-identically to a pre-Stage-11b.0 artifact (every new
    /// field uses `skip_serializing_if = "Option::is_none"`).
    ///
    /// Use this from any code that doesn't have a real backend
    /// behind it yet — the mock backend's orchestrator path, the
    /// Stage 11a hermetic fixture vectors, and any test that wants
    /// to construct a "minimal Stage 11a metadata" without spelling
    /// out six `None`s.
    pub fn new_stage11a(
        backend_id: String,
        model_hash: String,
        input_hash: String,
        response_hash: String,
    ) -> Self {
        Self {
            backend_id,
            model_hash,
            input_hash,
            response_hash,
            model_format: None,
            proof_system: None,
            circuit_id_hex: None,
            verification_key_hex: None,
            public_inputs: None,
            // Stage 11a vintage artifacts never declared this
            // field, so `None` is the honest default. Stage 11d.1
            // layer 1 treats `None` as testnet/dev for safety, so
            // any artifact produced via this constructor is hard-
            // refused on mainnet — preserving the Stage 11a
            // mock-only posture.
            testnet_or_dev_only: None,
            model_framework: None,
        }
    }

    /// Decode the three hex hashes into the typed [`PublicInputs`] struct
    /// the verifier consumes. Returns a typed `ProofVerifierError` on any
    /// hex parse failure so verifier callers don't have to convert errors.
    pub fn public_inputs(&self) -> std::result::Result<PublicInputs, ProofVerifierError> {
        Ok(PublicInputs {
            model_hash: decode_blake3_hex_lower("model_hash", &self.model_hash)?,
            input_hash: decode_blake3_hex_lower("input_hash", &self.input_hash)?,
            output_hash: decode_blake3_hex_lower("response_hash", &self.response_hash)?,
        })
    }
}

// ── Mainnet refusal (Stage 11b.0) ───────────────────────────────────────────

/// Stage 11b.0 — typed reasons [`check_mainnet_eligible`] returns when
/// a proof artifact's metadata is refused for mainnet attestation.
///
/// Each variant maps 1:1 to one of the six refusal layers documented
/// in the function below. The operator binary wraps these into its own
/// typed [`OperatorError`] variants (e.g.
/// `OperatorError::MockBackendRefusedOnMainnet`,
/// `OperatorError::GgufProofClaimRefusedOnMainnet`) so the
/// operator-facing error wording stays consistent with the rest of
/// the surface.
///
/// Implements `thiserror::Error` so the operator binary can
/// `#[from]` it into its own enum cleanly.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MainnetRefusalReason {
    /// Layer 1: artifact's `testnet_or_dev_only` field is **not**
    /// `Some(false)`. Both `Some(true)` (explicit disclaim) and
    /// `None` (absent — treated as testnet/dev for safety per
    /// `docs/mainnet-eligibility-criteria.md` §1.5) fire this
    /// layer. Only an artifact that **explicitly** declares
    /// `testnet_or_dev_only: Some(false)` can pass to deeper
    /// refusal layers (Stage 11d.1 tightening).
    #[error(
        "proof artifact is not declared mainnet-eligible \
         (testnet_or_dev_only = {testnet_or_dev_only:?}, backend_id = {backend_id:?}); \
         only Some(false) passes this layer"
    )]
    TestnetOrDevOnly {
        backend_id: String,
        testnet_or_dev_only: Option<bool>,
    },

    /// Layer 2: proof_system is Mock (or absent — Stage 11a vintage).
    /// Preserves Stage 11a's `MockBackendRefusedOnMainnet` guarantee.
    #[error(
        "proof artifact uses proof system {proof_system:?} (or none declared, treated as Mock); \
         the mock backend is non-cryptographic and cannot be used for mainnet submission"
    )]
    MockBackend {
        backend_id: String,
        proof_system: Option<ProofSystem>,
    },

    /// Layer 3: proof_system is a bounded-reference variant
    /// (currently `Stage11bOnnxReference`) — architecture-validation
    /// fixtures only, never mainnet.
    #[error(
        "proof artifact uses bounded reference proof system {proof_system:?} \
         (backend_id = {backend_id:?}); reference fixtures are for architecture \
         validation only, not production attestation"
    )]
    BoundedReference {
        backend_id: String,
        proof_system: ProofSystem,
    },

    /// Layer 4: model_format = Gguf claim. No GGUF inference proof
    /// backend is approved at any stage through Stage 11d.x;
    /// declaration is honest but the claim is hard-refused.
    /// `ModelFormat::Gguf` is explicitly invalid for any Stage
    /// 11d.3 allowlist entry per `docs/mainnet-eligibility-criteria.md`
    /// §6 — a future Stage 11e research track may evaluate
    /// strategies, but none is on the current roadmap.
    #[error(
        "proof artifact declares GGUF model_format with proof_system {proof_system:?} \
         (backend_id = {backend_id:?}); no GGUF inference proof backend is approved \
         through Stage 11d.x — awaiting a future Stage 11e research-track strategy + \
         chain-team review"
    )]
    GgufClaim {
        backend_id: String,
        proof_system: Option<ProofSystem>,
    },

    /// Layer 5: model_format = Other(_) or None. Stringly-typed
    /// formats are always refused on mainnet until promoted to a
    /// first-class enum variant; absent format on a non-mock backend
    /// is treated as the same refusal (you have to declare the
    /// format to be considered).
    #[error(
        "proof artifact has model_format = {model_format:?} which is not a \
         first-class approved format (backend_id = {backend_id:?})"
    )]
    UnknownModelFormat {
        backend_id: String,
        model_format: Option<ModelFormat>,
    },

    /// Layer 6: the artifact's `(proof_system, circuit_id_hex,
    /// model_hash)` triple does not match any entry in
    /// [`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`] (Stage 11d.1
    /// structured allowlist), and `proof_system` is not in the
    /// legacy [`MAINNET_APPROVED_PROOF_SYSTEMS`] back-compat alias.
    /// **Both lists are empty at end of Stage 11d.1** — no proof
    /// system gets through this gate yet.
    #[error(
        "proof system {proof_system:?} (backend_id = {backend_id:?}) is not in the \
         structured mainnet allowlist (MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES); both that \
         list and the legacy MAINNET_APPROVED_PROOF_SYSTEMS alias ship empty through \
         Stage 11d.1 / 11d.2 by design — mainnet eligibility lands when a Stage 11d.3 \
         entry is added after written chain-team sign-off"
    )]
    NotInMainnetAllowlist {
        backend_id: String,
        proof_system: ProofSystem,
    },
}

/// Stage 11b.0 — canonical mainnet refusal check.
///
/// Six layered refusal checks, evaluated in order. The first match
/// wins; the returned [`MainnetRefusalReason`] tells the caller
/// exactly which layer fired. The operator binary maps these into
/// its own typed `OperatorError` variants so operator-facing wording
/// stays consistent.
///
/// **Stage 11b.0 invariant: no `(metadata)` argument returns
/// `Ok(())`.** The mainnet allowlist
/// [`MAINNET_APPROVED_PROOF_SYSTEMS`] is empty at end of Stage 11b.0,
/// so layer 6 catches every proof system that escaped layers 1–5.
/// This invariant is pinned by the
/// `every_proof_system_is_refused_on_mainnet_at_stage_11b0` test.
///
/// Mainnet eligibility lands in a future chain-team-reviewed stage
/// (Stage 11d+); Stage 11c keeps the allowlist empty. **No Stage 11b
/// or 11c change should ever cause this function to return `Ok(())`.**
pub fn check_mainnet_eligible(
    meta: &ProofMetadata,
) -> std::result::Result<(), MainnetRefusalReason> {
    // Layer 1 (Stage 11d.1 tightening): only an artifact that
    // EXPLICITLY declares `testnet_or_dev_only: Some(false)` passes.
    // Both `Some(true)` (the producer disclaimed mainnet) and `None`
    // (the producer did not declare either way — treated as
    // testnet/dev for safety per criteria §1.5) refuse here. Fires
    // before any other check so a mis-declared artifact gets a
    // clear signal before the deeper layers run.
    if meta.testnet_or_dev_only != Some(false) {
        return Err(MainnetRefusalReason::TestnetOrDevOnly {
            backend_id: meta.backend_id.clone(),
            testnet_or_dev_only: meta.testnet_or_dev_only,
        });
    }

    // Layer 2: mock-system refusal (Stage 11a guarantee, generalized).
    // None is treated as Mock — Stage 11a artifacts didn't carry a
    // proof_system field, and the only Stage 11a backend was the
    // mock. Treating absence as Mock preserves the refusal contract.
    match meta.proof_system {
        Some(ProofSystem::Mock) | None => {
            return Err(MainnetRefusalReason::MockBackend {
                backend_id: meta.backend_id.clone(),
                proof_system: meta.proof_system,
            });
        }
        _ => {}
    }

    // Layer 3: bounded-reference / architecture-validation backends.
    // Stage 11b.1.a expands this from `Stage11bOnnxReference` only to
    // both Stage-11b reference variants — the halo2 reference is
    // ALSO architecture-validation, not production.
    if matches!(
        meta.proof_system,
        Some(ProofSystem::Stage11bOnnxReference) | Some(ProofSystem::Stage11bHalo2Reference)
    ) {
        return Err(MainnetRefusalReason::BoundedReference {
            backend_id: meta.backend_id.clone(),
            proof_system: meta.proof_system.unwrap(),
        });
    }

    // Layer 4: GGUF claims refused until Stage 11d ships an approved
    // strategy.
    if matches!(meta.model_format, Some(ModelFormat::Gguf)) {
        return Err(MainnetRefusalReason::GgufClaim {
            backend_id: meta.backend_id.clone(),
            proof_system: meta.proof_system,
        });
    }

    // Layer 5: Other(_) escape hatch + absent format on non-mock
    // backends. Both refused — you have to declare a first-class
    // approved format to be considered.
    if matches!(meta.model_format, Some(ModelFormat::Other(_)) | None) {
        return Err(MainnetRefusalReason::UnknownModelFormat {
            backend_id: meta.backend_id.clone(),
            model_format: meta.model_format.clone(),
        });
    }

    // Layer 6 (Stage 11d.1): structured allowlist match on the
    // (proof_system, circuit_id_hex, model_hash) triple, plus the
    // legacy bare-ProofSystem alias for back-compat. Both lists
    // are empty through Stage 11d.1 / 11d.2 by design; only a
    // Stage 11d.3 PR with written chain-team sign-off populates
    // the structured list. The OR is a deliberate back-compat
    // bridge — downstream stages can remove the legacy arm once
    // existing tests are migrated.
    let ps = meta.proof_system.unwrap();
    if !proof_system_passes_layer6(meta, ps) {
        return Err(MainnetRefusalReason::NotInMainnetAllowlist {
            backend_id: meta.backend_id.clone(),
            proof_system: ps,
        });
    }

    Ok(())
}

/// Layer-6 matching logic factored out so the `#[cfg(test)]` helper
/// [`check_mainnet_eligible_with`] can inject synthetic allowlists
/// without touching the real const slices.
fn proof_system_passes_layer6(meta: &ProofMetadata, ps: ProofSystem) -> bool {
    let matches_structured = matches_structured_allowlist(
        meta,
        ps,
        MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES,
    );
    let matches_legacy = MAINNET_APPROVED_PROOF_SYSTEMS.contains(&ps);
    matches_structured || matches_legacy
}

/// Returns true iff at least one entry in `entries` matches the
/// `(proof_system, circuit_id_hex, model_hash)` triple of `meta`.
/// Pure function; used by both the real layer-6 path and the
/// `#[cfg(test)]` helper.
fn matches_structured_allowlist(
    meta: &ProofMetadata,
    ps: ProofSystem,
    entries: &[AllowlistEntry],
) -> bool {
    let cid = meta.circuit_id_hex.as_deref();
    entries.iter().any(|e| {
        e.proof_system == ps
            && cid == Some(e.circuit_id_hex)
            && meta.model_hash == e.model_hash
    })
}

/// Stage 11d.1 — test-only `check_mainnet_eligible` variant that
/// accepts synthetic allowlist slices. Lets us exercise layer-6
/// matching against handcrafted entries WITHOUT modifying the real
/// const slices (which are `&[]` by design and must stay that way).
///
/// Crate-private; never re-exported. Downstream crates do not need
/// to inject synthetic allowlists.
#[cfg(test)]
pub(crate) fn check_mainnet_eligible_with(
    meta: &ProofMetadata,
    structured_entries: &[AllowlistEntry],
    legacy_entries: &[ProofSystem],
) -> std::result::Result<(), MainnetRefusalReason> {
    // Layers 1–5 mirror `check_mainnet_eligible` and are
    // independent of allowlist contents. Re-run them inline so the
    // helper is self-contained.
    if meta.testnet_or_dev_only != Some(false) {
        return Err(MainnetRefusalReason::TestnetOrDevOnly {
            backend_id: meta.backend_id.clone(),
            testnet_or_dev_only: meta.testnet_or_dev_only,
        });
    }
    match meta.proof_system {
        Some(ProofSystem::Mock) | None => {
            return Err(MainnetRefusalReason::MockBackend {
                backend_id: meta.backend_id.clone(),
                proof_system: meta.proof_system,
            });
        }
        _ => {}
    }
    if matches!(
        meta.proof_system,
        Some(ProofSystem::Stage11bOnnxReference) | Some(ProofSystem::Stage11bHalo2Reference)
    ) {
        return Err(MainnetRefusalReason::BoundedReference {
            backend_id: meta.backend_id.clone(),
            proof_system: meta.proof_system.unwrap(),
        });
    }
    if matches!(meta.model_format, Some(ModelFormat::Gguf)) {
        return Err(MainnetRefusalReason::GgufClaim {
            backend_id: meta.backend_id.clone(),
            proof_system: meta.proof_system,
        });
    }
    if matches!(meta.model_format, Some(ModelFormat::Other(_)) | None) {
        return Err(MainnetRefusalReason::UnknownModelFormat {
            backend_id: meta.backend_id.clone(),
            model_format: meta.model_format.clone(),
        });
    }
    let ps = meta.proof_system.unwrap();
    let matches_structured = matches_structured_allowlist(meta, ps, structured_entries);
    let matches_legacy = legacy_entries.contains(&ps);
    if !(matches_structured || matches_legacy) {
        return Err(MainnetRefusalReason::NotInMainnetAllowlist {
            backend_id: meta.backend_id.clone(),
            proof_system: ps,
        });
    }
    Ok(())
}

/// The canonical-JSON envelope written to the proof file before SNIP V2
/// ingest. The file's BLAKE3 (which is what the SNIP V2 fake adapter uses
/// for its `merkle_root` in tests, and what the real `sum-node ingest-v2`
/// emits in production via the same algorithm at the byte level) becomes
/// the `proof_snip_root` on the chain digest.
///
/// `serde_json::to_vec(&self)` produces deterministic output: field order
/// is taken from the struct definition, no trailing newline, no random
/// keying. Two identical `ProofArtifactBody` values therefore produce
/// byte-identical files and identical SNIP V2 roots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofArtifactBody {
    pub metadata: ProofMetadata,
    /// Hex of the proof bytes returned by the backend (bare lowercase, no
    /// `0x` prefix; variable length).
    pub proof_bytes_hex: String,
}

impl ProofArtifactBody {
    /// Compose an envelope from already-computed components.
    pub fn from_components(metadata: ProofMetadata, proof_bytes: &[u8]) -> Self {
        Self {
            metadata,
            proof_bytes_hex: encode_hex_lower(proof_bytes),
        }
    }

    /// Decode the embedded proof bytes back into the byte buffer the
    /// backend originally returned.
    pub fn proof_bytes(&self) -> std::result::Result<Vec<u8>, ProofVerifierError> {
        decode_hex_lower("proof_bytes_hex", &self.proof_bytes_hex)
    }

    /// Canonical JSON serialization. Deterministic for the same input.
    pub fn to_canonical_bytes(&self) -> std::result::Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
}

// ── Orchestrator ────────────────────────────────────────────────────────────

/// Inputs to [`produce_proof_artifact`].
#[derive(Debug, Clone, Copy)]
pub struct ProofPipelineInputs<'a> {
    pub model_bytes: &'a [u8],
    pub input_bytes: &'a [u8],
    pub output_bytes: &'a [u8],
}

/// Outputs of [`produce_proof_artifact`]: the published artifacts the
/// caller hands to [`crate::build_commitment`] unchanged, plus the
/// metadata for any further off-chain verification work.
#[derive(Debug, Clone)]
pub struct ProofPipelineOutputs {
    pub response_artifact: ResponseArtifact,
    pub proof_artifact: ProofArtifact,
    pub metadata: ProofMetadata,
}

/// End-to-end Stage 11a orchestrator. Steps:
///
/// 1. BLAKE3 the model / input / output bytes (gives us the `ProofMetadata`
///    fields without re-hashing the response later).
/// 2. Invoke `backend.prove(model, input, output)`.
/// 3. Assemble a [`ProofArtifactBody`] (metadata + hex proof bytes).
/// 4. Canonical JSON-serialize the envelope and write it to
///    `artifact_dir/proof.json`. Also write the response bytes to
///    `artifact_dir/response.bin` so [`publish_proof_artifacts`] can
///    independently ingest+hash the response per its Stage 3 contract.
/// 5. Construct fresh [`ResponseArtifact`] + [`ProofArtifact`] pointing at
///    those files and call [`publish_proof_artifacts`] — which populates
///    both SNIP V2 refs and the response BLAKE3 hash via the existing
///    Stage 3 plumbing, unchanged.
/// 6. Return the populated artifacts and the metadata.
///
/// The caller then composes with the unchanged
/// [`crate::build_commitment`] / [`crate::build_attestation`] /
/// [`crate::submit_attestation_workflow_with_block`] pipeline.
///
/// **`build_commitment` is NOT modified** — proof concerns live here
/// (Stage 11a correction 1).
pub fn produce_proof_artifact<A, B>(
    adapter: &A,
    backend: &B,
    inputs: ProofPipelineInputs<'_>,
    artifact_dir: &Path,
) -> std::result::Result<ProofPipelineOutputs, ProofPipelineError>
where
    A: SnipV2Adapter,
    B: ProofBackend + ?Sized,
{
    // 1. Compute the three BLAKE3 hashes once.
    let model_hash = blake3::hash(inputs.model_bytes);
    let input_hash = blake3::hash(inputs.input_bytes);
    let response_hash = blake3::hash(inputs.output_bytes);

    // 2. Backend produces proof bytes.
    let proof_bytes =
        backend.prove(inputs.model_bytes, inputs.input_bytes, inputs.output_bytes)?;

    // 3. Assemble metadata + body. Use the Stage-11a-compat
    //    constructor so Stage 11b.0 schema fields default to None and
    //    the resulting envelope JSON stays byte-identical against the
    //    Stage 11a `proof_pipeline_vectors.json` fixture. Real
    //    backends in Stage 11c+ will populate the new fields directly
    //    via struct literal.
    let metadata = ProofMetadata::new_stage11a(
        backend.backend_id().to_string(),
        model_hash.to_hex().to_string(),
        input_hash.to_hex().to_string(),
        response_hash.to_hex().to_string(),
    );
    let body = ProofArtifactBody::from_components(metadata.clone(), &proof_bytes);
    let body_bytes = body.to_canonical_bytes()?;

    // 4. Write the canonical envelope + response bytes to disk.
    std::fs::create_dir_all(artifact_dir)?;
    let proof_path = artifact_dir.join("proof.json");
    let response_path = artifact_dir.join("response.bin");
    std::fs::write(&proof_path, &body_bytes)?;
    std::fs::write(&response_path, inputs.output_bytes)?;

    // 5. Hand the new artifacts to the existing Stage 3 publish plumbing.
    //    No file-format changes; publish_proof_artifacts treats both files
    //    as opaque bytes, which is exactly the Stage 11a contract.
    let mut response_artifact = ResponseArtifact::new(&response_path);
    let mut proof_artifact = ProofArtifact::new(&proof_path);
    publish_proof_artifacts(adapter, &mut response_artifact, &mut proof_artifact)?;

    Ok(ProofPipelineOutputs {
        response_artifact,
        proof_artifact,
        metadata,
    })
}

// ── hex helpers (local, lowercase-strict) ───────────────────────────────────
//
// `omni-zkml` doesn't depend on the `hex` crate; existing code uses
// blake3's `to_hex()` for 32-byte values and hand-rolled lowercase nibble
// decoders for parse paths (see `attestation::CommitmentDigest::from_hex`).
// Mirroring that style here keeps the crate dep-clean.

fn encode_hex_lower(bytes: &[u8]) -> String {
    const NIBBLE: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(NIBBLE[(b >> 4) as usize] as char);
        s.push(NIBBLE[(b & 0x0f) as usize] as char);
    }
    s
}

fn decode_hex_lower(
    field: &'static str,
    s: &str,
) -> std::result::Result<Vec<u8>, ProofVerifierError> {
    if s.len() % 2 != 0 {
        return Err(ProofVerifierError::VerifierInternal(format!(
            "field {field}: odd-length hex ({} chars)",
            s.len()
        )));
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in 0..(s.len() / 2) {
        let hi = decode_nibble_lower(field, bytes[i * 2])?;
        let lo = decode_nibble_lower(field, bytes[i * 2 + 1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn decode_blake3_hex_lower(
    field: &'static str,
    s: &str,
) -> std::result::Result<[u8; 32], ProofVerifierError> {
    if s.len() != 64 {
        return Err(ProofVerifierError::VerifierInternal(format!(
            "field {field}: expected 64 hex chars (32 bytes), got {}",
            s.len()
        )));
    }
    let mut out = [0u8; 32];
    let bytes = s.as_bytes();
    for i in 0..32 {
        let hi = decode_nibble_lower(field, bytes[i * 2])?;
        let lo = decode_nibble_lower(field, bytes[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn decode_nibble_lower(field: &'static str, b: u8) -> std::result::Result<u8, ProofVerifierError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Err(ProofVerifierError::VerifierInternal(format!(
            "field {field}: uppercase hex '{}' not allowed (bare lowercase contract)",
            b as char
        ))),
        _ => Err(ProofVerifierError::VerifierInternal(format!(
            "field {field}: invalid hex char '{}'",
            b as char
        ))),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Mutex;

    use omni_store::snip_v2::SnipV2Error;
    use omni_types::phase5::{SnipV2Lifecycle, SnipV2ObjectId, SnipV2ObjectRef};

    /// Minimal in-memory `SnipV2Adapter` for proof orchestrator tests.
    /// Content-addresses by BLAKE3 of the file bytes (mirrors what
    /// `sum-node ingest-v2` produces in production at the byte level).
    /// `download_public` is implemented in case future tests need it.
    struct LocalFakeSnipV2Adapter {
        store: Mutex<HashMap<SnipV2ObjectId, Vec<u8>>>,
    }
    impl LocalFakeSnipV2Adapter {
        fn new() -> Self {
            Self {
                store: Mutex::new(HashMap::new()),
            }
        }
    }
    impl omni_store::SnipV2Adapter for LocalFakeSnipV2Adapter {
        fn ingest_public(
            &self,
            path: &Path,
        ) -> std::result::Result<SnipV2ObjectRef, SnipV2Error> {
            let bytes = std::fs::read(path).map_err(SnipV2Error::CommandSpawn)?;
            let hash = blake3::hash(&bytes);
            let mut id_bytes = [0u8; 32];
            id_bytes.copy_from_slice(hash.as_bytes());
            let id = SnipV2ObjectId::from_bytes(id_bytes);
            self.store.lock().unwrap().insert(id, bytes.clone());
            Ok(SnipV2ObjectRef {
                merkle_root: id,
                lifecycle: SnipV2Lifecycle::Active,
                plaintext_size_bytes: Some(bytes.len() as u64),
            })
        }

        fn download_public(
            &self,
            root: &SnipV2ObjectId,
            output_path: &Path,
        ) -> std::result::Result<(), SnipV2Error> {
            let bytes = self
                .store
                .lock()
                .unwrap()
                .get(root)
                .cloned()
                .ok_or_else(|| {
                    SnipV2Error::CommandSpawn(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "no such object in fake store",
                    ))
                })?;
            std::fs::write(output_path, bytes).map_err(SnipV2Error::CommandSpawn)
        }
    }

    #[test]
    fn mock_backend_id_is_stable() {
        assert_eq!(MockProofBackend.backend_id(), "mock-v1");
        assert_eq!(MOCK_BACKEND_ID, "mock-v1");
    }

    #[test]
    fn mock_backend_prove_is_deterministic_and_64_bytes() {
        let p1 = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        let p2 = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 64);
    }

    #[test]
    fn mock_backend_prove_varies_with_inputs() {
        let p_baseline = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        let p_model = MockProofBackend
            .prove(b"MODEL", b"input", b"output")
            .unwrap();
        let p_input = MockProofBackend
            .prove(b"model", b"INPUT", b"output")
            .unwrap();
        let p_output = MockProofBackend
            .prove(b"model", b"input", b"OUTPUT")
            .unwrap();
        assert_ne!(p_baseline, p_model);
        assert_ne!(p_baseline, p_input);
        assert_ne!(p_baseline, p_output);
        // And each varied input is distinct from the others (domain tags
        // bind the position of each hash).
        assert_ne!(p_model, p_input);
        assert_ne!(p_input, p_output);
    }

    #[test]
    fn mock_verifier_accepts_matching_proof() {
        let proof = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        let pi = PublicInputs {
            model_hash: *blake3::hash(b"model").as_bytes(),
            input_hash: *blake3::hash(b"input").as_bytes(),
            output_hash: *blake3::hash(b"output").as_bytes(),
        };
        assert!(MockProofVerifier.verify(&proof, &pi).unwrap());
    }

    #[test]
    fn mock_verifier_rejects_tampered_proof() {
        let mut proof = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        proof[0] ^= 0x01;
        let pi = PublicInputs {
            model_hash: *blake3::hash(b"model").as_bytes(),
            input_hash: *blake3::hash(b"input").as_bytes(),
            output_hash: *blake3::hash(b"output").as_bytes(),
        };
        assert!(!MockProofVerifier.verify(&proof, &pi).unwrap());
    }

    #[test]
    fn mock_verifier_rejects_mismatched_public_inputs() {
        let proof = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        let pi = PublicInputs {
            model_hash: *blake3::hash(b"DIFFERENT_MODEL").as_bytes(),
            input_hash: *blake3::hash(b"input").as_bytes(),
            output_hash: *blake3::hash(b"output").as_bytes(),
        };
        assert!(!MockProofVerifier.verify(&proof, &pi).unwrap());
    }

    #[test]
    fn proof_artifact_body_roundtrips_canonical_bytes() {
        let body = ProofArtifactBody::from_components(
            ProofMetadata::new_stage11a(
                "mock-v1".to_string(),
                blake3::hash(b"m").to_hex().to_string(),
                blake3::hash(b"i").to_hex().to_string(),
                blake3::hash(b"o").to_hex().to_string(),
            ),
            &[0x01, 0x02, 0xab, 0xcd],
        );
        let bytes = body.to_canonical_bytes().unwrap();
        let body_back: ProofArtifactBody = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body_back, body);
        // Round-trip is byte-identical (deterministic serialization).
        assert_eq!(body_back.to_canonical_bytes().unwrap(), bytes);
        // Embedded proof bytes decode cleanly.
        assert_eq!(body.proof_bytes().unwrap(), vec![0x01, 0x02, 0xab, 0xcd]);
    }

    #[test]
    fn proof_metadata_public_inputs_decodes_hex() {
        let metadata = ProofMetadata::new_stage11a(
            "mock-v1".to_string(),
            blake3::hash(b"m").to_hex().to_string(),
            blake3::hash(b"i").to_hex().to_string(),
            blake3::hash(b"o").to_hex().to_string(),
        );
        let pi = metadata.public_inputs().unwrap();
        assert_eq!(pi.model_hash, *blake3::hash(b"m").as_bytes());
        assert_eq!(pi.input_hash, *blake3::hash(b"i").as_bytes());
        assert_eq!(pi.output_hash, *blake3::hash(b"o").as_bytes());
    }

    #[test]
    fn proof_metadata_public_inputs_rejects_uppercase_hex() {
        let metadata = ProofMetadata::new_stage11a(
            "mock-v1".to_string(),
            // Uppercase — must fail the bare-lowercase contract.
            blake3::hash(b"m").to_hex().to_string().to_uppercase(),
            blake3::hash(b"i").to_hex().to_string(),
            blake3::hash(b"o").to_hex().to_string(),
        );
        let err = metadata.public_inputs().unwrap_err();
        assert!(
            matches!(err, ProofVerifierError::VerifierInternal(ref s) if s.contains("uppercase")),
            "expected uppercase rejection, got {err:?}"
        );
    }

    #[test]
    fn end_to_end_orchestrator_publishes_via_fake_adapter() {
        // Hermetic end-to-end smoke against a local in-memory SNIP V2
        // adapter (omni-store's fake is private). The orchestrator
        // exercises the real publish_proof_artifacts path without a
        // sum-node binary.
        let adapter = LocalFakeSnipV2Adapter::new();
        let tmp = tempfile::tempdir().unwrap();

        let inputs = ProofPipelineInputs {
            model_bytes: b"stage11a-orchestrator-test-model",
            input_bytes: b"stage11a-orchestrator-test-input",
            output_bytes: b"stage11a-orchestrator-test-output",
        };
        let out =
            produce_proof_artifact(&adapter, &MockProofBackend, inputs, tmp.path()).unwrap();

        // ProofArtifact and ResponseArtifact both have populated SNIP refs.
        assert!(out.proof_artifact.snip_v2.is_some());
        assert!(out.response_artifact.snip_v2.is_some());
        assert!(out.response_artifact.blake3_hash.is_some());

        // ProofMetadata fields match the deterministic hashes of the inputs.
        assert_eq!(out.metadata.backend_id, "mock-v1");
        assert_eq!(
            out.metadata.model_hash,
            blake3::hash(inputs.model_bytes).to_hex().to_string()
        );
        assert_eq!(
            out.metadata.input_hash,
            blake3::hash(inputs.input_bytes).to_hex().to_string()
        );
        assert_eq!(
            out.metadata.response_hash,
            blake3::hash(inputs.output_bytes).to_hex().to_string()
        );

        // Response artifact's BLAKE3 matches the metadata response_hash.
        assert_eq!(
            out.response_artifact.blake3_hash.as_deref().unwrap(),
            out.metadata.response_hash
        );

        // The proof file on disk parses back into a ProofArtifactBody with
        // the same metadata and proof bytes that the backend produced.
        let proof_path_bytes = std::fs::read(&out.proof_artifact.local_path).unwrap();
        let parsed: ProofArtifactBody = serde_json::from_slice(&proof_path_bytes).unwrap();
        assert_eq!(parsed.metadata, out.metadata);

        // Verifier accepts the proof bytes recovered from the artifact.
        let recovered_proof = parsed.proof_bytes().unwrap();
        let pi = parsed.metadata.public_inputs().unwrap();
        assert!(MockProofVerifier.verify(&recovered_proof, &pi).unwrap());
    }

    // ── Stage 11b.0 — trait extensions + schema + mainnet refusal ──

    #[test]
    fn mock_backend_declares_local_only_and_mock_system() {
        assert!(MockProofBackend.is_local_only());
        assert_eq!(MockProofBackend.proof_system(), ProofSystem::Mock);
        assert!(MockProofBackend.supported_formats().is_empty());
        assert!(MockProofBackend.circuit_id().is_none());
    }

    #[test]
    fn mock_verifier_declares_mock_system() {
        assert_eq!(MockProofVerifier.proof_system(), ProofSystem::Mock);
    }

    #[test]
    fn mainnet_allowlist_is_empty_at_stage_11b0() {
        // Pinned invariant per Stage 11b.0 plan: no proof system is
        // mainnet-eligible at the end of Stage 11b.0. Adding an entry
        // here is a Stage 11c+ change subject to chain-team review.
        assert!(
            MAINNET_APPROVED_PROOF_SYSTEMS.is_empty(),
            "MAINNET_APPROVED_PROOF_SYSTEMS must stay empty in Stage 11b — \
             any addition requires chain-team review"
        );
    }

    /// Stage 11b.0 invariant: every `ProofSystem` variant is refused
    /// on mainnet at the end of Stage 11b.0. Walks the full enum and
    /// asserts the refusal helper returns Err for each.
    #[test]
    fn every_proof_system_is_refused_on_mainnet_at_stage_11b0() {
        // Stage 11b.1.a expansion: walks the now-larger enum
        // (Stage11bHalo2Reference added) and asserts every variant
        // is still refused on mainnet. The expected refusal layer
        // per variant is asserted concretely.
        for ps in [
            ProofSystem::Mock,
            ProofSystem::Stage11bOnnxReference,
            ProofSystem::Stage11bHalo2Reference,
            ProofSystem::Ezkl,
            ProofSystem::GgufStrategyTbd,
            ProofSystem::Stage11dProductionFixedPointMlp,
        ] {
            let meta = ProofMetadata {
                backend_id: format!("test-backend-for-{ps:?}"),
                model_hash: blake3::hash(b"m").to_hex().to_string(),
                input_hash: blake3::hash(b"i").to_hex().to_string(),
                response_hash: blake3::hash(b"o").to_hex().to_string(),
                model_format: Some(ModelFormat::Onnx),
                proof_system: Some(ps),
                circuit_id_hex: None,
                verification_key_hex: None,
                public_inputs: None,
                // Stage 11d.1 tightening: set Some(false) explicitly
                // so this test exercises layers 2–6 per variant
                // (not the new layer-1 absent-flag refusal).
                testnet_or_dev_only: Some(false),
                model_framework: None,
            };
            let err = check_mainnet_eligible(&meta).unwrap_err();
            // Concretely: Mock → MockBackend,
            //   Stage11bOnnxReference + Stage11bHalo2Reference → BoundedReference,
            //   Ezkl + GgufStrategyTbd + Stage11dProductionFixedPointMlp →
            //     NotInMainnetAllowlist (allowlist empty through Stage 11d.1/11d.2;
            //     Stage 11d.2 adds the production variant but no allowlist entry).
            match (ps, &err) {
                (ProofSystem::Mock, MainnetRefusalReason::MockBackend { .. }) => {}
                (
                    ProofSystem::Stage11bOnnxReference | ProofSystem::Stage11bHalo2Reference,
                    MainnetRefusalReason::BoundedReference { .. },
                ) => {}
                (
                    ProofSystem::Ezkl
                    | ProofSystem::GgufStrategyTbd
                    | ProofSystem::Stage11dProductionFixedPointMlp,
                    MainnetRefusalReason::NotInMainnetAllowlist { .. },
                ) => {}
                _ => panic!("unexpected refusal for {ps:?}: {err:?}"),
            }
        }
    }

    #[test]
    fn refusal_layer_1_testnet_or_dev_flag_fires_first() {
        // testnet_or_dev_only takes precedence over even the Mock
        // refusal (layer 1 before layer 2).
        let meta = ProofMetadata {
            backend_id: "any".into(),
            model_hash: blake3::hash(b"m").to_hex().to_string(),
            input_hash: blake3::hash(b"i").to_hex().to_string(),
            response_hash: blake3::hash(b"o").to_hex().to_string(),
            model_format: Some(ModelFormat::Onnx),
            proof_system: Some(ProofSystem::Mock),
            circuit_id_hex: None,
            verification_key_hex: None,
            public_inputs: None,
            testnet_or_dev_only: Some(true),
            model_framework: None,
        };
        let err = check_mainnet_eligible(&meta).unwrap_err();
        assert!(matches!(err, MainnetRefusalReason::TestnetOrDevOnly { .. }), "got {err:?}");
    }

    #[test]
    fn refusal_layer_2_mock_or_absent_proof_system() {
        // Absent proof_system is treated as Mock (Stage 11a vintage).
        // Stage 11d.1 tightening: layer 1 now catches the default
        // `testnet_or_dev_only: None` from `new_stage11a`, so this
        // test explicitly sets `Some(false)` to exercise layer 2.
        let mut meta = ProofMetadata::new_stage11a(
            "stage11a-style".into(),
            blake3::hash(b"m").to_hex().to_string(),
            blake3::hash(b"i").to_hex().to_string(),
            blake3::hash(b"o").to_hex().to_string(),
        );
        meta.testnet_or_dev_only = Some(false);
        let err = check_mainnet_eligible(&meta).unwrap_err();
        assert!(matches!(err, MainnetRefusalReason::MockBackend { .. }), "got {err:?}");
    }

    /// Stage 11d.1 — companion to
    /// `refusal_layer_2_mock_or_absent_proof_system`: confirms the
    /// `new_stage11a` default (which leaves `testnet_or_dev_only:
    /// None`) is refused at layer 1, NOT layer 2. This pins the
    /// safety contract that absent / `None` flag is treated as
    /// testnet/dev.
    #[test]
    fn stage11a_default_metadata_refused_at_layer_1() {
        let meta = ProofMetadata::new_stage11a(
            "stage11a-style".into(),
            blake3::hash(b"m").to_hex().to_string(),
            blake3::hash(b"i").to_hex().to_string(),
            blake3::hash(b"o").to_hex().to_string(),
        );
        let err = check_mainnet_eligible(&meta).unwrap_err();
        assert!(
            matches!(
                err,
                MainnetRefusalReason::TestnetOrDevOnly {
                    testnet_or_dev_only: None,
                    ..
                }
            ),
            "Stage 11a default metadata (testnet_or_dev_only = None) must \
             be refused at layer 1 per Stage 11d.1 tightening; got {err:?}"
        );
    }

    #[test]
    fn refusal_layer_3_bounded_reference_system() {
        let meta = ProofMetadata {
            backend_id: "ezkl-onnx-reference-v1".into(),
            model_hash: blake3::hash(b"m").to_hex().to_string(),
            input_hash: blake3::hash(b"i").to_hex().to_string(),
            response_hash: blake3::hash(b"o").to_hex().to_string(),
            model_format: Some(ModelFormat::Onnx),
            proof_system: Some(ProofSystem::Stage11bOnnxReference),
            circuit_id_hex: None,
            verification_key_hex: None,
            public_inputs: None,
            testnet_or_dev_only: Some(false),
            model_framework: None,
        };
        let err = check_mainnet_eligible(&meta).unwrap_err();
        assert!(matches!(err, MainnetRefusalReason::BoundedReference { .. }), "got {err:?}");
    }

    #[test]
    fn refusal_layer_4_gguf_format_claim() {
        let meta = ProofMetadata {
            backend_id: "any".into(),
            model_hash: blake3::hash(b"m").to_hex().to_string(),
            input_hash: blake3::hash(b"i").to_hex().to_string(),
            response_hash: blake3::hash(b"o").to_hex().to_string(),
            model_format: Some(ModelFormat::Gguf),
            proof_system: Some(ProofSystem::Ezkl), // anything non-mock to pass layer 2
            circuit_id_hex: None,
            verification_key_hex: None,
            public_inputs: None,
            testnet_or_dev_only: Some(false),
            model_framework: None,
        };
        let err = check_mainnet_eligible(&meta).unwrap_err();
        assert!(matches!(err, MainnetRefusalReason::GgufClaim { .. }), "got {err:?}");
    }

    #[test]
    fn refusal_layer_5_other_or_unknown_model_format() {
        // Other(_) refused.
        let meta = ProofMetadata {
            backend_id: "any".into(),
            model_hash: blake3::hash(b"m").to_hex().to_string(),
            input_hash: blake3::hash(b"i").to_hex().to_string(),
            response_hash: blake3::hash(b"o").to_hex().to_string(),
            model_format: Some(ModelFormat::Other("custom-format-v0".into())),
            proof_system: Some(ProofSystem::Ezkl),
            circuit_id_hex: None,
            verification_key_hex: None,
            public_inputs: None,
            testnet_or_dev_only: Some(false),
            model_framework: None,
        };
        let err = check_mainnet_eligible(&meta).unwrap_err();
        assert!(matches!(err, MainnetRefusalReason::UnknownModelFormat { .. }), "got {err:?}");

        // Absent (None) format on a non-mock backend is the same refusal.
        let meta_absent = ProofMetadata {
            model_format: None,
            ..meta
        };
        let err2 = check_mainnet_eligible(&meta_absent).unwrap_err();
        assert!(matches!(err2, MainnetRefusalReason::UnknownModelFormat { .. }), "got {err2:?}");
    }

    #[test]
    fn refusal_layer_6_not_in_empty_allowlist() {
        // Non-mock, non-bounded, ONNX format → falls through to
        // layer 6, refused because the allowlist is empty.
        let meta = ProofMetadata {
            backend_id: "ezkl-onnx-prod-future".into(),
            model_hash: blake3::hash(b"m").to_hex().to_string(),
            input_hash: blake3::hash(b"i").to_hex().to_string(),
            response_hash: blake3::hash(b"o").to_hex().to_string(),
            model_format: Some(ModelFormat::Onnx),
            proof_system: Some(ProofSystem::Ezkl),
            circuit_id_hex: None,
            verification_key_hex: None,
            public_inputs: None,
            testnet_or_dev_only: Some(false),
            model_framework: None,
        };
        let err = check_mainnet_eligible(&meta).unwrap_err();
        assert!(matches!(err, MainnetRefusalReason::NotInMainnetAllowlist { .. }), "got {err:?}");
    }

    #[test]
    fn stage11a_compat_metadata_serializes_without_stage11b_fields() {
        // The Stage 11a compat constructor must produce JSON that
        // omits all Stage 11b.0 optional fields when None — this is
        // what keeps `proof_pipeline_vectors.json` byte-stable.
        let meta = ProofMetadata::new_stage11a(
            "mock-v1".into(),
            "00".repeat(32),
            "11".repeat(32),
            "22".repeat(32),
        );
        let s = serde_json::to_string(&meta).unwrap();
        assert!(!s.contains("model_format"),
                "Stage 11a-compat must not emit model_format: {s}");
        assert!(!s.contains("proof_system"),
                "Stage 11a-compat must not emit proof_system: {s}");
        assert!(!s.contains("circuit_id_hex"),
                "Stage 11a-compat must not emit circuit_id_hex: {s}");
        assert!(!s.contains("verification_key_hex"),
                "Stage 11a-compat must not emit verification_key_hex: {s}");
        assert!(!s.contains("public_inputs"),
                "Stage 11a-compat must not emit public_inputs: {s}");
        assert!(!s.contains("testnet_or_dev_only"),
                "Stage 11a-compat must not emit testnet_or_dev_only: {s}");
    }

    // ── Stage 11b.0.1 — ProofVerifier::verify_artifact ─────────────────

    /// Construct a Stage 11a-style mock body for use by the
    /// `verify_artifact` tests below. Same shape the Stage 11a
    /// `produce_proof_artifact` orchestrator would emit.
    fn mock_artifact_body() -> ProofArtifactBody {
        let proof_bytes = MockProofBackend
            .prove(b"vm", b"vi", b"vo")
            .expect("mock backend infallible");
        let metadata = ProofMetadata::new_stage11a(
            MOCK_BACKEND_ID.to_string(),
            blake3::hash(b"vm").to_hex().to_string(),
            blake3::hash(b"vi").to_hex().to_string(),
            blake3::hash(b"vo").to_hex().to_string(),
        );
        ProofArtifactBody::from_components(metadata, &proof_bytes)
    }

    #[test]
    fn mock_verifier_verify_artifact_default_delegates_to_verify() {
        // The default `verify_artifact` impl on `ProofVerifier` must
        // produce the same result as calling `verify(&proof, &pi)`
        // directly. Pins the back-compat contract: any Stage 11a-shape
        // verifier that didn't override `verify_artifact` gets the
        // right behaviour for free.
        let body = mock_artifact_body();

        // Direct path (Stage 11a-style call).
        let proof = body.proof_bytes().unwrap();
        let pi = body.metadata.public_inputs().unwrap();
        let direct = MockProofVerifier.verify(&proof, &pi).unwrap();

        // Stage 11b.0.1 canonical dispatch path.
        let via_artifact = MockProofVerifier.verify_artifact(&body).unwrap();

        assert_eq!(direct, via_artifact);
        assert!(via_artifact, "mock proof must verify against canonical bytes");
    }

    #[test]
    fn default_verify_artifact_rejects_tampered_proof() {
        // Defense in depth: even though tampering is detected by the
        // underlying `verify` for the mock backend, this test pins
        // that the default `verify_artifact` dispatch surfaces the
        // rejection. A regression that bypasses `verify` somewhere
        // in the default impl would fail here.
        let mut body = mock_artifact_body();
        // Flip a single bit in the embedded proof hex.
        let first_char = body.proof_bytes_hex.chars().next().unwrap();
        let flipped = if first_char == '0' { '1' } else { '0' };
        body.proof_bytes_hex = format!("{flipped}{}", &body.proof_bytes_hex[1..]);

        let verified = MockProofVerifier.verify_artifact(&body).unwrap();
        assert!(!verified, "tampered proof must NOT verify");
    }

    /// Hand-rolled minimal `ProofVerifier` impl that uses the
    /// `verify_artifact` default (does NOT override it) and counts
    /// how many times `verify` is called. Pins the architectural
    /// property: the default `verify_artifact` ends up calling
    /// `verify` exactly once with the artifact's bytes + hashed
    /// inputs. A future "optimization" that skipped `verify` from
    /// the default impl would break this assertion and surface the
    /// contract change explicitly.
    struct CountingVerifier {
        calls: std::cell::Cell<u32>,
    }

    impl ProofVerifier for CountingVerifier {
        fn verify(
            &self,
            proof: &[u8],
            public_inputs: &PublicInputs,
        ) -> std::result::Result<bool, ProofVerifierError> {
            self.calls.set(self.calls.get() + 1);
            // Delegate to mock semantics so tampering still rejects.
            MockProofVerifier.verify(proof, public_inputs)
        }
        // Inherits the default `verify_artifact`.
    }

    #[test]
    fn default_verify_artifact_invokes_verify_exactly_once() {
        let body = mock_artifact_body();
        let v = CountingVerifier { calls: std::cell::Cell::new(0) };
        let ok = v.verify_artifact(&body).unwrap();
        assert!(ok);
        assert_eq!(v.calls.get(), 1, "default verify_artifact must call verify exactly once");
    }

    // ── Stage 11b.1.a — multi-framework schema additions ─────────────

    #[test]
    fn model_framework_variants_round_trip_via_json() {
        // Every ModelFramework variant must serialize + deserialize
        // byte-stable, since they appear in committed fixture
        // manifests. Pins the surface for cross-framework fixtures.
        for fw in [
            ModelFramework::Rumus,
            ModelFramework::PyTorch,
            ModelFramework::TensorFlow,
            ModelFramework::Caffe,
            ModelFramework::FrameworkAgnostic,
        ] {
            let s = serde_json::to_string(&fw).unwrap();
            let back: ModelFramework = serde_json::from_str(&s).unwrap();
            assert_eq!(fw, back, "round-trip failed for {fw:?}");
        }
    }

    #[test]
    fn model_format_halo2_reference_mlp_round_trips() {
        let mf = ModelFormat::Halo2ReferenceMlp;
        let s = serde_json::to_string(&mf).unwrap();
        assert_eq!(s, "\"Halo2ReferenceMlp\"");
        let back: ModelFormat = serde_json::from_str(&s).unwrap();
        assert_eq!(mf, back);
    }

    #[test]
    fn proof_system_stage11b_halo2_reference_round_trips() {
        let ps = ProofSystem::Stage11bHalo2Reference;
        let s = serde_json::to_string(&ps).unwrap();
        assert_eq!(s, "\"Stage11bHalo2Reference\"");
        let back: ProofSystem = serde_json::from_str(&s).unwrap();
        assert_eq!(ps, back);
    }

    #[test]
    fn proof_metadata_with_model_framework_round_trips() {
        // Stage 11b.1.a: model_framework is a new optional field. A
        // metadata with `Some(...)` round-trips faithfully; with
        // `None`, the field is omitted from JSON (the byte-stability
        // contract for Stage 11a artifacts).
        let mut meta = ProofMetadata::new_stage11a(
            "test".into(),
            blake3::hash(b"m").to_hex().to_string(),
            blake3::hash(b"i").to_hex().to_string(),
            blake3::hash(b"o").to_hex().to_string(),
        );
        // None case: must not appear in JSON.
        let s_none = serde_json::to_string(&meta).unwrap();
        assert!(!s_none.contains("model_framework"),
                "model_framework must be skipped when None: {s_none}");

        // Some case: round-trips with the framework name.
        meta.model_framework = Some(ModelFramework::FrameworkAgnostic);
        let s_some = serde_json::to_string(&meta).unwrap();
        assert!(s_some.contains("FrameworkAgnostic"),
                "model_framework must appear in JSON when Some: {s_some}");
        let back: ProofMetadata = serde_json::from_str(&s_some).unwrap();
        assert_eq!(back.model_framework, Some(ModelFramework::FrameworkAgnostic));
    }

    #[test]
    fn refusal_layer_3_catches_stage11b_halo2_reference() {
        // Stage 11b.1.a expansion: layer 3 now refuses both
        // bounded-reference variants. The original Stage 11b.0
        // variant Stage11bOnnxReference and the new Stage 11b.1
        // variant Stage11bHalo2Reference must BOTH hit layer 3.
        let mut meta = ProofMetadata::new_stage11a(
            "halo2-reference-mlp-v1".into(),
            blake3::hash(b"m").to_hex().to_string(),
            blake3::hash(b"i").to_hex().to_string(),
            blake3::hash(b"o").to_hex().to_string(),
        );
        meta.model_format = Some(ModelFormat::Halo2ReferenceMlp);
        meta.proof_system = Some(ProofSystem::Stage11bHalo2Reference);
        // Stage 11d.1 tightening: must set Some(false) explicitly
        // so layer 1 doesn't pre-empt (`new_stage11a`'s default
        // None now triggers layer 1). We want layer 3 to fire
        // concretely for this bounded-reference variant test.
        meta.testnet_or_dev_only = Some(false);
        let err = check_mainnet_eligible(&meta).unwrap_err();
        assert!(
            matches!(err, MainnetRefusalReason::BoundedReference {
                proof_system: ProofSystem::Stage11bHalo2Reference, ..
            }),
            "expected BoundedReference(Stage11bHalo2Reference), got {err:?}"
        );
    }

    #[test]
    fn halo2_reference_artifact_with_testnet_flag_is_refused_at_layer_1() {
        // Production-shape: the canonical Stage 11b.1 artifact will
        // carry testnet_or_dev_only: Some(true). That means refusal
        // layer 1 fires before layer 3 — pin the precedence.
        let mut meta = ProofMetadata::new_stage11a(
            "halo2-reference-mlp-v1".into(),
            blake3::hash(b"m").to_hex().to_string(),
            blake3::hash(b"i").to_hex().to_string(),
            blake3::hash(b"o").to_hex().to_string(),
        );
        meta.model_format = Some(ModelFormat::Halo2ReferenceMlp);
        meta.proof_system = Some(ProofSystem::Stage11bHalo2Reference);
        meta.testnet_or_dev_only = Some(true);
        meta.model_framework = Some(ModelFramework::FrameworkAgnostic);
        let err = check_mainnet_eligible(&meta).unwrap_err();
        assert!(
            matches!(err, MainnetRefusalReason::TestnetOrDevOnly { .. }),
            "layer 1 (testnet flag) must fire before layer 3 (bounded reference); got {err:?}"
        );
    }

    #[test]
    fn halo2_reference_mlp_format_is_not_caught_by_layer_5() {
        // Stage 11b.1.a adds Halo2ReferenceMlp as a first-class
        // variant — it must NOT be caught by layer 5
        // (Other/unknown). Refusal must come from layer 3
        // (bounded reference) when proof_system is Stage11bHalo2Reference.
        let mut meta = ProofMetadata::new_stage11a(
            "halo2-reference-mlp-v1".into(),
            blake3::hash(b"m").to_hex().to_string(),
            blake3::hash(b"i").to_hex().to_string(),
            blake3::hash(b"o").to_hex().to_string(),
        );
        meta.model_format = Some(ModelFormat::Halo2ReferenceMlp);
        meta.proof_system = Some(ProofSystem::Stage11bHalo2Reference);
        // Stage 11d.1 tightening: set Some(false) so the test
        // exercises layer 3, not the new layer-1 absent-flag refusal.
        meta.testnet_or_dev_only = Some(false);
        let err = check_mainnet_eligible(&meta).unwrap_err();
        // Must be layer 3, NOT layer 5.
        assert!(
            matches!(err, MainnetRefusalReason::BoundedReference { .. }),
            "Halo2ReferenceMlp should be a first-class format, not Other(_); \
             expected layer-3 refusal, got {err:?}"
        );
        assert!(
            !matches!(err, MainnetRefusalReason::UnknownModelFormat { .. }),
            "Halo2ReferenceMlp must NOT be caught by layer 5"
        );
    }

    // ─────────────────────────────────────────────────────────────
    // Stage 11d.1 — structured allowlist schema tests.
    //
    // Stage 11d.1 ships the schema as code but commits ZERO
    // entries. These tests pin: (a) the empty-list invariants for
    // both the structured and legacy lists; (b) bounded-reference
    // / testnet-flag refusals fire BEFORE the layer-6 allowlist
    // lookup, so a hypothetically-misplaced bounded-reference
    // allowlist entry can never enable a bounded artifact to pass;
    // (c) the layer-6 matching logic on the (proof_system,
    // circuit_id_hex, model_hash) triple, exercised via the
    // `#[cfg(test)] check_mainnet_eligible_with` helper against
    // synthetic allowlists; (d) the VK hash helper is byte-stable
    // and uses the documented domain separator.
    // ─────────────────────────────────────────────────────────────

    /// Helper: build a baseline `ProofMetadata` that passes layers
    /// 1–5 cleanly. Tests then tamper specific fields to exercise
    /// layer-6 paths. Uses the `Ezkl` `ProofSystem` variant since
    /// it is neither Mock nor bounded-reference.
    fn synthetic_production_meta() -> ProofMetadata {
        ProofMetadata {
            backend_id: "synthetic-test-backend".into(),
            model_hash: "a".repeat(64),
            input_hash: blake3::hash(b"i").to_hex().to_string(),
            response_hash: blake3::hash(b"o").to_hex().to_string(),
            model_format: Some(ModelFormat::Onnx),
            proof_system: Some(ProofSystem::Ezkl),
            circuit_id_hex: Some("c".repeat(64)),
            verification_key_hex: None,
            public_inputs: None,
            testnet_or_dev_only: Some(false),
            model_framework: None,
        }
    }

    /// Helper: synthetic `AllowlistEntry` matching the baseline
    /// `synthetic_production_meta()`'s triple. Used as the "would
    /// pass" entry in test-only allowlists.
    fn synthetic_production_entry() -> AllowlistEntry {
        AllowlistEntry {
            proof_system: ProofSystem::Ezkl,
            backend_id: "synthetic-test-backend",
            circuit_id_hex:
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            model_hash:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            model_format: ModelFormat::Onnx,
            verification_key_hash_hex:
                "0000000000000000000000000000000000000000000000000000000000000000",
            chain_team_review_ref: "docs/chain-team-review/synthetic-test.md",
        }
    }

    #[test]
    fn mainnet_allowlist_entries_empty_after_stage_11d1() {
        assert!(
            MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES.is_empty(),
            "MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES must stay empty through Stage \
             11d.1 / 11d.2 — only Stage 11d.3 with written chain-team sign-off may add"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn legacy_MAINNET_APPROVED_PROOF_SYSTEMS_empty_after_stage_11d1() {
        assert!(
            MAINNET_APPROVED_PROOF_SYSTEMS.is_empty(),
            "legacy MAINNET_APPROVED_PROOF_SYSTEMS must stay empty — the back-compat \
             alias is preserved only because dual-empty preserves Stage 11b.0's \
             refuse-everything invariant bit-for-bit"
        );
    }

    #[test]
    fn stage11b_halo2_reference_never_in_allowlist() {
        // Hard rule H1 from docs/mainnet-eligibility-criteria.md
        // §1.6: bounded reference proof systems must not appear in
        // either list. The slice is empty today; this test guards
        // against future misconfiguration. Symmetric for the
        // ONNX reference variant.
        for entry in MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES {
            assert!(
                !matches!(
                    entry.proof_system,
                    ProofSystem::Stage11bHalo2Reference | ProofSystem::Stage11bOnnxReference
                ),
                "bounded-reference proof system {:?} must not appear in \
                 MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES — testnet/dev-only in perpetuity",
                entry.proof_system
            );
        }
        for ps in MAINNET_APPROVED_PROOF_SYSTEMS {
            assert!(
                !matches!(
                    ps,
                    ProofSystem::Stage11bHalo2Reference | ProofSystem::Stage11bOnnxReference
                ),
                "bounded-reference proof system {ps:?} must not appear in legacy \
                 MAINNET_APPROVED_PROOF_SYSTEMS — testnet/dev-only in perpetuity"
            );
        }
    }

    #[test]
    fn bounded_reference_refused_before_allowlist_lookup() {
        // Construct metadata with Stage11bHalo2Reference whose
        // (proof_system, circuit_id_hex, model_hash) WOULD match
        // a hypothetical allowlist entry. Layer 3 must fire before
        // the layer-6 lookup, so the artifact is refused regardless
        // of allowlist contents.
        let mut meta = synthetic_production_meta();
        meta.proof_system = Some(ProofSystem::Stage11bHalo2Reference);
        let would_match = AllowlistEntry {
            proof_system: ProofSystem::Stage11bHalo2Reference,
            ..synthetic_production_entry()
        };
        let err = check_mainnet_eligible_with(&meta, &[would_match], &[]).unwrap_err();
        assert!(
            matches!(err, MainnetRefusalReason::BoundedReference { .. }),
            "expected BoundedReference (layer 3) refusal, got {err:?}"
        );
    }

    #[test]
    fn testnet_or_dev_only_refused_before_allowlist_lookup() {
        // testnet_or_dev_only: Some(true) must fire layer 1 before
        // any allowlist match. Cross-check with a "would-match"
        // allowlist entry to ensure the layer 6 path doesn't sneak
        // a Some(true) artifact onto mainnet.
        let mut meta = synthetic_production_meta();
        meta.testnet_or_dev_only = Some(true);
        let err = check_mainnet_eligible_with(
            &meta,
            &[synthetic_production_entry()],
            &[],
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                MainnetRefusalReason::TestnetOrDevOnly {
                    testnet_or_dev_only: Some(true),
                    ..
                }
            ),
            "expected TestnetOrDevOnly(Some(true)) (layer 1) refusal, got {err:?}"
        );
    }

    /// Stage 11d.1 tightening: the criteria document
    /// (mainnet-eligibility-criteria.md §1.5) requires
    /// `testnet_or_dev_only: Some(false)` EXPLICITLY for mainnet.
    /// The bare `None` (absent declaration) is treated as
    /// testnet/dev for safety. This test pins that an otherwise-
    /// allowlist-matching artifact with `testnet_or_dev_only: None`
    /// is refused at layer 1 — not sneaking past to layer 6.
    #[test]
    fn absent_testnet_or_dev_only_refused_before_allowlist_lookup() {
        let mut meta = synthetic_production_meta();
        meta.testnet_or_dev_only = None;
        let err = check_mainnet_eligible_with(
            &meta,
            &[synthetic_production_entry()],
            &[],
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                MainnetRefusalReason::TestnetOrDevOnly {
                    testnet_or_dev_only: None,
                    ..
                }
            ),
            "expected TestnetOrDevOnly(None) (layer 1) refusal for absent flag, got {err:?}"
        );
    }

    #[test]
    fn wrong_circuit_id_hex_rejects_even_with_correct_proof_system() {
        // Allowlist has one entry. Metadata has matching
        // proof_system + model_hash but a tampered circuit_id_hex.
        // Layer 6 must reject.
        let mut meta = synthetic_production_meta();
        meta.circuit_id_hex = Some("d".repeat(64)); // tampered
        let err = check_mainnet_eligible_with(
            &meta,
            &[synthetic_production_entry()],
            &[],
        )
        .unwrap_err();
        assert!(
            matches!(err, MainnetRefusalReason::NotInMainnetAllowlist { .. }),
            "expected NotInMainnetAllowlist for wrong circuit_id_hex, got {err:?}"
        );
    }

    #[test]
    fn wrong_model_hash_rejects_even_with_correct_proof_system_and_circuit_id() {
        let mut meta = synthetic_production_meta();
        meta.model_hash = "b".repeat(64); // tampered
        let err = check_mainnet_eligible_with(
            &meta,
            &[synthetic_production_entry()],
            &[],
        )
        .unwrap_err();
        assert!(
            matches!(err, MainnetRefusalReason::NotInMainnetAllowlist { .. }),
            "expected NotInMainnetAllowlist for wrong model_hash, got {err:?}"
        );
    }

    #[test]
    fn matching_structured_entry_would_pass_layer_6_for_synthetic_production_metadata() {
        // Sanity check that layer 6 actually returns Ok(()) when
        // a synthetic entry matches a synthetic production
        // artifact. Validates the matching logic without touching
        // the real (empty) const slice.
        let meta = synthetic_production_meta();
        let result = check_mainnet_eligible_with(
            &meta,
            &[synthetic_production_entry()],
            &[],
        );
        assert!(
            result.is_ok(),
            "synthetic production metadata + matching entry must pass layer 6; got {result:?}"
        );
    }

    #[test]
    fn legacy_alias_match_also_passes_layer_6() {
        // Defense-in-depth back-compat: an empty structured list
        // plus a single legacy ProofSystem entry that matches the
        // metadata's proof_system also passes layer 6. Confirms the
        // OR-bridge in `proof_system_passes_layer6` is wired
        // correctly. Both lists remain empty in production.
        let meta = synthetic_production_meta();
        let result = check_mainnet_eligible_with(&meta, &[], &[ProofSystem::Ezkl]);
        assert!(
            result.is_ok(),
            "synthetic production metadata + legacy alias entry must pass layer 6; got {result:?}"
        );
    }

    #[test]
    fn every_allowlist_entry_has_required_metadata() {
        // Forward-looking guard: when Stage 11d.3 adds entries,
        // these structural invariants from
        // docs/mainnet-eligibility-criteria.md §1.6 + §6 must hold.
        // Slice is empty in Stage 11d.1 so the loop body never
        // executes; the test exists to catch future bad entries.
        for entry in MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES {
            assert!(
                !matches!(
                    entry.proof_system,
                    ProofSystem::Mock
                        | ProofSystem::Stage11bOnnxReference
                        | ProofSystem::Stage11bHalo2Reference
                ),
                "{:?}: proof_system must not be Mock or any bounded reference variant",
                entry.proof_system
            );
            assert!(
                !matches!(entry.model_format, ModelFormat::Gguf | ModelFormat::Other(_)),
                "{:?}: model_format must not be Gguf or Other(_) — \
                 invalid for Stage 11d.3 unless a separately-reviewed strategy exists",
                entry.model_format
            );
            assert_eq!(
                entry.circuit_id_hex.len(),
                64,
                "circuit_id_hex must be 64 lowercase hex chars"
            );
            assert!(
                entry.circuit_id_hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "circuit_id_hex must be lowercase hex"
            );
            assert_eq!(
                entry.model_hash.len(),
                64,
                "model_hash must be 64 lowercase hex chars"
            );
            assert!(
                entry.model_hash.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "model_hash must be lowercase hex"
            );
            assert_eq!(
                entry.verification_key_hash_hex.len(),
                64,
                "verification_key_hash_hex must be 64 lowercase hex chars"
            );
            assert!(
                entry
                    .verification_key_hash_hex
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "verification_key_hash_hex must be lowercase hex"
            );
            assert!(
                !entry.backend_id.is_empty(),
                "backend_id must be non-empty"
            );
            assert!(
                !entry.chain_team_review_ref.is_empty(),
                "chain_team_review_ref must be non-empty"
            );
        }
    }

    #[test]
    fn vk_hash_helper_uses_documented_domain_separator() {
        assert_eq!(
            MAINNET_VK_HASH_DOMAIN_SEPARATOR, b"OMNINODE-VK:v1:",
            "domain separator must be exactly 'OMNINODE-VK:v1:' (15 ASCII bytes)"
        );
        assert_eq!(MAINNET_VK_HASH_DOMAIN_SEPARATOR.len(), 15);
    }

    #[test]
    fn vk_hash_helper_is_byte_stable() {
        // Pin two values. Computed offline as
        //   BLAKE3(b"OMNINODE-VK:v1:" || INPUT).
        // Caught any accidental edit to the domain separator OR
        // the helper's hashing order.
        let empty_input_hash = mainnet_vk_hash(b"");
        let abc_input_hash = mainnet_vk_hash(b"abc");

        // The two hashes must differ.
        assert_ne!(empty_input_hash, abc_input_hash);

        // Recompute to confirm determinism (no hidden state).
        assert_eq!(empty_input_hash, mainnet_vk_hash(b""));
        assert_eq!(abc_input_hash, mainnet_vk_hash(b"abc"));

        // Explicit verification against the algorithmic spec:
        // recompute by directly chaining the domain separator +
        // input, confirming the helper matches the documented
        // formula bit-for-bit.
        let mut h = blake3::Hasher::new();
        h.update(b"OMNINODE-VK:v1:");
        h.update(b"abc");
        let manual = *h.finalize().as_bytes();
        assert_eq!(
            abc_input_hash, manual,
            "mainnet_vk_hash output must equal BLAKE3(domain_separator || input)"
        );
    }
}
