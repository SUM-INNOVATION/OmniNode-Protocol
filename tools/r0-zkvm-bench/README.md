# r0-zkvm-bench — OmniNode #101 R0 host-side crate (B0 wire adoption)

> ## Disposition: NON-SELECTION research crate — NOT a zkVM harness
>
> This crate **adopts** the published, frozen **B0** candidate-neutral wire types
> from the SUM Chain team's `sumchain-wire 0.2.1` crates.io release and exercises
> the host-side, crypto-free groundwork around them. It **proves nothing**,
> integrates **no** zkVM SDK / toolchain / container / build-script, and contains
> no guest, guest compilation, candidate proving, Groth16 wrapping, measurement,
> or real receipt verification. Every artifact it can build is stamped
> `NON_SELECTION / INVALID_FOR_B0`.

## What is adopted from `sumchain_wire::b0`

The bespoke OmniNode wire types and their `OMNINODE.R0.*` identity domains are
**deleted** and replaced by the frozen B0 family (leading `SUMCHAIN/R0/*` tags):

| B0 type | Bytes | Used for |
|---|---|---|
| `object_commitment::ObjectCommitmentV1` | 80 | every object commitment (checked ctors, private fields) |
| `manifest::{OutputManifestV1, InputManifestV1}` + slot descriptors | 38 + 85·n | per-slot commitments (slot-kind ↔ object-kind enforced) |
| `derived_input::DerivedInputV1` | 350 | derived-input commitment |
| `statement::R0ComputationStatementV2` | 996 | the candidate-neutral statement (built only as a zero-spec-hash **template**) |
| `partial_proof::PartialComputeProofV1` | 137 | candidate-**neutral** proof identity (4 hashes) |
| `proof_envelope::ProductionProofEnvelopeV1` | 235 | candidate-**specific** identity (candidate / program id / verifier material) |
| `allowlist::GuestProgramAllowlistV1` | var | guest-set membership binding |
| `verifier_material::VerifierMaterialManifestV1` | var | verifier-material identity |
| `merkle::{merkle_root, chunk_count_checked, CHUNK}` | — | SNIP Merkle root + chunk-count rules |
| `enums::{ObjectKind, SlotKind, InputSlotKind, Candidate, UnitKind, Arch, …}` | — | frozen enums; `ObjectKind::{Tokenizer=2, Slot=8}` reserved + rejected |
| `workload::{residual_state_bytes, kv_state_bytes, token_seq_bytes}` | — | frozen raw state/token byte encoders |

## What this crate adds around them (host-side only)

- `merkle` — retains an in-memory tree that yields **index-aware inclusion
  proofs** and a length-unambiguous **chunk-witness** verifier (B0 supplies the
  root/count rules; a parity test pins `MerkleTree::root() == merkle_root(data)`).
- `model_auth` — Approach-2 weight-chunk authentication against the public model
  commitment's B0 accessors.
- `workload` — the bounded integer decoder-only transformer layer-group +
  SelectToken reference executor, emitting the 996-byte statement as a
  `SyntheticJournal` template.
- `verifier` — allowlist / identity / journal binding **logic** over an
  explicitly synthetic `CannedReceipt` (NOT proof verification).
- `fixture` — clearly-synthetic, non-selection B0 identities and binding sets.
- `bench` — the reproducible benchmark artifact schema, always non-selection.

## What it emits, and how non-selection is forced

The crate emits exactly one artifact: a JSON research record
(`bench::BenchArtifact`). That record carries **no measured samples**, is **not a
selection-valid artifact**, and contains **no proof / finalization /
protocol-hash data**. Non-selection is forced structurally, not by convention:

- `classification::RunClass` has a **single** non-selection variant — there is no
  `Final`/`Official`/`Selection`/`Eligible` variant or code path.
- The statement is only ever a **zero-spec-hash template** (via
  `statement::SyntheticJournal`). No public path through the crate reaches B0's
  `materialize_final`, a raw `Writer`, or the full B0 statement module: the `b0`
  namespace is re-exported **crate-privately** and only focused safe types are
  public (enforced by `tests/api_surface.rs`).
- Every invariant-bearing field of `BenchArtifact` / `CandidateArchResult` /
  `CostBreakdown` / `PhaseCost` is **private** with read-only accessors. The only
  constructor (`BenchArtifact::research`) forces the status, the zero template
  spec hash, the exact ABSENT toolchain marker, and unmeasured phases; `PhaseCost`
  has exactly one constructible state (`NOT_MEASURED_TOOLCHAIN_ABSENT`, empty
  samples). A hardened validated `Deserialize` rejects any tampered JSON (altered/
  omitted status, nonzero spec hash, non-empty samples, forged measured state,
  non-ABSENT digest, inconsistent candidate/hex data, or unknown fields).
- `verifier::CannedReceipt` is an explicitly synthetic stand-in — never proof
  verification.

## Boundary

Intentionally NOT a member of the OmniNode workspace (the root `Cargo.toml` sets
`exclude = ["tools"]`). No production crate depends on it; isolation is enforced
by `tests/dependency_isolation.rs` and `scripts/check_no_prod_dep.sh`. The single
external dependency is the crates.io `sumchain-wire` at the exact pin `=0.2.1`
(no path/git source), plus `blake3` / `serde` / `serde_json`.

## Build & test (offline, plain stable Rust)

```console
$ cd tools/r0-zkvm-bench
$ cargo build
$ cargo test
$ cargo fmt -- --check
$ cargo clippy --all-targets -- -D warnings
$ ./scripts/check_no_prod_dep.sh
```

## Environment (why there is no proving)

No zkVM SDK / toolchain / container is present or integrated, and none is
declared. There are deliberately **no** `sp1`/`risc0` feature gates and **no**
guest scaffolding — nothing here pretends to be a working prover. Real proving is
a separate, later, explicitly-authorized effort; this crate is host-side wire
adoption only.
