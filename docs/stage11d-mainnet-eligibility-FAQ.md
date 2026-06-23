# Stage 11d Mainnet Eligibility FAQ

**Operator-facing Q&A introduced by Stage 11d.0.** Cross-references the [criteria document](mainnet-eligibility-criteria.md) and the [review-packet template](stage11d-review-packet.md).

**Wording guardrail**: Stage 11d **defines and prepares** the eligibility path. A later Stage 11d.3 entry may add the first eligible proof system after written sign-off. Until then, every artifact reaching `operator verify-proof` will report `mainnet_eligible=false`.

---

### Q1: Is anything mainnet-eligible after Stage 11d.0?

**No.** Stage 11d.0 is docs-only. It defines what eligibility means and prepares the review-packet template. `MAINNET_APPROVED_PROOF_SYSTEMS` (and its Stage 11d.1 structured replacement `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`) remain empty. Every proof artifact `operator verify-proof` sees today reports `mainnet_eligible=false`.

### Q2: Why isn't `Stage11bHalo2Reference` registered for mainnet eligibility now that Stage 11c made it sound?

**Two separate reasons.**

1. The bounded reference circuit certifies an architecture-validation fixture (a 4→8→4 toy MLP with frozen weights). It does not certify any real OmniNode inference workload. Soundness over a tiny toy model is not the same property as soundness over a production model.
2. Even if a chain team wanted to register it for some narrow use, the Stage 11d.0 criteria say bounded reference circuits MUST stay distinguishable from production proof systems. The hard rule H1/H2/H3 in [criteria §1.6](mainnet-eligibility-criteria.md#16-distinguishability-from-bounded-reference) requires a different `ProofSystem` variant + different `circuit_id_hex` + different `model_hash` for any production candidate. The Stage 11d.1 CI gate `stage11b_halo2_reference_never_in_eligibility_registry` enforces this.

`Stage11bHalo2Reference` stays testnet/dev-only **in perpetuity**.

### Q3: When does the first mainnet-eligible proof system land?

In **Stage 11d.3**, and only when:

- A specific production proof class has been selected by the SUM Chain review board (during Stage 11d.0/11d.1 review).
- Stage 11d.2 has shipped the implementation (verifier, fixtures, regen tool, opt-in cargo feature).
- The Stage 11d.3 review packet (filled-in copy of [`stage11d-review-packet.md`](stage11d-review-packet.md)) has been signed off by:
  - the proposing engineer,
  - an external cryptographer (criteria Claim 1.1.S2),
  - one non-author OmniNode engineer (review-packet section R2),
  - an independent reproducer (R4),
  - an operator team representative (R6),
  - the SUM Chain review board (R8).

If any sign-off is missing, the 11d.3 PR does not open.

### Q4: What does an operator do when `verify-proof` reports `mainnet_eligible=false`?

Nothing automatic. The boolean is a reported result, not a workflow trigger. The accompanying refusal reason (one of the six `MainnetRefusalReason` variants) tells the operator *which* gate fired. The most common cases:

| Refusal | Operator action |
|---|---|
| `TestnetOrDevOnly` | The producer of the artifact explicitly disclaimed mainnet. Use a mainnet-approved producer (none ship through Stage 11c). |
| `MockBackend` | A `mock-v1` artifact reached a mainnet code path. Switch to a real backend (none ship through Stage 11c). |
| `BoundedReference` | A `Stage11bOnnxReference` or `Stage11bHalo2Reference` artifact reached a mainnet code path. Bounded references are testnet/dev-only by design. |
| `GgufClaim` | A GGUF-format artifact reached a mainnet code path. No GGUF proof strategy exists through Stage 11d. |
| `UnknownModelFormat` | The artifact's `model_format` is `Other(_)` or absent on a non-mock backend. Promote the format to a first-class variant via a chain-team-reviewed PR. |
| `NotInMainnetAllowlist` | The artifact's `(proof_system, circuit_id_hex, model_hash)` triple is not in `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`. Through Stage 11d.0/.1/.2, this list is empty by design — no proof system is approved yet. |

### Q5: How does an external prover get a proof system reviewed?

Today, the path is:

1. Read the [criteria document](mainnet-eligibility-criteria.md).
2. Implement the proof system (in OmniNode or as an external project) satisfying every numbered criterion.
3. Fill in the [review packet template](stage11d-review-packet.md) — including the external cryptographer sign-off field.
4. Submit the filled packet to the SUM Chain review board for evaluation.
5. The review board reviews per criteria §2 (R1–R8).
6. On approval, a Stage 11d.3 PR proposes the new `AllowlistEntry`; the PR cites the signed review document path in the entry's `chain_team_review_ref` field.

The SUM Chain review board roster is a placeholder at Stage 11d.0 (intentional — the board's composition is not yet recorded in this repo). External proposers should contact the SUM Chain core team via existing channels for the current roster.

### Q6: Why isn't GGUF / LLM inference a candidate?

No published proof strategy for transformer-class LLM inference with bounded soundness exists at the time of Stage 11d.0. Without a concrete bounded invariant (e.g., "this circuit certifies the next-token logit computation for a sub-model of depth N with attention head width W under fixed weights"), there's nothing to register. GGUF artifacts continue to be refused by `check_mainnet_eligible` layer 4 (`GgufClaim`) regardless of any other metadata.

If a published, peer-reviewed proof strategy for GGUF/LLM inference becomes available, it would be a separate chain-team plan — not a Stage 11d.x substage.

### Q7: Does Stage 11d ship any chain-side verifier?

**No.** Stage 11d substages 0 through 3 are entirely off-chain. `InferenceAttestationDigest` (Stage 7b) is unchanged. No new SUM Chain RPC ships. No chain-side mainnet eligibility registry mirror ships.

A future on-chain verifier (Stage 12+) is a separate chain-team plan. The Stage 11d schema is shaped so when that stage lands it's a refactor rather than a schema change — the on-chain verifier would consume the same `(proof_system, circuit_id_hex, model_hash)` triple Stage 11d.1 defines.

### Q8: What if Stage 11d.2 implementation surfaces a soundness gap?

**Halt and report.** The Stage 11c [halt-and-report](mainnet-eligibility-criteria.md#7-open-fields-to-be-filled-before-downstream-stages-may-proceed) posture carries forward. If the proposing engineer discovers that the production proof class requires changing `canonical_spec.json`, `EXPECTED_SPEC_HASH`, the canonical evaluator, framework manifests, or any other spec/source-of-truth surface to be valid: stop, append to the review packet's §10 halt-and-report log, and surface to the SUM Chain review board. Do not silently patch.

### Q9: How is an eligibility registry entry removed if a soundness regression is later discovered?

A PR that removes the entry from `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` and bumps the `mainnet_allowlist_snapshot.json` byte-stability fixture (filename grandfathered). The same CODEOWNERS-style review controls (planned for downstream stages — Stage 11d.0 does not introduce CODEOWNERS) apply.

Operator communication path for a post-merge soundness regression is recorded in the per-entry review packet's §8.3 (Reversibility).

### Q10: Where do I read the canonical refusal logic?

`crates/omni-zkml/src/proof.rs` — function `check_mainnet_eligible`. Six layers, evaluated in order; the first match wins. Stage 11d.1 rewires layer 6 to consult `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`; layers 1–5 are unchanged from Stage 11b.0.

### Q11: Is there a way for me as an operator to opt into running the bounded-reference verifier on mainnet?

**No, and there should not be.** The bounded-reference verifier is reachable today only under the opt-in `halo2-reference-verify` feature, and even when reachable it does not change the artifact's `mainnet_eligible` status. An artifact carrying `proof_system: Some(Stage11bHalo2Reference)` is refused on mainnet by layer 3 regardless of whether the verifier is registered. The operator's only effective action is to verify on testnet/dev.

### Q12: What is the VK hash scheme?

`BLAKE3(MAINNET_VK_HASH_DOMAIN_SEPARATOR || canonical_vk_bytes)` where `MAINNET_VK_HASH_DOMAIN_SEPARATOR = b"OMNINODE-VK:v1:"` (15 ASCII bytes, no null terminator, no length prefix). The trailing `v1` allows a future migration to a new scheme without ambiguity over which hash an eligibility registry entry's `verification_key_hash_hex` was computed under. `canonical_vk_bytes` is the per-verifier canonical serialization of its `VerifyingKey`; each production verifier must document its scheme. See [`docs/mainnet-eligibility-criteria.md` §1.7](mainnet-eligibility-criteria.md) for the full spec. Stage 11d.1 ships the helper `omni_zkml::mainnet_vk_hash(&[u8]) -> [u8; 32]` and the `MAINNET_VK_HASH_DOMAIN_SEPARATOR` constant; cross-validation between an artifact's `metadata.verification_key_hex` and the eligibility registry's `verification_key_hash_hex` lands in Stage 11d.2 with the first production verifier.

### Q13: Can `Stage11bHalo2Reference` or any other bounded reference ever be registered for mainnet eligibility?

**No.** Hard rule H1 from the criteria document § 1.6: bounded reference proof systems are testnet/dev-only in perpetuity. Three independent guards enforce this:

1. Layer 3 of `check_mainnet_eligible` (`BoundedReference`) refuses the artifact **before** the layer-6 eligibility registry lookup. Pinned by `bounded_reference_refused_before_eligibility_registry_lookup`.
2. The `stage11b_halo2_reference_never_in_eligibility_registry` test asserts no entry in `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` (or the legacy `MAINNET_APPROVED_PROOF_SYSTEMS`) has `proof_system ∈ {Stage11bOnnxReference, Stage11bHalo2Reference}`.
3. The `every_allowlist_entry_has_required_metadata` test additionally rejects bounded-reference `proof_system` values in any entry (test name grandfathered per Stage 11d.3B).

### Q14: What happens if a candidate proof class's fixture isn't byte-deterministic?

**It is not eligible.** Criteria §1.8 makes this a hard requirement. The Stage 11d.2 implementation must include a developer-host regen tool with a `verify-only` mode that fails on drift, and two consecutive `regen` runs must produce byte-identical fixture files (`params.bin`, `proof.bin`, `proof_artifact.json`). If the prover requires true OS randomness, the proof system is not eligible until either (a) all randomness is seeded via a pinned RNG (Stage 11b.1.b precedent: `rand_chacha::ChaCha20Rng::from_seed(PINNED_SEED)`), or (b) an alternative proof class is proposed that admits deterministic byte-stable proving.

### Q15: What's the difference between artifact `verification_key_hex` and the eligibility registry's `verification_key_hash_hex`?

The artifact-side field (`ProofMetadata::verification_key_hex`) is the **existing** Stage 11b.0 schema field — its name is historical and ambiguous (it might be a raw VK encoding or already a hash, depending on backend). Stage 11d.1 does NOT rename this field; that's a wider schema migration deferred to a future stage.

The eligibility-registry-side field (`AllowlistEntry::verification_key_hash_hex`, introduced in Stage 11d.1) is unambiguous: it is the hex of `mainnet_vk_hash(canonical_vk_bytes)` per Q12 above. Per-verifier code at Stage 11d.2+ time cross-validates the two by hashing the artifact's `verification_key_hex` (or whatever raw bytes the verifier extracts from it) under the §1.7 scheme and comparing to the registered hex. Stage 11d.1's layer 6 does NOT perform this cross-check — only the structured `(proof_system, circuit_id_hex, model_hash)` triple match.
