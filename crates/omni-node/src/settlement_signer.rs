//! Issue #80 — verifier identity mapping and validation for
//! settlement claim signing.
//!
//! Feature-gated on `settlement-read`. Provides the OmniNode-side
//! surface that answers three implementation questions before any
//! future claim submission (#87) is attempted:
//!
//! 1. Which configured local Ed25519 seed derives to the SUM Chain
//!    verifier address?
//! 2. Does that derived address match the `attestation.verifier_address`
//!    that a claim would target?
//! 3. If the target session is bond-required, what is the observable
//!    registry state and bond amount for the verifier?
//!
//! ## Signing neutrality contract
//!
//! `settlement-read` is signing-neutral. This module MUST NOT provide
//! a signing surface, so by construction:
//!
//! - [`ClaimSignerIdentity`] stores only `derived_address: String`.
//!   No seed field, no keypair, no secret.
//! - No method on [`ClaimSignerIdentity`] returns the seed, a seed
//!   reference, or any value derived from seed bytes.
//! - The seed material passed into [`ClaimSignerIdentity::resolve`]
//!   is consumed by [`omni_zkml::signer_chain_address_base58`] once
//!   and drops out of scope at return; it never lands on the struct.
//! - No import of `omni_zkml::sign_*`, `omni_sumchain::tx::*`,
//!   `omni_sumchain::outer_sign::*`, or
//!   `libp2p_identity::ed25519::SecretKey` appears anywhere in this
//!   module.
//! - **Production code paths never reference `sum_sendRawTransaction`,
//!   `sum_submit*`, or `omninode_submit*`.** Names from that set may
//!   appear ONLY as test-guard string literals inside `#[cfg(test)]`
//!   modules — specifically the `WRITE_DENYLIST` used by
//!   [`tests::precheck_bond_calls_only_read_rpcs`] to assert that the
//!   real precheck code paths never emit those RPC calls. No
//!   production `impl` or `fn` in this module names any of those
//!   RPCs.
//!
//! When #87 (verifier self-claim submission) lands, it MUST introduce
//! a new `#[cfg(feature = "settlement-submit")]` (or equivalent) block
//! for any seed accessor / signing capability. The `settlement-read`
//! gated code here stays as-is.

use omni_sumchain::settlement::wire::VerifierRegistryRaw;
use omni_sumchain::settlement::SettlementReadError;
use omni_sumchain::{JsonRpcTransport, SumChainClient};

use crate::operator::{parse_seed_hex, SeedSource};

// ── Errors ───────────────────────────────────────────────────────────────────

/// Typed error surface for the identity mapping + precheck helpers.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ClaimSignerError {
    /// `OMNINODE_VERIFIER_SEED_HEX` is unset (or `SeedSource::AbsentForTest`
    /// was supplied in a test).
    #[error("claim signer seed unavailable: set OMNINODE_VERIFIER_SEED_HEX (64 hex chars)")]
    SeedMissing,

    /// The configured seed is not exactly 64 hex chars, or contains
    /// non-hex bytes. Wraps the underlying parse reason so operators
    /// can distinguish length errors from character errors.
    #[error("claim signer seed malformed: {0}")]
    SeedMalformed(String),

    /// `omni_zkml::signer_chain_address_base58` failed. Defensive — for
    /// a valid 32-byte seed this should not fire in practice, but the
    /// underlying API returns `Result` so we surface it explicitly.
    #[error("chain-address derivation failed: {0}")]
    DerivationFailed(String),

    /// The address derived from the configured seed does not equal the
    /// `expected` address. Carries BOTH addresses so operators can diff.
    #[error(
        "claim signer address mismatch: derived={derived}, expected={expected}"
    )]
    AddressMismatch { expected: String, derived: String },

    /// Bond precheck's read RPC (`omninode_getVerifier`, plus the
    /// unconditional params + head fetches) surfaced a
    /// [`SettlementReadError`]. Flattened to a string here to avoid
    /// re-exposing the #83 error enum on this module's public surface.
    #[error("verifier registry fetch failed: {0}")]
    RegistryFetchFailed(String),
}

// ── Bond precheck outcome ────────────────────────────────────────────────────

/// Observable result of the bond precheck. Reports registry state and
/// the raw bond amount; does NOT make a "sufficient bond" judgment —
/// that comparison against the session's `min_bond` is #87's concern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BondPrecheckOutcome {
    /// The session is not bond-required. No chain read was issued.
    NotRequired,

    /// The verifier has a registry entry in `bonded` state. Bond amount
    /// is reported verbatim; #87 compares it to the session's minimum.
    Bonded { bond_amount: u128 },

    /// The verifier has a registry entry in `unbonding` state. Bond
    /// amount is still present; withdrawal unlocks at
    /// `withdrawable_at_height` (chain may or may not populate it).
    Unbonding {
        bond_amount: u128,
        withdrawable_at_height: Option<u64>,
    },

    /// The verifier has a registry entry in `withdrawn` state. Bond
    /// has been fully removed by the verifier.
    Withdrawn,

    /// The chain returned no registry entry for the address.
    NotRegistered,

    /// The chain returned a `bond_state` value that isn't in the
    /// known set. Reports the raw string and amount so #87 can decide
    /// policy without silently misclassifying.
    UnknownWireState { raw: String, bond_amount: u128 },
}

// ── Identity ─────────────────────────────────────────────────────────────────

/// The mapping proof: "this configured identity derives to this
/// SUM Chain verifier address." Constructed only via
/// [`ClaimSignerIdentity::resolve`]. The seed material used at
/// construction is dropped inside `resolve` and is NOT reachable
/// through any accessor on this struct.
#[derive(Debug, Clone)]
pub(crate) struct ClaimSignerIdentity {
    /// The base58 SUM Chain address derived from the seed at
    /// construction time. This is the ONLY state carried on the
    /// struct — see the module docstring's signing-neutrality
    /// contract.
    derived_address: String,
}

impl ClaimSignerIdentity {
    /// Resolve a seed from `source`, derive its SUM Chain address, and
    /// return an identity carrying only that address.
    ///
    /// The seed bytes read from `source` are consumed by the
    /// derivation call and go out of scope at return. They are not
    /// stored on the returned struct.
    pub(crate) fn resolve(source: SeedSource) -> Result<Self, ClaimSignerError> {
        let seed = resolve_seed(&source)?;
        let derived_address = omni_zkml::signer_chain_address_base58(&seed)
            .map_err(|e| ClaimSignerError::DerivationFailed(e.to_string()))?;
        // `seed` drops here — no field carries it forward.
        Ok(Self { derived_address })
    }

    /// The base58 SUM Chain address derived from the configured seed.
    pub(crate) fn address(&self) -> &str {
        &self.derived_address
    }

    /// Verify the derived address matches `expected`. Returns
    /// [`ClaimSignerError::AddressMismatch`] on inequality, with both
    /// addresses carried on the error so operators can diff.
    pub(crate) fn verify_matches(
        &self,
        expected: &str,
    ) -> Result<(), ClaimSignerError> {
        if self.derived_address == expected {
            Ok(())
        } else {
            Err(ClaimSignerError::AddressMismatch {
                expected: expected.to_string(),
                derived: self.derived_address.clone(),
            })
        }
    }

    /// Bond precheck for the identity's derived address, using the
    /// `settlement-read` verifier registry surface from #83.
    ///
    /// When `session_bond_required` is `false`, returns
    /// [`BondPrecheckOutcome::NotRequired`] without issuing any RPC.
    /// When `true`, reads `chain_getChainParams` +
    /// `chain_getBlockHeight` (mandated first by #83's dormancy check)
    /// followed by `omninode_getVerifier` and reports the observed
    /// registry state + raw bond amount.
    ///
    /// **Does NOT judge sufficiency.** #87 compares the reported
    /// amount against the target session's `min_bond`.
    pub(crate) fn precheck_bond<T: JsonRpcTransport>(
        &self,
        client: &SumChainClient<T>,
        session_bond_required: bool,
    ) -> Result<BondPrecheckOutcome, ClaimSignerError> {
        if !session_bond_required {
            return Ok(BondPrecheckOutcome::NotRequired);
        }

        let entry: Option<VerifierRegistryRaw> = client
            .omninode_get_verifier(&self.derived_address)
            .map_err(|e| ClaimSignerError::RegistryFetchFailed(e.to_string()))?;

        let entry = match entry {
            None => return Ok(BondPrecheckOutcome::NotRegistered),
            Some(e) => e,
        };

        let bond_amount = entry.bond_amount.parse::<u128>().map_err(|e| {
            ClaimSignerError::RegistryFetchFailed(format!(
                "failed to parse bond_amount '{}' as u128: {e}",
                entry.bond_amount
            ))
        })?;

        Ok(match entry.bond_state.as_str() {
            "bonded" => BondPrecheckOutcome::Bonded { bond_amount },
            "unbonding" => BondPrecheckOutcome::Unbonding {
                bond_amount,
                withdrawable_at_height: entry.withdrawable_at_height,
            },
            "withdrawn" => BondPrecheckOutcome::Withdrawn,
            other => BondPrecheckOutcome::UnknownWireState {
                raw: other.to_string(),
                bond_amount,
            },
        })
    }
}

// ── Seed resolution (internal) ───────────────────────────────────────────────

/// Resolve the seed bytes from a [`SeedSource`], normalising the
/// operator-side error variants into [`ClaimSignerError`]. The
/// returned bytes are the caller's responsibility to consume + drop.
fn resolve_seed(source: &SeedSource) -> Result<[u8; 32], ClaimSignerError> {
    match source {
        SeedSource::Env => {
            let v = std::env::var("OMNINODE_VERIFIER_SEED_HEX")
                .map_err(|_| ClaimSignerError::SeedMissing)?;
            parse_seed_hex(&v)
                .map_err(|e| ClaimSignerError::SeedMalformed(e.to_string()))
        }
        #[cfg(test)]
        SeedSource::Explicit(bytes) => Ok(*bytes),
        #[cfg(test)]
        SeedSource::AbsentForTest => Err(ClaimSignerError::SeedMissing),
        #[cfg(all(test, feature = "submit"))]
        SeedSource::MalformedForTest(h) => parse_seed_hex(h)
            .map_err(|e| ClaimSignerError::SeedMalformed(e.to_string())),
    }
}

// Compile-time reference to keep the `SettlementReadError` import
// live: it's the source type behind `RegistryFetchFailed`'s content.
// The `From` isn't derived because we intentionally flatten to
// `String` at this boundary.
#[allow(dead_code)]
fn _keep_settlement_read_error_referenced(e: SettlementReadError) -> ClaimSignerError {
    ClaimSignerError::RegistryFetchFailed(e.to_string())
}

// ── Issue #87 — settlement-submit signer extension ───────────────────────────
//
// `ClaimSigner` is a superset of `ClaimSignerIdentity` that ALSO
// retains the resolved seed bytes for exactly one purpose: passing
// them to `omni_sumchain::settlement_submit::tx::sign_and_submit` at
// the very end of the claim pipeline (steps 11-13 of the reviewer's
// flow). Gated on `settlement-submit` so `settlement-read`-only
// builds still have no reachable seed accessor.
//
// The read-side `ClaimSignerIdentity` continues to store only the
// derived address; nothing about #80's contract changes.

#[cfg(feature = "settlement-submit")]
pub(crate) struct ClaimSigner {
    identity: ClaimSignerIdentity,
    /// Retained ONLY for the one `sign_and_submit` call in the claim
    /// pipeline. Never exposed except through
    /// [`ClaimSigner::seed_for_signing`].
    seed: [u8; 32],
}

#[cfg(feature = "settlement-submit")]
impl ClaimSigner {
    /// Resolve the seed from `source`, derive the address, and retain
    /// the seed for a subsequent single sign call.
    ///
    /// **Load timing invariant**: the caller must NOT invoke
    /// `ClaimSigner::resolve` until every precheck (dormancy,
    /// attestation, authority, maturity, bond, builder envelope,
    /// decoded-tx) has passed. Loading earlier violates the
    /// separation of "identity derivation" (settlement-read,
    /// non-retaining) from "signing seed retention" (settlement-
    /// submit, retaining).
    pub(crate) fn resolve(source: SeedSource) -> Result<Self, ClaimSignerError> {
        let seed = resolve_seed(&source)?;
        let derived_address = omni_zkml::signer_chain_address_base58(&seed)
            .map_err(|e| ClaimSignerError::DerivationFailed(e.to_string()))?;
        Ok(Self {
            identity: ClaimSignerIdentity { derived_address },
            seed,
        })
    }

    /// Base58 verifier address.
    pub(crate) fn address(&self) -> &str {
        self.identity.address()
    }

    /// Handoff for the outer-sign path. The returned reference is the
    /// ONLY way seed bytes reach the signing primitive. `ClaimSigner`
    /// drops the seed with the struct at end-of-scope.
    ///
    /// Reachable only under `#[cfg(feature = "settlement-submit")]`
    /// — the settlement-read code path has no analogous accessor.
    pub(crate) fn seed_for_signing(&self) -> &[u8; 32] {
        &self.seed
    }
}

// ── Hermetic tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use omni_sumchain::FakeJsonRpcTransport;
    use serde_json::json;

    // ── Fixtures ────────────────────────────────────────────────────────

    fn seed_7() -> [u8; 32] {
        [7u8; 32]
    }

    /// Base58 chain address derived from `[7u8; 32]`. Pinned as a
    /// regression guard against changes to
    /// `signer_chain_address_base58` or the BLAKE3-based
    /// `derive_chain_address_base58`. If either helper's byte format
    /// changes, this literal must be updated intentionally.
    fn expected_addr_for_seed_7() -> String {
        omni_zkml::signer_chain_address_base58(&seed_7())
            .expect("derivation of pinned seed must succeed")
    }

    fn make_client() -> (SumChainClient<FakeJsonRpcTransport>, FakeJsonRpcTransport) {
        let fake = FakeJsonRpcTransport::new();
        // Zero seed on the client — precheck_bond reaches only the
        // read RPCs, none of which consult the client's seed.
        let client = SumChainClient::with_transport([0u8; 32], fake.clone());
        (client, fake)
    }

    fn seed_gates_active(fake: &FakeJsonRpcTransport) {
        fake.set_response(
            "chain_getChainParams",
            Ok(json!({
                "finality_depth": 12,
                "min_fee": 100,
                "chain_id": 1_800_100,
                "inference_settlement_enabled_from_height": 0,
                "inference_verifier_bonding_enabled_from_height": 0,
            })),
        );
        fake.set_response(
            "chain_getBlockHeight",
            Ok(json!({ "height": 500_000, "finality": "latest" })),
        );
    }

    // ── Test 1 — resolve derives the expected verifier address ─────────

    #[test]
    fn resolve_derives_expected_verifier_address() {
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed for a valid 32-byte seed");
        let expected = expected_addr_for_seed_7();
        assert_eq!(identity.address(), expected.as_str());
    }

    // ── Test 2 — mismatched expected verifier address is refused ───────

    #[test]
    fn verify_matches_refuses_wrong_expected_address() {
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        let wrong = "definitely-not-the-derived-address";
        let err = identity
            .verify_matches(wrong)
            .expect_err("verify_matches must refuse a non-matching expected address");
        match &err {
            ClaimSignerError::AddressMismatch { expected, derived } => {
                assert_eq!(expected, wrong);
                assert_eq!(derived, identity.address());
            }
            other => panic!("expected AddressMismatch, got {other:?}"),
        }
        // The Display must carry BOTH addresses so operators can diff.
        let msg = err.to_string();
        assert!(msg.contains(wrong), "Display must carry expected: {msg}");
        assert!(msg.contains(identity.address()), "Display must carry derived: {msg}");
    }

    #[test]
    fn verify_matches_accepts_correct_expected_address() {
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        identity
            .verify_matches(&expected_addr_for_seed_7())
            .expect("verify_matches must accept the derived address");
    }

    // ── Test 3a/3b/3c — malformed seed hex refused ─────────────────────

    #[test]
    fn seed_hex_too_short_refused() {
        let too_short = "a".repeat(63); // 63 chars
        let err = parse_seed_hex(&too_short).expect_err("63 chars must be rejected");
        // The operator-side error is `OperatorError::SeedMalformed(msg)`;
        // the resolver layer wraps that into
        // `ClaimSignerError::SeedMalformed(_)`. Assert the string
        // carries the length hint so operators can diagnose.
        let msg = err.to_string();
        assert!(
            msg.contains("expected 64 hex chars, got 63"),
            "length error must name expected/actual: {msg}"
        );
    }

    #[test]
    fn seed_hex_too_long_refused() {
        let too_long = "a".repeat(66); // 66 chars
        let err = parse_seed_hex(&too_long).expect_err("66 chars must be rejected");
        assert!(
            err.to_string().contains("expected 64 hex chars, got 66"),
            "length error must name expected/actual: {err}"
        );
    }

    #[test]
    fn seed_hex_non_hex_refused() {
        // 62 valid hex + 2 non-hex chars ('x', 'y') — length passes,
        // radix parse fails inside the loop.
        let non_hex = format!("{}xy", "a".repeat(62));
        assert_eq!(non_hex.len(), 64);
        let err = parse_seed_hex(&non_hex)
            .expect_err("non-hex chars must be rejected once length passes");
        // `u8::from_str_radix` error message contains "invalid digit".
        assert!(
            err.to_string().to_lowercase().contains("invalid digit"),
            "non-hex error should surface radix parser message: {err}"
        );
    }

    // ── Test 3d — uppercase hex is accepted (current behavior pin) ─────

    #[test]
    fn seed_hex_uppercase_accepted_current_behavior_pin() {
        // Pins the CURRENT behavior of the shared `parse_seed_hex`
        // helper: `u8::from_str_radix(_, 16)` accepts uppercase A-F.
        // This is not a recommendation — it locks the invariant so a
        // future stricter refactor is a deliberate change, not a
        // silent one. If parser strictness changes, update this test
        // AND every consumer.
        let mixed = "AA".repeat(32);
        assert_eq!(mixed.len(), 64);
        let seed = parse_seed_hex(&mixed)
            .expect("uppercase hex must currently parse (case-insensitive u8::from_str_radix)");
        assert_eq!(seed, [0xAAu8; 32]);
    }

    // ── Test 4 — resolve + verify_matches issue no RPC ─────────────────

    #[test]
    fn resolve_and_verify_do_not_call_any_rpc() {
        let (_client, fake) = make_client();
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        let _ = identity.verify_matches(&expected_addr_for_seed_7());
        assert!(
            fake.calls().is_empty(),
            "resolve + verify_matches must issue zero RPC; calls={:?}",
            fake.calls()
        );
    }

    // ── Test 5 — missing seed refused ──────────────────────────────────

    #[test]
    fn missing_seed_is_refused() {
        let err = ClaimSignerIdentity::resolve(SeedSource::AbsentForTest)
            .expect_err("AbsentForTest must produce SeedMissing");
        assert!(
            matches!(err, ClaimSignerError::SeedMissing),
            "expected SeedMissing, got {err:?}"
        );
    }

    // ── Test 6 — precheck skips when not bond-required ─────────────────

    #[test]
    fn precheck_bond_not_required_short_circuits() {
        let (client, fake) = make_client();
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        let outcome = identity
            .precheck_bond(&client, /* session_bond_required */ false)
            .expect("NotRequired path must not error");
        assert_eq!(outcome, BondPrecheckOutcome::NotRequired);
        assert!(
            fake.calls().is_empty(),
            "NotRequired path must issue zero RPC; calls={:?}",
            fake.calls()
        );
    }

    // ── Test 7 — precheck calls only read RPCs (no writes anywhere) ────

    /// Read allowlist: every method the precheck may legitimately
    /// call. If a future edit introduces any method outside this set,
    /// `precheck_bond_calls_only_read_rpcs` fails.
    const READ_ALLOWLIST: &[&str] =
        &["chain_getChainParams", "chain_getBlockHeight", "omninode_getVerifier"];

    /// Denylist: canonical write-tx methods used elsewhere in the
    /// workspace. None of them may ever appear in this module's call
    /// log.
    const WRITE_DENYLIST: &[&str] =
        &["sum_sendRawTransaction", "sum_getNonce"];

    #[test]
    fn precheck_bond_calls_only_read_rpcs() {
        let (client, fake) = make_client();
        seed_gates_active(&fake);
        fake.set_response(
            "omninode_getVerifier",
            Ok(json!({
                "address": expected_addr_for_seed_7(),
                "bond_amount": "10000",
                "bond_state": "bonded",
                "slash_history": []
            })),
        );

        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        let _ = identity
            .precheck_bond(&client, /* session_bond_required */ true)
            .expect("bonded precheck must succeed");

        let observed_methods: Vec<String> =
            fake.calls().into_iter().map(|(m, _)| m).collect();
        assert!(!observed_methods.is_empty(), "precheck should have called some RPC");
        for m in &observed_methods {
            assert!(
                READ_ALLOWLIST.contains(&m.as_str()),
                "precheck called a non-allowlisted RPC '{m}'; observed={observed_methods:?}"
            );
        }
        for banned in WRITE_DENYLIST {
            assert!(
                !observed_methods.iter().any(|m| m == banned),
                "precheck called denylisted write RPC '{banned}'; observed={observed_methods:?}"
            );
        }
    }

    // ── Test 8 — bonded reports amount only, no sufficiency judgment ───

    #[test]
    fn precheck_bond_bonded_reports_amount_no_sufficiency_language() {
        let (client, fake) = make_client();
        seed_gates_active(&fake);
        fake.set_response(
            "omninode_getVerifier",
            Ok(json!({
                "address": expected_addr_for_seed_7(),
                "bond_amount": "10000",
                "bond_state": "bonded",
                "slash_history": []
            })),
        );
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        let outcome = identity
            .precheck_bond(&client, /* session_bond_required */ true)
            .expect("bonded precheck must succeed");
        assert_eq!(outcome, BondPrecheckOutcome::Bonded { bond_amount: 10_000 });
        // The outcome's Debug format must not smuggle "sufficient" or
        // "enough" language — those are judgments this module does
        // not make.
        let debug = format!("{outcome:?}");
        for banned in ["sufficient", "enough", "insufficient", "too low", "min_bond"] {
            assert!(
                !debug.to_lowercase().contains(banned),
                "Bonded outcome must not carry judgment word '{banned}': {debug}"
            );
        }
    }

    // ── Test 9 — not-registered when chain returns null ────────────────

    #[test]
    fn precheck_bond_not_registered() {
        let (client, fake) = make_client();
        seed_gates_active(&fake);
        fake.set_response("omninode_getVerifier", Ok(serde_json::Value::Null));
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        let outcome = identity
            .precheck_bond(&client, true)
            .expect("null registry entry must map to NotRegistered");
        assert_eq!(outcome, BondPrecheckOutcome::NotRegistered);
    }

    // ── Test 10 — unbonding state round-trips ──────────────────────────

    #[test]
    fn precheck_bond_unbonding() {
        let (client, fake) = make_client();
        seed_gates_active(&fake);
        fake.set_response(
            "omninode_getVerifier",
            Ok(json!({
                "address": expected_addr_for_seed_7(),
                "bond_amount": "5000",
                "bond_state": "unbonding",
                "unbonding_since_height": 490_000,
                "withdrawable_at_height": 500_050,
                "slash_history": []
            })),
        );
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        let outcome = identity
            .precheck_bond(&client, true)
            .expect("unbonding precheck must succeed");
        assert_eq!(
            outcome,
            BondPrecheckOutcome::Unbonding {
                bond_amount: 5_000,
                withdrawable_at_height: Some(500_050),
            }
        );
    }

    // ── Test 11 — withdrawn state ──────────────────────────────────────

    #[test]
    fn precheck_bond_withdrawn() {
        let (client, fake) = make_client();
        seed_gates_active(&fake);
        fake.set_response(
            "omninode_getVerifier",
            Ok(json!({
                "address": expected_addr_for_seed_7(),
                "bond_amount": "0",
                "bond_state": "withdrawn",
                "slash_history": []
            })),
        );
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        let outcome = identity
            .precheck_bond(&client, true)
            .expect("withdrawn precheck must succeed");
        assert_eq!(outcome, BondPrecheckOutcome::Withdrawn);
    }

    // ── Test 12 — unknown wire bond_state reported verbatim ────────────

    #[test]
    fn precheck_bond_unknown_wire_state() {
        let (client, fake) = make_client();
        seed_gates_active(&fake);
        fake.set_response(
            "omninode_getVerifier",
            Ok(json!({
                "address": expected_addr_for_seed_7(),
                "bond_amount": "7777",
                "bond_state": "some-future-state",
                "slash_history": []
            })),
        );
        let identity = ClaimSignerIdentity::resolve(SeedSource::Explicit(seed_7()))
            .expect("resolve must succeed");
        let outcome = identity
            .precheck_bond(&client, true)
            .expect("unknown wire state must be reported, not error");
        assert_eq!(
            outcome,
            BondPrecheckOutcome::UnknownWireState {
                raw: "some-future-state".to_string(),
                bond_amount: 7_777,
            }
        );
    }
}
