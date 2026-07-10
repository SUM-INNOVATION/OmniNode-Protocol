# InferenceSettlement Protocol Reference

Current-state protocol reference for the SUM Chain `InferenceSettlement`
subprotocol as consumed by OmniNode. Covers the on-chain activation
gates, the eligibility model for verifier claims, the validator-quorum
dispute-resolution model, verifier bonding, consistency / plurality
mode, sponsored attestation, and the boundary of what chain state
mechanically enforces vs. what OmniNode enforces off-chain as
defense-in-depth.

**Scope.** This is the operator- and integrator-facing reference for
what the settlement subprotocol *is*. Historical evidence bundles,
per-issue decision records, and CI-side gates live in their respective
issue PRs / stage docs and are not re-cited here.

**Non-mainnet-active claim.** No OmniNode doc, error message, or
release note may state or imply that settlement is active on a specific
chain unless the relevant activation param on that chain is
`Some(<past-height>)` and OmniNode has independently observed a head at
or beyond that height. Dormancy detection is enforced locally by the
OmniNode client against `chain_getChainParams` — chain read RPCs may
still return empty state while a subprotocol gate is dormant, and
empty state is never evidence of activation.

---

## 1. Activation gates

The settlement subprotocol ships as five independently-configurable
chain-param gates. Each is `Option<u64>` on
`chain_getChainParams`; the subprotocol behind that gate is active at
the block height inside `Some(_)` once head has reached it.

| Chain param | Governs |
| --- | --- |
| `inference_settlement_enabled_from_height` | Base settlement subprotocol: sessions, claims, disputes, refunds. |
| `inference_settlement_consistency_enabled_from_height` | Optional consistency / plurality reward mode over full-digest-tuple groups. |
| `inference_verifier_bonding_enabled_from_height` | Verifier registry + bonded stake. Required for bond-required sessions. |
| `omninode_sponsored_attestation_enabled_from_height` | Sponsored attestation v2 — a distinct chain surface from settlement claims (see §7). |
| `inference_settlement_dispute_threshold_bps` | Basis-point threshold that a `ResolveDispute` tx's validator approvals must meet over the active PoA validator set. Not a height gate; a governance-set integer. |

OmniNode enforces each gate locally. Any `None` reading, or any
`Some(N)` with `head < N`, is dormant for that gate.

---

## 2. Chain-side settlement summary

- **Separate subprotocol from `InferenceAttestation`.** Settlement
  ships as a distinct on-chain subprotocol. It references attestations
  by identity but does not own them; the attestation schema is
  unchanged (`InferenceAttestationDigest` and
  `InferenceAttestationTxData` are pinned).
- **Attestation identity used by settlement:** the tuple
  `(session_id, verifier_address)`. This tuple uniquely identifies the
  attestation a claim references. In consistency-mode sessions, the
  chain additionally indexes attestations by the full digest tuple
  `(model_hash, manifest_root, response_hash, proof_root)` to group
  equivalent inferences (see §5).
- **Escrow-funded and supply-conserving.** Each session is backed by a
  funder-provided escrow deposit. Rewards are paid **out of the
  session escrow**; nothing is minted; total supply is conserved.
  Unclaimed escrow returns to the funder via the refund path once
  eligible claims have matured or been resolved.
- **No unbounded rewards.** Every reward disbursement is bounded by
  the escrow balance remaining for the session; the runtime rejects
  claims that would over-draw.
- **No on-chain AI semantic-correctness verification.** The chain does
  not evaluate whether an attested inference is semantically correct.
  Settlement enforces escrow-funded eligibility, signatures, maturity,
  disputes, and (where activated) consistency and bond rules.
  Off-chain proof artifacts + verifier attestations anchor
  contribution commitments; on-chain zkML / Halo2 verification is
  **not** part of the settlement subprotocol.

---

## 3. Claim eligibility

A verifier claim (`ClaimReward` tx via
`omninode_buildClaimInferenceReward` + `sum_sendRawTransaction`) is
accepted only when every rule below holds. Any single failure fails
the tx with a typed reason; there is no partial claim.

1. **Settlement gate active.**
   `inference_settlement_enabled_from_height` is `Some(N)` and current
   head `>= N`.
2. **Session exists and is `Open`.** The session referenced by the
   claim is registered on chain and is not in a terminal state
   (`Settled` / `Refunded`).
3. **Matching attestation exists.** An on-chain `InferenceAttestation`
   matches the tuple `(session_id, verifier_address)` where
   `verifier_address` equals the claim tx signer.
4. **Claim signer is the verifier address.** The transaction is
   signed by the same chain address that holds the attestation.
   **Verifier self-claim only.** No coordinator-claim; no
   permissionless claim-on-behalf; no contributor-triggered tx path.
5. **Maturity — the claim window has opened.**
   ```
   claim_ready_block = attestation.included_at_height
                     + chain_params.finality_depth
                     + session.dispute_window_blocks
   ```
   The claim is accepted only when `head >= claim_ready_block`. Never
   write `tx_finalized_block + finality_depth + …` — that would
   double-count finality if `tx_finalized_block` already denotes
   finalized height.
6. **No duplicate claim.** A given verifier may claim at most once
   per `(session_id, verifier_address)`.
7. **No unresolved or `ResolvedDenyClaim` dispute.** If a dispute
   against the attestation is in `Open` state, the claim is refused
   pending resolution. If a dispute has resolved with
   `ResolvedDenyClaim`, the claim is permanently refused — this is
   the **reward denial** path (§8).
8. **Cap available.** `session.claims_count < session.max_verifiers`.
9. **Escrow available.** Session escrow has enough balance to cover
   the reward this claim would disburse.
10. **Bond requirement met (bond-required sessions only).** For
    sessions with `bond_required == true`, the verifier's registry
    entry (via `omninode_getVerifier`) must exist and be in `Bonded`
    state. See §6.
11. **Consistency requirement met (consistency-mode sessions only).**
    For sessions with `consistency_required == true`, the verifier's
    attestation's full digest tuple must belong to the plurality
    consistency group at claim-ready time. See §5.

OmniNode's local claim CLI (`operator settlement claim`) enforces
rules 1, 3, 4, 5, 7 (via chain-provided `claimable_now`), and 10
locally before invoking the chain builder RPC, so the operator
doesn't burn a fee on a chain-rejected tx.

---

## 4. Disputes

The dispute-resolution authority model is **validator-quorum**, not
solo-resolver:

- **`OpenDispute`** is signed by the **session funder** only. The
  chain has no permissionless dispute-opening surface in this
  subprotocol.
- **`ResolveDispute`** is authorized by validator approvals whose
  weight reaches `inference_settlement_dispute_threshold_bps` over
  the active PoA validator set at the block the resolution is
  applied. **Non-signing validators count in the denominator as
  abstentions.** The threshold is computed against the full active
  validator set, not against a subset of "engaged" validators; enough
  abstentions can still prevent quorum from being reached.
- **`tx.from` on `ResolveDispute` is the fee payer, not the
  authority.** Any address can submit if the tx carries sufficient
  validator approvals; a "resolver address" without approvals cannot
  succeed.
- **There is no solo-resolver key.** OmniNode does not operate one
  and never has. Any earlier draft language that implied a solo
  resolver is superseded.

Dispute lifecycle states OmniNode consumes on the read surface:

| State | Semantic |
| --- | --- |
| `Open` | Under review; claims blocked pending resolution. |
| `ResolvedAllowClaim` | Chain-team public term for the "claim proceeds" resolution. |
| `ResolvedDenyClaim` | Chain-team public term for the "claim refused" resolution. Reward is denied; for bond-required sessions this can additionally trigger bond penalty per §6. |

OmniNode's operator dispute CLI (`operator settlement dispute
open` / `resolve`) delegates tx construction to
`omninode_buildOpenInferenceDispute` /
`omninode_buildResolveInferenceDispute` and locally enforces funder
authority (open), open-status of the session (open), the maturity
window (open — dispute must open before the target claim matures),
duplicate-dispute (open — one open dispute per verifier per session),
`Open`-state of the target dispute (resolve), and non-empty approvals
(resolve).

---

## 5. Consistency / plurality mode

Sessions may be created with `consistency_required == true`. The
chain-team public grouping model:

- Attestations are indexed by the full digest tuple
  `(model_hash, manifest_root, response_hash, proof_root)`.
- All attestations sharing the exact same 4-tuple form one
  **consistency group**.
- The chain resolves a **plurality group** — the largest group by
  member count — at claim-ready time.
- In consistency-mode sessions, a verifier's claim is eligible only
  if that verifier's attestation belongs to the plurality group.
- Verifiers whose attestations land in a losing (non-plurality)
  group receive reward denial via §3 rule 11.

OmniNode's read surface exposes group state via
`omninode_getInferenceConsistency` (feature-gated on
`settlement-read` + the consistency chain gate active) and composes
per-verifier group membership into `SettlementSessionView` when the
session is consistency-mode.

Consistency mode is optional; sessions with
`consistency_required == false` continue to reward all eligible
verifiers up to `max_verifiers` per §3.

---

## 6. Verifier registry, bonds, and bond penalty

Verifier bonding ships as a distinct chain gate:
`inference_verifier_bonding_enabled_from_height`. When active:

- **Verifier registry** — verifiers register on-chain via
  `RegisterVerifier` with an initial bond. Registry state (`Bonded`
  / `Unbonding` / `Withdrawn` plus bond amount + unbonding schedule)
  is readable via `omninode_getVerifier`.
- **Verifier bonds** — a verifier's registered bond amount is
  on-chain stake tied to that verifier's chain identity. Additional
  bond can be added via `AddVerifierBond`; a verifier may initiate
  withdrawal via `BeginVerifierUnbond` followed by
  `WithdrawVerifierBond` after the unbonding schedule matures.
- **Bond-required sessions** — sessions may be created with
  `bond_required == true`. Verifiers whose registry state is not
  `Bonded` at claim-ready time are refused per §3 rule 10.

**Bond penalty** (previously discussed in some earlier drafts as
"slashing"):

- Applies **only** when both conditions hold:
  1. The target session is bond-required.
  2. A dispute against the verifier resolves with
     `ResolvedDenyClaim` under the validator-quorum authority
     described in §4.
- Under those two conditions, chain-side execution deducts penalty
  from the verifier's bonded stake. The exact penalty schedule is
  chain-side policy and is not restated here.
- **Reward denial alone is not bond penalty.** A claim can fail any
  §3 rule (missing attestation, escrow exhausted, cap reached,
  maturity not met, duplicate, non-plurality group, etc.) without
  any stake being taken. Reward denial is the general outcome; bond
  penalty is a narrower outcome gated on the two conditions above.

OmniNode's operator surface never refers to bond penalty as
"slashing." The runtime CLI shows registry state + bond amount via
`operator settlement verifier <address>`; policy interpretation lives
with the chain team.

---

## 7. Sponsored attestation v2

`omninode_sponsored_attestation_enabled_from_height` gates a distinct
chain surface — **sponsored attestation v2** — which allows third-party
funding of attestation costs. This is **not** part of the settlement
claim path:

- Sponsored attestation lives on the attestation submission side, not
  the settlement claim side.
- A sponsored-attestation tx does not trigger any `ClaimReward` or
  `ResolveDispute` behavior.
- OmniNode's settlement code paths do not consume the sponsored
  attestation gate. It is documented here because the same chain
  release ships the gate; conflating the two surfaces has historically
  been a source of confusion.

Any future OmniNode integration with sponsored attestation is a
distinct workstream from the settlement track.

---

## 8. Reward denial vs. bond penalty

A verifier's claim can be **denied** for many reasons drawn from §3
(missing attestation, cap reached, escrow exhausted, non-plurality
group, `ResolvedDenyClaim` dispute, and others). Under most of these
outcomes **no stake is taken**; the verifier simply does not receive
that reward.

**Bond penalty** is narrower: it applies only when a bond-required
session has a `ResolvedDenyClaim` dispute against the verifier per §6.

OmniNode-side invariants:

- No OmniNode doc, error message, marker, or code path refers to
  reward denial as "slashing."
- No OmniNode-side surface can trigger bond penalty independently
  of chain execution. All bond-penalty outcomes are chain-authorized.
- The word "slashing" is avoided across OmniNode operator-facing
  surfaces to prevent confusion between the two distinct outcomes.

---

## 9. Canonical terminology

Every OmniNode doc, error message, and code comment in the
InferenceSettlement track uses these words for these concepts.

| Term | Definition |
| --- | --- |
| **Session** | A settlement scope on chain, registered by the funder with an escrow deposit, identified by `session_id`. May be consistency-mode (`consistency_required`) and/or bond-required (`bond_required`). |
| **Funder** | The address that opens a session by depositing escrow. The sole legal signer of `OpenDispute`. Reclaims unspent escrow via the refund path. |
| **Verifier** | The chain-address role that holds an `InferenceAttestation` and is authorized to submit its own `ClaimReward`. **Verifier self-claim only.** |
| **Contributor** | The off-chain role that performs the inference work and produces a `ContributorResult`. Identity mapping to a chain verifier address is operator-configured; the chain-visible identity is the verifier address that signs the claim tx. |
| **Reward** | The payout to a verifier for a valid claim against an attested session. Paid out of the session escrow. Supply-conserving. |
| **Claim** | The `ClaimReward` transaction a verifier submits. Subject to the eligibility rules in §3. |
| **Dispute** | A challenge lodged against an attestation by the session funder. Resolution is validator-quorum authorized (§4). |
| **`ResolvedAllowClaim` / `ResolvedDenyClaim`** | Chain-team public terminology for the two dispute-resolution outcomes. |
| **Consistency group** | A group of attestations sharing the same `(model_hash, manifest_root, response_hash, proof_root)` 4-tuple. The **plurality group** is the largest such group at claim-ready time; in consistency-mode sessions, only members of the plurality group are eligible per §3 rule 11. |
| **Verifier registry** | On-chain record of verifier identities plus their bond state, active when `inference_verifier_bonding_enabled_from_height` is active. Readable via `omninode_getVerifier`. |
| **Verifier bond** | Bonded stake tied to a registered verifier. Prerequisite for participating in bond-required sessions. |
| **Bond-required session** | A session with `bond_required == true`. Claim eligibility (§3 rule 10) requires the verifier to be `Bonded`. Reward denial via `ResolvedDenyClaim` dispute additionally triggers bond penalty (§6). |
| **`inference_settlement_dispute_threshold_bps`** | Governance-set basis-point threshold that a `ResolveDispute` tx's validator approvals must meet over the active PoA validator set. |
| **`included_at_height`** | The chain height at which the attestation tx was included. |
| **`finality_depth`** | Chain-param blocks after `included_at_height` at which the attestation is considered finalized. |
| **`dispute_window_blocks`** | Session-carried block window during which disputes can be opened after finalization. |
| **`claim_ready_block`** | `attestation.included_at_height + chain_params.finality_depth + session.dispute_window_blocks`. Never `tx_finalized_block + finality_depth + …`. |
| **Reward denial** | The outcome when a verifier's claim is refused for any §3 reason. **No stake taken by itself.** |
| **Bond penalty** | The narrow outcome where bonded stake is taken. Requires §6's two conditions. **Not called "slashing" in OmniNode surfaces.** |
| **Sponsored attestation v2** | Distinct chain surface gated by `omninode_sponsored_attestation_enabled_from_height` for third-party funding of attestations. **Not part of the settlement claim path.** |

---

## 10. What is NOT in this subprotocol

Explicit boundary against a few adjacent concepts that historically
have been confused with settlement:

- **No chain semantic AI correctness verification.** Settlement
  enforces escrow-funded eligibility, signatures, maturity, disputes,
  and (where activated) consistency and bond rules. It does not
  evaluate whether an attested inference is semantically correct.
- **No on-chain Halo2 / zkML verification.** The chain does not
  execute a zkML circuit or verify a Halo2 proof as part of
  settlement. Off-chain proof artifacts + verifier attestations
  anchor contribution commitments; the chain reads attestation
  identity, not proof correctness.
- **No coordinator claim.** Only the verifier holding the
  attestation may claim.
- **No permissionless claim-on-behalf.**
- **No contributor-triggered claim tx path.**
- **No auto-claim daemon.**
- **No solo dispute resolver key.** Dispute resolution is
  validator-quorum only.
- **Sponsored attestation is a separate surface**, not a settlement
  claim variant (§7).

---

## Cross-references

- OmniNode operator runbook: [`operator-runbook.md`](operator-runbook.md).
- Off-chain production MLP proof surface (distinct from settlement):
  [`production-mlp-proof.md`](production-mlp-proof.md).
- Immutable Phase 5 release-candidate audit snapshot at 2026-06-24:
  [`phase5-rc-audit-2026-06-24.md`](phase5-rc-audit-2026-06-24.md).
- SUM Chain issue tracker (upstream): `sum-chain#76`, `sum-chain#86`,
  `sum-chain#110`.
