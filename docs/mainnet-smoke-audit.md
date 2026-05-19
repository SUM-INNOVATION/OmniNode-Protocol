# Mainnet Smoke Audit — Phase 5 Stage 8b

**Status: PASS**

First real OmniNode `InferenceAttestation` submitted and finalized on
SUM Chain mainnet after the OmniNode subprotocol activated at block
6,000,000. Run executed by the operator using the Stage 8b `omni-node
operator` commands on `main` (`5d972d5`). This note records on-chain /
operational facts only — **no private seed material, no local
secrets**. `verifier_address`, `tx_hash`, and `attestation_id` are
public on-chain values.

## Environment

| Field | Value |
|---|---|
| Network | SUM Chain mainnet |
| RPC | `https://rpc.sumchain.io` |
| `chain_id` | `1` |
| `v2_enabled_from_height` | `5,200,000` |
| `omninode_enabled_from_height` | `6,000,000` |
| Observed head at preflight | `6,049,148` |
| Observed `submitted_at_block` | `6,049,200` |

## Attestation

| Field | Value |
|---|---|
| `session_id` | `mainnet-smoke-2026-05-18-001` |
| `verifier_address` | `2mvPk4h883B7DrcZvwy7yWKXyGYHuVzGP` |
| `attestation_id` | `d77d8e95c96e6ae2264cbe3baf1383d9a3ea82e59a49d7fbf97574a04d791f1d` |

Source attestation was a real operator-provided `InferenceAttestation`
file (`--attestation-json`), **not** synthetic. `--synthetic` was not
used (and is forbidden on `chain_id = 1`).

## Submit

| Field | Value |
|---|---|
| `tx_hash` | `0x3a9cbf85945136e55a3ab8bb04a09d406d52438d9c2fa1f77850a706a1c32a56` |
| `fee` | `1000` (chain `min_fee`) |
| `submitted_at` | `2026-05-19T05:31:08Z` |

Exactly one real `sum_sendRawTransaction` was observed.

## Finality

| Field | Value |
|---|---|
| Status path observed by `smoke` | `Submitted → Included → Finalized` |
| `included_at_height` | `6,049,201` |
| `finalized` | `true` |
| Finalized observed at | `2026-05-19T05:31:19Z` |

Observed mainnet finality latency: ~**11 s** wall-clock from submit
(`05:31:08Z` → `05:31:19Z`), inclusion ~1 block past
`submitted_at_block` (`6,049,200` → `6,049,201`). `smoke` correctly
treated `Included` as progress only and returned success solely on
`Finalized`. The run used `--confirm-timeout-secs 300`, which was far
more than required.

## Query

- `operator query` located the record by `(session_id,
  verifier_address)`.
- Query `tx_hash` matched the submit `tx_hash`.
- Query `finalized = true`.

## Local idempotency

- Re-running the exact same `operator smoke` command with the same
  `--registry-path` reused the same `tx_hash`.
- **No new chain submission** was observed.
- Finalized confirmed immediately from the existing local record.

This exercises the local-registry guard (Stage 5 idempotency), **not**
chain-side duplicate rejection (which was not tested — would require a
fresh registry, deliberately out of scope here).

## Loop checks

| Run | Result |
|---|---|
| Monitor-only loop (`--max-ticks 1`, no `--allow-submit`) | local record `Submitted → Finalized`; `swept=0`, `retried=0`, `errors=0` |
| Retry-enabled loop (`--max-ticks 1`, `--allow-submit --allow-mainnet-submit`) | no local `Dropped` records; `polled=0`, `swept=0`, `retried=0`, `errors=0` |

## Known non-blocking warning

`cargo run -p omni-node` emits a build-hygiene warning:
`SeedSource::{Explicit, AbsentForTest, MalformedForTest}` are
test-only injection variants and are `dead_code` in production
builds. This did **not** affect runtime or smoke success. A
warning-clean fix is scoped into the Stage 8c plan (not folded into
this run record).

## Outcome

Phase 5 end-to-end is proven on mainnet: Stage 4/6 attestation →
Stage 7b SUM Chain submit → Stage 5.1/5.2/5.3 lifecycle → Stage 8a/8b
operator surface, finalized on `chain_id = 1`.

---

## Follow-up (pointer only — the run record above is frozen)

The facts above are an immutable historical record of the 2026-05-19
run and are intentionally **not** edited to match later UX. Hardening
prompted by this run was implemented in **Stage 8c** (see the Stage 8c
section in [`README.md`](../README.md)): the noted `SeedSource`
build-hygiene warning is now resolved (`#[cfg(test)]`-gated, the
production build is warning-clean), `operator smoke` prints a
consolidated summary block, `operator query` gained a `--tx-hash`
status mode, and an `operator derive-address` helper was added. None
of this changes what was observed on 2026-05-19.
