# Stage 15.3 — release bundle aarch64 expansion

**Status: shipped as workflow + docs only.** Zero Rust source / Cargo /
`ci.yml` / schema / enum / proof / contributor / `error.rs` changes.
`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` stays `&[]`. The verified
`v0.1.0` release path (the existing `x86_64-unknown-linux-gnu`
binaries + cosign keyless signature + Sigstore Fulcio cert) remains
behavior-equivalent. The 2026-05-21 and 2026-06-24 RC audits stay
immutable.

Stage 15.3 expands [`.github/workflows/release.yml`](../.github/workflows/release.yml)
from `{x86_64} × {default, submit}` = 2 binaries to
`{x86_64, aarch64} × {default, submit}` = 4 binaries. Per-tag bundle
inventory grows by 2 binaries (+ 0 snapshots; see §3 below);
cosign-signed `SHA256SUMS` covers all of it.

Stage 15.3 does **not** ship: the `production-prove` release variant
(Option B in the plan), SLSA build provenance (Option C), macOS /
Windows targets, package managers, container images, or any
release-workflow restructure. Those remain Stage 15.4+ candidates.

The first aarch64 release artifact lands when an operator cuts
`v0.1.1` (or a later tag) post-merge. `v0.1.0` remains permanently
x86_64-only — re-cutting `v0.1.0` to include aarch64 is forbidden
per the cosign / Rekor transparency-log integrity invariant.

---

## 1. Why aarch64

Per the Stage 15.3 plan §6 recommendation, aarch64 was picked as the
narrowest first slice with the highest immediate production value:

- **Real operator demand today.** Cloud aarch64 instances (AWS
  Graviton, GCP Tau T2A, Azure Ampere) are common production
  hardware. Operators running OmniNode on those hosts cannot use the
  `v0.1.0` x86_64 binaries — they must build from source.
- **Lowest novel risk** conditional on runner availability, which was
  confirmed by the Stage 15.3 probe PR #70 (run
  [28197315331](https://github.com/SUM-INNOVATION/OmniNode-Protocol/actions/runs/28197315331)).
- **Pattern-validates the matrix.** Stage 15.4+ candidates (production-prove,
  SLSA, additional arches) attach cleanly to a proven 2-arch matrix.
- **Value lands immediately**, unlike Option B whose production value
  ripens with Stage 11d.3C consumption (chain-side blocker).

---

## 2. Runner availability — probe outcome

Pre-15.3 probe PR #70 (closed unmerged per the throwaway pattern)
confirmed `ubuntu-24.04-arm` runners are available to this repo.

| Probe fact | Value |
| --- | --- |
| Run ID | [28197315331](https://github.com/SUM-INNOVATION/OmniNode-Protocol/actions/runs/28197315331) |
| Conclusion | `success` |
| `uname -a` | `Linux runnervmjddhd 6.17.0-1018-azure #18~24.04.1-Ubuntu SMP … aarch64 aarch64 aarch64 GNU/Linux` |
| `arch` / `uname -m` | `aarch64` |
| OS | `Ubuntu 24.04.4 LTS (Noble Numbat)` |
| Run time | < 1 minute (no queueing) |

Per Stage 15.3 Q2 lock, this means Stage 15.3 uses **native
`ubuntu-24.04-arm`** — the `cross`-based fallback was not needed.

---

## 3. Workflow design

### Matrix expansion

```yaml
strategy:
  fail-fast: false
  matrix:
    target:
      - { triple: x86_64-unknown-linux-gnu,  runner: ubuntu-24.04,     file_grep: x86-64  }
      - { triple: aarch64-unknown-linux-gnu, runner: ubuntu-24.04-arm, file_grep: aarch64 }
    variant:
      - { name: default, features: "" }
      - { name: submit,  features: "submit" }
```

`runs-on: ${{ matrix.target.runner }}` switches per matrix entry. The
cargo build command interpolates `--target ${{ matrix.target.triple }}`,
filenames embed both `${{ matrix.target.triple }}` and the variant
name, and the cache key includes the triple so x86_64 / aarch64
caches don't collide on the same variant.

The matrix expands from 2 jobs to 4 jobs running in parallel.

### Bundle inventory (per `v0.1.1+` release)

| File | Per-arch? | Notes |
| --- | --- | --- |
| `omni-node-v<TAG>-x86_64-unknown-linux-gnu-default` | x86_64 | Stripped release binary, no features |
| `omni-node-v<TAG>-x86_64-unknown-linux-gnu-submit` | x86_64 | Stripped release binary, `--features submit` |
| `omni-node-v<TAG>-aarch64-unknown-linux-gnu-default` | **aarch64** (new) | Stripped release binary, no features |
| `omni-node-v<TAG>-aarch64-unknown-linux-gnu-submit` | **aarch64** (new) | Stripped release binary, `--features submit` |
| `omni-node-default-version.txt` | x86_64-only capture | `--version` output. Identical byte-for-byte on aarch64 (CARGO_PKG_VERSION-based) — capturing once keeps the bundle clean. |
| `omni-node-submit-version.txt` | x86_64-only capture | Same. |
| `omni-node-default-help.txt` | x86_64-only capture | `--help` output. clap-derived; arch-independent. |
| `omni-node-submit-help.txt` | x86_64-only capture | Same. |
| `SHA256SUMS` | — | Sorted hashes over every file above |
| `SHA256SUMS.sig` | — | Detached cosign signature |
| `SHA256SUMS.cert` | — | Fulcio certificate |

**Total: 11 files per release** (4 binaries + 2 version + 2 help + 3
signing). Bundle delta from v0.1.0: **+2 binaries**.

### `--version` / `--help` snapshot capture (x86_64 only)

The bundle deduplicates these because their content is arch-independent:

- `--version` reads `env!("CARGO_PKG_VERSION")` at compile time; same
  Cargo metadata produces the same bytes regardless of host arch.
- `--help` text is generated by clap from the source code's
  `derive(Parser)` annotations; arch-independent.

So shipping per-arch copies of identical text would just duplicate
file bytes and grow the bundle without informational gain. The
workflow's `Capture --version + --help snapshots (x86_64 only)` step
is conditional (`if: matrix.target.triple == 'x86_64-unknown-linux-gnu'`)
so only the x86_64 matrix entries upload these snapshots.

**aarch64 runtime executability is not skipped.** Every matrix entry
(both arches) runs:

- A `Smoke --version` step that executes the just-built binary on the
  native runner. The aarch64 binary runs on `ubuntu-24.04-arm` —
  proving the binary actually executes on aarch64 hardware.
- A `Defense in depth — assert binary --version matches tag` step
  that re-asserts the captured semver matches the tag.

### Q6 lock — `file` ELF check

Every matrix entry runs:

```bash
OUT=$(file -b "$BIN")
echo "file: $OUT"
if ! grep -qE "ELF 64-bit LSB .*${{ matrix.target.file_grep }}" <<<"$OUT"; then
  echo "::error::ELF arch mismatch …"
  exit 1
fi
```

The `file_grep` per-matrix-entry tag (`x86-64` for the Intel target,
`aarch64` for the ARM target) catches a mis-configured `--target`
flag at the workflow level. If the build step somehow produced an
x86_64 ELF in an aarch64 job (e.g. cached wrong toolchain), the
job fails loudly rather than smuggling a wrong-arch binary into the
release bundle.

### Cosign signing — unchanged

The `bundle-and-sign` job:

- Downloads all 4 matrix-entry artifacts.
- Generates a sorted `SHA256SUMS` over the merged `dist/`.
- Runs `sha256sum -c SHA256SUMS` round-trip.
- Signs `SHA256SUMS` via cosign keyless OIDC (`sigstore/cosign-installer@v3`,
  `cosign-release: 'v2.4.1'` — same as Stage 15.2).
- Verifies the signature in-workflow against the production cert
  identity regex (on tag / dispatch runs) or a repo-scoped permissive
  regex (on PR dry-runs).

**The production cert identity regex is unchanged** by Stage 15.3:

```
https://github\.com/SUM-INNOVATION/OmniNode-Protocol/\.github/workflows/release\.yml@refs/tags/v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$
```

The cert identity binds to the workflow file path + tag ref. Stage 15.3's
matrix expansion is internal to that workflow; the bound identity is
the same.

### Publish-draft-release — unchanged

`softprops/action-gh-release@v2` with `draft: true`, gated on
`push: refs/tags/v*`. Operator promotes manually after verifying.

---

## 4. Operator verification — what changes

**The verification recipe is unchanged.** The cosign-verify-blob
command, the cert identity regex, the `sha256sum -c` step — all
identical to Stage 15.2's published recipe in
[`docs/stage15.2-release-artifact-workflow.md`](stage15.2-release-artifact-workflow.md)
§4 and the operator runbook §14 step 9.

The only operator-visible difference is the bundle has 2 extra
binaries. Operators on aarch64 hosts now download
`omni-node-v<TAG>-aarch64-unknown-linux-gnu-{default,submit}` instead
of building from source.

### One-liner verification on aarch64

```bash
cd $(mktemp -d)
gh release download v0.1.1 \
  --repo SUM-INNOVATION/OmniNode-Protocol \
  --pattern '*'

cosign verify-blob \
  --certificate         SHA256SUMS.cert \
  --signature           SHA256SUMS.sig \
  --certificate-identity-regexp \
    'https://github\.com/SUM-INNOVATION/OmniNode-Protocol/\.github/workflows/release\.yml@refs/tags/v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  SHA256SUMS

sha256sum -c SHA256SUMS

chmod +x omni-node-v0.1.1-aarch64-unknown-linux-gnu-submit
./omni-node-v0.1.1-aarch64-unknown-linux-gnu-submit --version
# Expected: omni-node 0.1.1
file ./omni-node-v0.1.1-aarch64-unknown-linux-gnu-submit
# Expected: ELF 64-bit LSB executable, ARM aarch64, …
```

---

## 5. What's still out of scope

Stage 15.3 deliberately does **not** ship:

- `production-prove` release variant (Option B in the plan — Stage 15.4 candidate; value gated by chain-side Stage 11d.3C unblock).
- SLSA build provenance via `actions/attest-build-provenance` (Option C — Stage 15.4 candidate; cosign already provides Sigstore/Rekor provenance, SLSA layers structured metadata on top).
- macOS / Windows release binaries.
- Package managers (`.deb` / `.rpm` / Homebrew / Snap / Flatpak / AUR).
- Container images (Docker / Helm).
- SBOM emission (CycloneDX / SPDX).
- Templated `RELEASE_NOTES.md`; per-release fixtures manifest; docs tarball.
- Mutation of cosign cert identity regex.
- Mutation of either RC audit doc.
- `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` mutation, chain registry consumption, activation, `Active` record.
- Economics / staking / slashing / rewards.
- New proof systems / `ProofSystem` / `ModelFormat` variants.
- Mutation of [`ci.yml`](../.github/workflows/ci.yml).
- New GitHub secrets.
- Cutting `v0.1.1` (or any subsequent tag) in this PR — tags are
  separate operator actions.

---

## 6. Stage 15.3A — first aarch64 release ratification (post-merge)

Mirrors the Stage 15.2 → 15.2A pattern. After Stage 15.3 merges:

1. An operator opens a small "workspace version bump" PR
   (`Cargo.toml` `[workspace.package].version = "0.1.0"` → `"0.1.1"`).
2. After that PR merges, the operator cuts `v0.1.1`. The release
   workflow produces 4 binaries, 6 metadata files, and the cosign
   bundle. Draft Release published.
3. An operator verifies on a **separate aarch64 host** (not the
   GitHub Actions runner) via the recipe in §4. Repeat the same
   verification on an x86_64 host. Run `file` on each binary to
   confirm both ELF arches are correct.
4. Promote `v0.1.1` to non-draft + `--latest`.
5. Stage 15.3A docs PR ratifies: README "Stage 15.3 — Complete
   (verified via v0.1.1)" status row, this doc gets a §7 first-arm-release
   append (mirroring Stage 15.2 doc §9), operator runbook §14 gets
   a second worked example.

Stage 15.3A is itself docs-only. No further code / CI / workflow
changes.

---

## 7. Stage 15.4 candidates (deferred)

For the next round, when the time comes:

- **`production-prove` release variant** (Option B). 3rd variant axis;
  `--features submit,stage11d-production-prove`. Runtime requires
  `RUST_MIN_STACK=67108864`; documented in the runbook. Value gated
  by Stage 11d.3C consumption.
- **SLSA build provenance** (Option C). `actions/attest-build-provenance`
  attesting `SHA256SUMS`. Adds `attestations: write` permission to
  the bundle-and-sign job; verification recipe extends with
  `gh attestation verify` and `cosign verify-attestation`.
- **SBOM** (CycloneDX / SPDX) via `actions/attest-sbom`.
- **macOS / Windows targets** (gated on `omni-net/src/identity.rs`
  cross-platform review first).
- **Templated `RELEASE_NOTES.md`** — generated body with per-stage
  rollup and `Cargo.lock` dep delta.
- **Per-release fixtures manifest**.

Each is a separate stage with its own plan-only review cycle.
