# Stage 15.2 — Release artifact workflow with cosign keyless signing

**Status: shipped as workflow + docs only.** Zero Rust source / Cargo /
schema / enum / proof artifact / contributor schema / `error.rs` /
[`ci.yml`](../.github/workflows/ci.yml) changes. No `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`
mutation, no eligibility activation, no `Active` record, no chain RPC
change, no economics / staking / slashing / rewards implementation, no
EZKL revisit, no live-chain CI tests.

Stage 15.2 closes the **artifact signing / provenance** finding the
2026-06-24 Phase 5 RC audit
([`docs/phase5-rc-audit-2026-06-24.md`](phase5-rc-audit-2026-06-24.md)
§3b) rated **Missing**. After Stage 15.2 ships and the inaugural
`v0.x.y` tag verifies on an independent host, the §3b rating moves to
**Shipped**. The audit itself is **not edited** (immutable per Stage
15.0/15.1 convention); the rating change is recorded here.

Stage 15.2 adds [`.github/workflows/release.yml`](../.github/workflows/release.yml)
and updates [`docs/operator-runbook.md`](operator-runbook.md) §14 to
match. Cutting the inaugural `v0.1.0` tag is **a separate operator
action after this stage merges** — Stage 15.2 ships the machinery, not
the tag.

---

## 1. Scope

| In scope | Out of scope |
| --- | --- |
| `omni-node` release binaries for `x86_64-unknown-linux-gnu` only | aarch64 (Stage 15.3 candidate, pending GitHub-hosted arm runner availability check) |
| Two feature variants: `default` (lean monitor-only operator) and `submit` (chain-submitting operator) | `production-prove` / `all-prove` / `halo2-reference-prove` release variants (Stage 15.3 candidate) |
| `SHA256SUMS` over every release file, signed with **cosign keyless OIDC** (Sigstore Fulcio + Rekor) | `.deb` / `.rpm` / Homebrew / Snap / Flatpak — future work, not built |
| `--version` and `--help` snapshots per variant | Docker / Helm — future work, not built |
| **Draft** GitHub Release on tag push (manual promotion to non-draft after operator verifies) | SLSA-3 / SLSA-4 build provenance via `actions/attest-build-provenance` — Stage 15.3 candidate |
| Hard tag/version match gate **before** any build runs | Cutting the inaugural `v0.1.0` tag — separate operator action |
| PR dry-run on changes to release-relevant paths | Generated `RELEASE_NOTES.md` template (GitHub's auto-generated body is good enough for v0.1.0) |
| | Fixtures manifest tarball (fixtures already pinned by Stage 6 + Stage 11a CI byte-stability gates) |
| | Docs tarball (operators get docs from source tree at the tag SHA) |
| | macOS / Windows release binaries |
| | `crates.io publish` (workspace is not a publishable library set) |
| | Chain registry activation, Proof Eligibility Registry consumption, economics implementation, EZKL revisit |
| | Edits to either Stage 15.1 (2026-06-24) or 2026-05-21 RC audit docs |

---

## 2. Release bundle inventory

Per `v0.x.y` tag, 9 files end up on the Release page:

| File | Contents |
| --- | --- |
| `omni-node-v<TAG>-x86_64-unknown-linux-gnu-default` | Stripped release binary, no features. The lean operator. |
| `omni-node-v<TAG>-x86_64-unknown-linux-gnu-submit` | Stripped release binary, `--features submit`. The chain-submitting operator. |
| `omni-node-default-version.txt` | `./omni-node-…-default --version` captured during the build matrix. |
| `omni-node-submit-version.txt` | `./omni-node-…-submit --version` captured during the build matrix. |
| `omni-node-default-help.txt` | `./omni-node-…-default --help`. |
| `omni-node-submit-help.txt` | `./omni-node-…-submit --help`. |
| `SHA256SUMS` | One text file. `<sha256-hex>  <filename>` for every file above. Sorted; deterministic. |
| `SHA256SUMS.sig` | Detached cosign signature over `SHA256SUMS`. Produced by the workflow's OIDC token via Sigstore Fulcio. |
| `SHA256SUMS.cert` | Fulcio certificate certifying the workflow's OIDC identity at signing time. |

GitHub's auto-generated source archive (`Source code (zip)` / `(tar.gz)`)
is **not** produced by this workflow but is automatically attached by
GitHub to every Release. It is not covered by `SHA256SUMS` and is not
signed by this workflow; operators who need to verify source-tree
integrity should `git checkout v<TAG>` and verify against the
[`Cargo.lock`](../Cargo.lock) committed at that SHA.

### Naming convention

`omni-node-v<TAG>-x86_64-unknown-linux-gnu-<VARIANT>` — target-triple
first preserves sortability; variant suffix is explicit. No silent
default. Operators paste the filename into release notes verbatim.

---

## 3. Hard tag/version match gate

The workflow's first job (`tag-version-gate`) runs before any matrix
build. It:

1. Determines the run **mode**: `dispatch` (`workflow_dispatch` with
   `inputs.tag`), `tag` (`push: refs/tags/v*`), or `pr` (`pull_request`
   on a release-relevant path).
2. For non-PR modes, asserts the tag name matches
   `^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$`. Fails otherwise.
3. Extracts the workspace version via
   `cargo metadata --format-version 1 --no-deps |
   jq -r '.packages[] | select(.name == "omni-node") | .version'`.
   The `omni-node` package is the only `[[bin]]` in the workspace; its
   `version.workspace = true` inherits from `[workspace.package].version`
   verbatim.
4. For non-PR modes, asserts the tag's semver part equals the workspace
   version. **Hard `exit 1` on mismatch**, before any matrix build is
   started.
5. For PR mode, logs both values (advisory only) — PRs may legitimately
   diverge from the workspace version.

Defense in depth: each `build-matrix` job additionally cross-checks
that the captured `--version` output's semver matches the tag's semver.
A binary that prints a different version than the gate parsed is also a
hard failure.

These two checks ensure a release bundle's filename, binary content, and
`SHA256SUMS` all agree on the same `v<X.Y.Z>` value at signing time.

---

## 4. Cosign keyless OIDC

### Why keyless

- **No long-lived private key** anywhere. No GitHub secret holds a
  signing key; no operator holds a backup `.priv` file; no key-rotation
  / revocation ceremony to maintain.
- **Identity = the workflow's OIDC token.** GitHub Actions injects an
  OIDC token signed by `https://token.actions.githubusercontent.com`
  whose `sub` claim encodes the repo + workflow + ref. Sigstore Fulcio
  consumes the token and issues a short-lived X.509 certificate
  embedding the same identity. The cert + signature + Rekor entry form
  a publicly verifiable record.
- **Rekor transparency log** receives an append-only entry per signing.
  Any operator can independently confirm "this repo's release.yml
  signed this checksums hash at this UTC time" without trusting any
  single party.

### Pinned cosign version

- `actions/cosign-installer@v3` with `cosign-release: 'v2.4.1'`.
- **Bump policy.** Never `@latest`. Bumps happen via an explicit PR that
  edits this doc and the workflow together, with a justification line.
  Cosign minor releases occasionally change CLI flags; the workflow
  pinning isolates the release pipeline from upstream churn.

### Permissions

The `bundle-and-sign` job declares:

```yaml
permissions:
  id-token: write
  contents: read
```

`id-token: write` is required for OIDC token issuance. No other job in
the workflow has it. `publish-draft-release` adds `contents: write`
(needed to create the GitHub Release) but does **not** have
`id-token: write`.

### Signing command (verbatim)

```bash
cosign sign-blob --yes \
  --output-signature SHA256SUMS.sig \
  --output-certificate SHA256SUMS.cert \
  SHA256SUMS
```

### Governance lock — cert identity regex

**Operators MUST verify against the following regex.** It is reproduced
here byte-for-byte so it can be copy-pasted into operator scripts
without ambiguity:

```
https://github\.com/SUM-INNOVATION/OmniNode-Protocol/\.github/workflows/release\.yml@refs/tags/v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$
```

The regex pins three things:

1. **The repository** — `SUM-INNOVATION/OmniNode-Protocol`. Forks
   can't quietly sign for the published identity.
2. **The workflow file** — `.github/workflows/release.yml`. A new
   workflow added later (`release-experimental.yml`, etc.) won't
   verify against this regex.
3. **The ref shape** — `refs/tags/v<semver>` only. Branch builds, PR
   dry-runs, and `workflow_dispatch` runs from non-tag refs produce
   signatures whose cert identity will **not** match this regex even
   though they pass the workflow's internal PR-mode verification.

**Bumping the regex** requires:
- A PR that edits this doc, the workflow's `Verify cosign signature`
  step, **and** every operator-facing recipe that publishes it.
- A justification line on the PR explaining why the new identity is
  authoritative.
- A note in the runbook §14 release-readiness checklist that the regex
  bump landed at commit `<SHA>` so post-bump release verifications
  reference the new regex from that commit onward.

The PR dry-run inside the workflow uses a deliberately permissive
regex (`https://github\.com/SUM-INNOVATION/OmniNode-Protocol/\.github/workflows/release\.yml@.*$`)
so the signing pipeline is exercised end-to-end on PRs without
requiring a tag-bound identity. **PR-dry-run signatures do not verify
against the production regex.** That asymmetry is intentional and is
the reason draft Releases are gated on `push: refs/tags/v*`, not on
PRs.

### Operator verification recipe

```bash
# 1. Download SHA256SUMS, SHA256SUMS.sig, SHA256SUMS.cert from the
#    GitHub Release page for the v<X.Y.Z> tag.

# 2. Verify the signature against the production cert identity regex.
cosign verify-blob \
  --certificate         SHA256SUMS.cert \
  --signature           SHA256SUMS.sig \
  --certificate-identity-regexp \
    'https://github\.com/SUM-INNOVATION/OmniNode-Protocol/\.github/workflows/release\.yml@refs/tags/v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  SHA256SUMS

# 3. If step 2 passes, verify the binaries against SHA256SUMS.
sha256sum -c SHA256SUMS

# 4. (Optional) Inspect the captured --version and --help snapshots and
#    confirm they match what your local binary reports:
./omni-node-v<X.Y.Z>-x86_64-unknown-linux-gnu-submit --version
./omni-node-v<X.Y.Z>-x86_64-unknown-linux-gnu-submit --help
```

If any step fails — wrong identity, missing Rekor entry, hash
mismatch — **stop**. Do not deploy the binary. Open an issue against
this repo and reference the failing step.

---

## 5. Workflow design

### Trigger model

| Trigger | When it fires | Publishes a Release? |
| --- | --- | --- |
| `workflow_dispatch` | Operator-initiated. Takes a `tag` input + `dry_run` boolean. | No (build + sign only). |
| `push: refs/tags/v*` | A `v<semver>`-shaped tag is pushed. | **Yes — draft Release.** Manual promotion to non-draft is the operator's job. |
| `pull_request` on release-relevant paths | A PR modifies `.github/workflows/release.yml`, `Cargo.toml`, `Cargo.lock`, or `rust-toolchain.toml`. | No — dry-run only; bundle uploaded as `actions/upload-artifact`. |

Other PRs do **not** trigger this workflow. Normal CI cost stays
unchanged.

### Jobs (in order)

1. **`tag-version-gate`** — §3. ~10 s.
2. **`build-matrix`** — two variants in parallel, `x86_64-unknown-linux-gnu`. `cargo build --release --locked`, `strip`, capture `--version` + `--help`, defense-in-depth tag/version cross-check.
3. **`bundle-and-sign`** — generate `SHA256SUMS`, `sha256sum -c` round-trip, install cosign `v2.4.1`, sign, verify (production regex for tag/dispatch; permissive repo-scoped regex for PR dry-runs), upload the full bundle as a single `release-bundle` artifact.
4. **`publish-draft-release`** — only on `push: refs/tags/v*` and `!inputs.dry_run`. Downloads the bundle, calls `softprops/action-gh-release@v2` with `draft: true`. `prerelease: true` if the tag contains a `-` (pre-release shape).

### Promotion to non-draft

After `publish-draft-release` runs, the operator:

1. Downloads `SHA256SUMS`, `SHA256SUMS.sig`, `SHA256SUMS.cert` from the
   draft Release page on a **separate host** from the build host.
2. Runs the verification recipe in §4.
3. Spot-checks one or both binaries (e.g., `./omni-node-…-submit --help`).
4. Manually edits the GitHub Release to flip "draft" off, confirming
   `latest` if appropriate.

The flip is a one-click GitHub UI action; no API call. Stage 15.2
deliberately does **not** automate this — the manual step is the
release-acceptance gate.

### Secrets

**Zero new GitHub secrets.** Cosign keyless uses the workflow's OIDC
token; the `id-token: write` permission is granted via
`permissions:` in the workflow's job definition, not via a secret.

### Existing CI is untouched

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) is **not
edited** by Stage 15.2. The build/test matrix (default + submit +
halo2-reference-{verify,prove} + stage11d-production-{verify,prove} +
tree-isolation gates + Stage 6 + Stage 11a fixture byte-stability
gates) continues to run on every PR exactly as before. The release
workflow's PR trigger is scoped to release-relevant paths only;
non-release PRs pay zero release-build cost.

---

## 6. Verification matrix shipped in this PR

| Verification | Where |
| --- | --- |
| Workflow syntax | YAML is well-formed; manually `actionlint`ed before commit. |
| Tag/version gate behavior | PR dry-run runs the gate in advisory mode and prints the parsed values to the workflow log. A future PR that bumps `Cargo.toml`'s version without bumping a downstream tag will surface the mismatch via the `dispatch` / `tag` paths. |
| Build matrix passes | PR dry-run builds both variants; failures surface immediately. |
| Cosign sign + verify round-trip | PR dry-run signs `SHA256SUMS` and verifies against the repo-scoped permissive regex inside the workflow. |
| Production cert identity regex is reachable | Will be confirmed when the inaugural `v0.1.0` tag is pushed post-merge. |
| Operator-side verification recipe | Reproduced in §4 of this doc; will be exercised by the first operator who verifies `v0.1.0` on an independent host. |

---

## 7. Stage 15.3+ candidate scope

The locks from Stage 15.2 explicitly deferred the following items.
Listed here as future-work pointers, not as Stage 15.2 deliverables.

- **aarch64** — `aarch64-unknown-linux-gnu` release binary. Pending a
  one-shot check that GitHub-hosted arm runners (the `ubuntu-24.04-arm`
  label, currently in GA roll-out depending on org/plan) are available
  to this repo. Fall back to `cross`-based cross-compile on
  `ubuntu-24.04` if the native runner is not provisioned.
- **`production-prove` release variant** — adds a third release binary
  per arch. Useful for operators running off-chain production proof
  generation as part of their workflow. Stage 15.3 should justify
  including it in the release set vs. leaving it as a build-from-source
  surface.
- **SLSA build provenance** — `actions/attest-build-provenance` on top
  of cosign keyless. Adds a SLSA-style attestation to the release. Not
  required to satisfy the §3b audit finding, but a credible next step.
- **`actions/attest-sbom`** — emit an SBOM (CycloneDX or SPDX) per
  release bundle. Useful for downstream auditors.
- **Generated `RELEASE_NOTES.md` template** — currently relies on
  GitHub's auto-generated body. A templated approach (per-stage
  rollup + breaking-change call-out + Cargo.lock dependency delta) is a
  candidate.
- **Per-release fixtures manifest** — would record the SHA-256 of every
  committed fixture at the tag SHA. Not blocking for v0.1.0 (Stage 6 +
  Stage 11a CI gates already pin them); useful for downstream auditors
  who want the manifest in-bundle.
- **`docs-snapshot.tar.gz`** — docs tree at the tag SHA. Redundant for
  operators with `git`; useful for archival hosts.
- **macOS / Windows release binaries** — only if explicit operator
  demand surfaces. POSIX-leaning identity code in
  `crates/omni-net/src/identity.rs` would need cross-platform review
  first.
- **Package managers** (`.deb` / `.rpm` / Homebrew / Snap / Flatpak)
  and **container images** (Docker / Helm) — separate stages with their
  own scopes.

Stage 15.3 should be planned only after the inaugural `v0.1.0` tag cuts
cleanly under Stage 15.2 and verifies on at least one independent
operator host.

---

## 8. Out of scope (locks re-stated)

- No Rust source changes.
- No `Cargo.toml` / `Cargo.lock` / `rust-toolchain.toml` changes.
- No mutation of [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
- No mutation of `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` or any
  registry state.
- No eligibility activation, no `Active` record, no chain RPC change.
- No proof-system / `ModelFormat` / artifact / contributor schema
  change.
- No economics implementation, no staking, no slashing, no rewards.
- No `crates/omni-zkml/src/error.rs` change.
- No new GitHub secrets.
- No new long-lived signing key.
- No edits to the 2026-05-21 or 2026-06-24 RC audits.
- No [`README.md`](../README.md) status-table mutation in this PR.
  After the first `v0.x.y` tag verifies on an independent operator
  host, a follow-up PR may flip the "Stage 10b — Release artifact
  workflow + signing | Planned" row to "Stage 10b — Release artifact
  workflow + cosign keyless signing (Stage 15.2) | Complete".
- No cutting of `v0.1.0` in this PR. The tag is a separate operator
  action.

---

## 9. First signed release — `v0.1.0` (2026-06-25)

Appended by Stage 15.2A after the inaugural tag was cut and verified
end-to-end. **Append only.** §1 – §8 above are untouched.

### Event chain

| Step | Time (UTC) | Evidence |
| --- | --- | --- |
| Tag pushed | 2026-06-25T16:39:* | Tag SHA `820b54ee9dcb10f039bc0748e1b2a8dfa9fb403c` (annotated tag at then-`origin/main` `8bb0be9`) |
| `release.yml` run started | 2026-06-25T16:39:16Z | https://github.com/SUM-INNOVATION/OmniNode-Protocol/actions/runs/28185649617, `event=push`, `headBranch=v0.1.0` |
| Workflow conclusion | shortly after | `success` — all five jobs (`Tag/version match gate`, `Build (default)`, `Build (submit)`, `Bundle + cosign sign`, `Publish draft Release`) reported `pass` |
| Draft Release published | 2026-06-25T19:32:05Z | Pre-promotion URL slug: `…/releases/tag/untagged-96a25bff432ea2001e2f` (GitHub-internal draft slug — documented gotcha; see §10 below) |
| Operator verification on a separate host (per [issue #68](https://github.com/SUM-INNOVATION/OmniNode-Protocol/issues/68)) | 2026-06-25T19:32:* | `cosign verify-blob` → `Verified OK`; `sha256sum -c SHA256SUMS` → all 9 OK; `--version` → `omni-node 0.1.0` on both variants; `--version` snapshots match captured files; `--help` subcommand surfaces sane |
| Cross-verification (cert SAN inspection on another host) | 2026-06-25T19:33:* | `openssl x509 -ext subjectAltName` on `SHA256SUMS.cert` after `base64 -d` yields URI matching the production regex byte-for-byte |
| Promotion to non-draft + `--latest` | 2026-06-25T19:32:11Z | `gh release edit v0.1.0 --draft=false --latest`; post-promotion `gh api repos/.../releases/latest` resolves to `v0.1.0` |

### Fulcio cert facts

| Field | Value |
| --- | --- |
| Subject Alternative Name (URI) | `https://github.com/SUM-INNOVATION/OmniNode-Protocol/.github/workflows/release.yml@refs/tags/v0.1.0` |
| Matches production regex (§4)? | **Yes** — byte-for-byte |
| Issuer | `O=sigstore.dev, CN=sigstore-intermediate` |
| Validity | `2026-06-25T16:42:51Z` → `2026-06-25T16:52:51Z` (10-minute ephemeral, typical for cosign keyless) |
| Serial | `3C617D73778F9FA07E1BBC59F65E6F3CB697C9E7` |

The cert is base64-PEM-wrapped on disk; operators inspecting locally
without cosign installed can run:

```bash
base64 -d < SHA256SUMS.cert > cert.pem
openssl x509 -in cert.pem -noout -ext subjectAltName
# Expected URI line ends in `release.yml@refs/tags/v<X.Y.Z>`
```

### `SHA256SUMS` (verbatim, `v0.1.0`)

```
975a6c412205fdcff440ffb252fcd09464f96353ff2cb99ca6266896c4caba04  omni-node-default-help.txt
ae8a2313f93113d2b1e8f84e0369c3fd86d02002714627a23bcaf30671074f79  omni-node-default-version.txt
b25db4527900c89e6d5719dfcc81eceb524e70c63cf19de049a6d848b9662b62  omni-node-submit-help.txt
ae8a2313f93113d2b1e8f84e0369c3fd86d02002714627a23bcaf30671074f79  omni-node-submit-version.txt
3f0c3f7b3c019b5a2de2013e629952dd15cb6a19246a3277e486184c481af326  omni-node-v0.1.0-x86_64-unknown-linux-gnu-default
2a15d114e1bcd17e23b67859fc7384286b0fe0fe593304d201a37268960f2cab  omni-node-v0.1.0-x86_64-unknown-linux-gnu-submit
```

Note: the two `*-version.txt` files share a SHA-256 because both
`--version` outputs are identical (`omni-node 0.1.0`); the `*-help.txt`
files differ because the `submit` build adds a feature-gated subcommand
surface (chain attestation / submit lifecycle).

### Audit-finding rating change

This event flips the audit finding at
[`docs/phase5-rc-audit-2026-06-24.md`](phase5-rc-audit-2026-06-24.md)
§3b from **Missing** to **Shipped**. **The 2026-06-24 audit doc is not
edited** — its rating is preserved as the historical snapshot of "where
the surface was at Stage 15.1 close." A future dated RC audit will pick
up the new rating; this §9 entry is the durable record in the meantime.
README row 115 is updated in the same Stage 15.2A PR.

### What did *not* go to plan — `v0.1.0` lessons

1. **Pre-promotion URL slug.** Issue #68 §7 listed the *post-promotion*
   release URL (`…/releases/tag/v0.1.0`) as the expected `url` field
   from `gh release view`. Pre-promotion, GitHub serves drafts under a
   `…/releases/tag/untagged-<hex>` slug — the binding to the tag is
   intact via the `tagName` field, but the URL is cosmetically
   different. The operator's agent surfaced the mismatch rather than
   silently continuing (per the §8 hard rule), which was the correct
   behavior. The issue template is corrected by Stage 15.2A. The
   workflow itself is unchanged.

2. **No workflow regressions.** The release pipeline behaved correctly
   end-to-end across `Tag/version match gate` → `Build (default)` →
   `Build (submit)` → `Bundle + cosign sign` → `Publish draft Release`.

### Cross-references

- Operator runbook: [`docs/operator-runbook.md`](operator-runbook.md) §14 (release-readiness checklist, now with first-release worked example).
- Stage 14.8 readiness packet (the closure that precedes Stage 15.x): [`docs/stage14.8-proof-generation-readiness.md`](stage14.8-proof-generation-readiness.md).
- Stage 15.1 RC audit (immutable; rates §3b as Missing at 2026-06-24): [`docs/phase5-rc-audit-2026-06-24.md`](phase5-rc-audit-2026-06-24.md).
- Issue #68 (the runbook the v0.1.0 operator ran): [Cut and verify v0.1.0 signed release](https://github.com/SUM-INNOVATION/OmniNode-Protocol/issues/68).

---

## 10. Operator gotcha — `gh release view` URL field on draft releases

When `gh release view <TAG> --json url` is invoked against a **draft**
release, the returned `url` field carries GitHub's internal slug:

```
https://github.com/<OWNER>/<REPO>/releases/tag/untagged-<hex>
```

After the draft is promoted to non-draft, the `url` flips to the public
shape:

```
https://github.com/<OWNER>/<REPO>/releases/tag/<TAG>
```

The **binding from the git tag to the release is intact in both states**
— `tagName` reads `<TAG>` in both, and `gh release view <TAG>`
successfully resolves to the same release object regardless of state.
Only the URL slug differs.

This is documented GitHub behavior, not a workflow fault. Operators
verifying `v<X.Y.Z>` should:

- Treat the **`isDraft` field** as the load-bearing assertion (it must
  be `true` pre-promotion to confirm the workflow did not promote
  prematurely).
- Treat the `tagName` field as the binding check (it must match the
  intended tag).
- **Not** rely on the `url` field for tag-vs-URL equality before
  promotion.
- Re-query `gh release view <TAG> --json url` post-promotion and confirm
  it now ends in `…/releases/tag/<TAG>` before treating the release as
  publicly resolvable.

