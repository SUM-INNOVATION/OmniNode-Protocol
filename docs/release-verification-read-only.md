# Read-only release verification (#73)

How to verify a signed release bundle when the release is still a
draft, without giving any workflow a write-capable token. Applies to
`v0.1.1` (#73) and to later tags.

Tools:

- [`.github/workflows/verify-release.yml`](../.github/workflows/verify-release.yml):
  hosted, native x86_64 and aarch64 runners.
- [`scripts/verify-release-local.sh`](../scripts/verify-release-local.sh):
  maintainer-run, any host.

Both run the #73 runbook's §4–§7 checks, and neither ever edits,
uploads to, publishes or promotes a release.

## 1. Why a workflow cannot read the draft

A draft release is visible only to accounts with push access. For a
workflow's `GITHUB_TOKEN`, that means `contents: write`. With
`contents: read`, the v0.1.1 draft was invisible:

| Request (read-only `GITHUB_TOKEN`) | Result | Run |
| --- | --- | --- |
| `GET /releases/tags/v0.1.1` | 404 | 38486439048 |
| `GET /releases` | 200, draft omitted | 38486439048 |
| `GET /releases/345106514` | 403 "Resource not accessible by integration" | 38490425406 |
| `GET /releases/assets/<id>` (JSON and octet-stream) | 403 "Resource not accessible by integration" | 38490425406 |

The 403 responses carried `x-accepted-github-permissions: contents=read`,
but the token was still refused. Draft visibility is decided by push
access, not by that header.

## 2. Options and the permission each needs

| # | Source of the bytes | Permission | Owner decision | What it covers | v0.1.1 |
| --- | --- | --- | --- | --- | --- |
| A | The draft, read by `scripts/verify-release-local.sh` with the maintainer's own `gh` login | None new. Uses the push access the maintainer already has. | None: read-only | §4, §5, §6.1, §6.2 and §7 on any host. §6.3–§6.7 only for a Linux arch the host runs natively. | Usable now. On a macOS host, both arches' `--version`/`--help` checks are NOT RUN. |
| B | `release-bundle` artifact of the release.yml run that built the tag (the exact files the `Publish draft Release` job uploads). Workflow input `source=run-artifacts` | `actions: read` and `contents: read` on the verify job's `GITHUB_TOKEN` | None for the token. Running it needs the workflow on `main` (dispatch). | §4–§6 natively on x86_64 and aarch64. Bound to the draft by running A with `--expect-sums-sha256`. | Closed: the artifacts of run 28214844938 expired (14-day retention). Usable for a future tag within 14 days of its release run. |
| C | A published pre-release. Workflow input `source=release`, or the `release: published` trigger | `contents: read` on the verify job's `GITHUB_TOKEN` | Publish before native verification, as a pre-release. This amends #73 §8 ("do not promote before verification"). | §4–§6 natively on both arches, plus §7 unchanged before and after | Usable once the owner approves publishing v0.1.1 as a pre-release. |
| D | The draft, read by the hosted workflow | `contents: write` on the verify job's `GITHUB_TOKEN`, used for GET requests only | Authorize a broader token | Everything, with no publication | Not implemented. The workflow refuses by design. |

Trigger notes:

- `release: published` and `workflow_dispatch` run only once
  `verify-release.yml` is on the default branch. Only those two events
  run the full `verify` job.
- A release published with a maintainer's credentials triggers
  `release: published`. A release published by a workflow's
  `GITHUB_TOKEN` would not trigger it.
- **Pull requests that touch this tooling run only `signature-selftest`.**
  - It runs the same §5 script (`scripts/verify-release-signature.sh`),
    with the same pinned cosign, against `v0.1.0`, an already-published
    full release.
  - A negative control must then reject an altered `SHA256SUMS`.
  - It shows the signature check executes and passes on a real published
    release. It says nothing about the `v0.1.1` draft: its signature is
    verified only by a run of option A, B or C on its own bytes.

## 3. Sequences

### v0.1.1 (draft exists; its run artifacts have expired)

1. **A, now.** A maintainer runs:

   ```bash
   scripts/verify-release-local.sh v0.1.1 \
     --expect-sums-sha256 f24da5b080ca9301b31e3e9156e9036d67385675b0f5f02d4478498059072092
   ```

   That digest is GitHub's own digest of the draft's `SHA256SUMS`
   asset.

2. **The owner chooses C or D.** Native `--version`/`--help` on both
   arches needs Linux x86_64 and aarch64 hosts. Only the hosted runners
   provide those.

3. **If C:**

   ```bash
   gh release edit v0.1.1 --repo SUM-INNOVATION/OmniNode-Protocol --draft=false --prerelease
   ```

   Then the hosted run executes:
   - automatically on `release: published` once on `main`;
   - otherwise via dispatch with `source=release`,
     `expect_sums_sha256=f24da5b0…`.

   Every job must pass. Only then:

   ```bash
   gh release edit v0.1.1 --repo SUM-INNOVATION/OmniNode-Protocol --prerelease=false --latest
   ```

   If anything fails, leave it as a pre-release, do not promote, and
   prefer re-cutting `v0.1.2` (#73 §10).

### Later tags (verification before the draft is used)

1. Within 14 days of the tag's release.yml run, dispatch this workflow
   with `source=run-artifacts` and `run_id=<release.yml run>`.
2. The run checks three things before it touches the bundle:
   - the run is a successful release.yml `push` run for the tag;
   - it was built from the commit the tag points at;
   - its `release-bundle` artifact has not expired.

   The download step itself rejects an artifact whose digest does not
   match.
3. The job summary prints `SHA256SUMS sha256: <D>`. A maintainer then
   runs:

   ```bash
   scripts/verify-release-local.sh <tag> --expect-sums-sha256 <D>
   ```

   This proves the draft holds the bytes that were verified natively.
4. Promotion stays the owner's §8 decision.

## 4. What each tool checks

| Runbook step | `verify-release.yml` | `verify-release-local.sh` |
| --- | --- | --- |
| §7 before | `source=release`: published, `tagName = name = <tag>`, pre-release (gate) or full (evidence only) | draft, pre-release or full; `tagName = name = <tag>` |
| §4 | 11 expected files and no others; bytes equal GitHub's asset digests (`source=release`) or the artifact digest (`source=run-artifacts`); optional `expect_sums_sha256` | 11 expected files and no others; bytes equal GitHub's asset digests; optional `--expect-sums-sha256` |
| §5 | runbook regex, release.yml regex, exact identity bound to the tag's commit; SAN (`scripts/verify-release-signature.sh`) | the same checks, inline; needs a real `cosign` ≥ 2.4.1 on the host |
| §6.1, §6.2 | `sha256sum -c` 8 of 8; `file -b` on all 4 binaries | the same (`shasum -a 256` where `sha256sum` is missing) |
| §6.3–§6.7 | native on both runners; snapshot diffs | native arch only; every other arch reported NOT RUN |
| §7 after | release metadata and asset digests unchanged (`source=release`) | the same |
