#!/usr/bin/env bash
# Maintainer-run, read-only verification of a signed release bundle (#73).
#
# WHY THIS EXISTS
# ---------------
# A draft release is invisible to a workflow's read-only GITHUB_TOKEN: on the
# v0.1.1 draft, lookup by tag returned 404 and lookup by release or asset ID
# returned 403 (#73; runs 38486439048 and 38490425406). Reading draft assets
# from a workflow would need `contents: write`, which the verification
# workflow does not take. A maintainer with push access can already read the
# draft with their own `gh` login, so this script runs the #73 runbook
# against it without needing anything new:
#
#   §7 before  release state (draft, pre-release or full) and asset digests
#   §4         download every asset; exactly the 11 expected files; bytes
#              equal GitHub's asset digests
#   §5         cosign verify-blob with the runbook regex, the release.yml
#              regex and the exact tag identity bound to the tag's commit;
#              Fulcio SAN
#   §6.1/6.2   sha256sum -c (8 of 8); file -b on all 4 binaries
#   §6.3-6.7   only the binaries this host runs natively (Linux x86_64 or
#              aarch64); every other one is reported NOT RUN
#   §7 after   release state and asset digests unchanged
#
# It calls only `gh release view`, `gh release download` and `gh api` GETs.
# It never edits, uploads to, publishes or promotes a release.
#
# To bind the draft to a hosted run of .github/workflows/verify-release.yml
# (which verifies the same bundle natively on x86_64 and aarch64 runners),
# pass the SHA256SUMS digest that run printed:
#   --expect-sums-sha256 <hex>   fail unless SHA256SUMS has this sha256
#
# Requires: gh (logged in; push access to see a draft), cosign >= 2.4.1, jq,
# openssl, file, and sha256sum or shasum.
#
# Usage:
#   scripts/verify-release-local.sh <tag> [--expect-sums-sha256 <hex>] [--keep]
#
# Exit: 0 = every check this host can run passed (NOT RUN items are listed),
# 1 = a check failed (stop; do not promote), 2 = usage or missing tool.
set -euo pipefail

REPO=SUM-INNOVATION/OmniNode-Protocol
OIDC_ISSUER=https://token.actions.githubusercontent.com
# §5 regex, byte-for-byte from the #73 runbook and
# docs/stage15.2-release-artifact-workflow.md §4.
RUNBOOK_REGEX='https://github\.com/SUM-INNOVATION/OmniNode-Protocol/\.github/workflows/release\.yml@refs/tags/v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$'
# PROD_REGEX, byte-for-byte from release.yml `Verify cosign signature`.
RELEASE_YML_REGEX='https://github\.com/SUM-INNOVATION/OmniNode-Protocol/\.github/workflows/release\.yml@refs/tags/v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9\.-]+)?$'
TAG_RE='^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$'
HEX64_RE='^[0-9a-f]{64}$'
# release.yml captures the --version / --help snapshots on x86_64 only.
SNAPSHOT_TRIPLE=x86_64-unknown-linux-gnu

ok()      { printf '  ok       %s\n' "$*"; }
not_run() { printf '  NOT RUN  %s\n' "$*"; NOT_RUN+=("$*"); }
fail()    { printf '  FAIL     %s\n' "$*" >&2; printf '\nSTOP: do not promote. Comment on #73 with the output above.\n' >&2; exit 1; }
usage()   { sed -n '36,37p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
section() { printf '\n%s\n' "$*"; }

[ $# -ge 1 ] || usage
TAG=$1
shift
EXPECT_SUMS=""
KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --expect-sums-sha256)
            [ $# -ge 2 ] || usage
            EXPECT_SUMS=$(printf '%s' "$2" | tr 'A-F' 'a-f')
            shift 2
            ;;
        --keep) KEEP=1; shift ;;
        *) usage ;;
    esac
done
[[ "$TAG" =~ $TAG_RE ]] || { echo "error: tag '$TAG' is not v<semver>" >&2; exit 2; }
if [ -n "$EXPECT_SUMS" ] && ! [[ "$EXPECT_SUMS" =~ $HEX64_RE ]]; then
    echo "error: --expect-sums-sha256 must be 64 hex digits" >&2
    exit 2
fi

missing=()
for t in gh cosign jq openssl file; do
    command -v "$t" >/dev/null || missing+=("$t")
done
if command -v sha256sum >/dev/null; then
    SHA=(sha256sum)
elif command -v shasum >/dev/null; then
    SHA=(shasum -a 256)
else
    missing+=("sha256sum or shasum")
fi
if [ ${#missing[@]} -gt 0 ]; then
    echo "error: missing on PATH: ${missing[*]}" >&2
    exit 2
fi
COSIGN_VERSION=$(cosign version 2>/dev/null | sed -n 's/^GitVersion: *v\{0,1\}\([0-9][0-9.]*\).*/\1/p')
IFS=. read -r cv_major cv_minor cv_patch <<<"${COSIGN_VERSION:-0.0.0}"
if [ $((cv_major * 1000000 + cv_minor * 1000 + ${cv_patch:-0})) -lt 2004001 ]; then
    echo "error: cosign ${COSIGN_VERSION:-unknown} is older than the required v2.4.1" >&2
    exit 2
fi

NOT_RUN=()
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/omni-verify.XXXXXX")
if [ "$KEEP" -eq 1 ]; then
    trap 'echo "kept: $WORKDIR"' EXIT
else
    trap 'rm -rf "${WORKDIR:?}"' EXIT
fi
DL=$WORKDIR/dl
mkdir -p "$DL"
sha256_of() { "${SHA[@]}" "$1" | awk '{print $1}'; }
STATE_FILTER='{id, tag_name, name, draft, prerelease, created_at, published_at, updated_at, target_commitish, html_url, body, assets: [.assets[] | {id, name, size, digest, state, updated_at}] | sort_by(.name)}'

echo "Read-only verification of $REPO $TAG (#73)"
echo "host: $(uname -sm); cosign $COSIGN_VERSION; $(date -u +%Y-%m-%dT%H:%M:%SZ)"

section "§7 before: release state"
gh release view "$TAG" --repo "$REPO" --json isDraft,isPrerelease,name,tagName,url >"$WORKDIR/view-before.json" ||
    fail "gh release view $TAG: not visible to this login (a draft needs push access)"
jq -e --arg t "$TAG" '.tagName == $t and .name == $t' "$WORKDIR/view-before.json" >/dev/null ||
    fail "expected tagName = name = $TAG: $(cat "$WORKDIR/view-before.json")"
if jq -e '.isDraft' "$WORKDIR/view-before.json" >/dev/null; then
    STATE=draft
elif jq -e '.isPrerelease' "$WORKDIR/view-before.json" >/dev/null; then
    STATE=pre-release
else
    STATE="full release (already promoted: this is post-publication evidence, not a gate)"
fi
IDS=$(gh api --paginate "repos/$REPO/releases?per_page=100" | jq -rs --arg t "$TAG" '[.[][] | select(.tag_name == $t) | .id] | join(" ")')
read -r -a IDA <<<"$IDS"
[ ${#IDA[@]} -eq 1 ] || fail "expected exactly one release for $TAG, found: '${IDS}'"
RELEASE_ID=${IDA[0]}
gh api "repos/$REPO/releases/$RELEASE_ID" | jq "$STATE_FILTER" >"$WORKDIR/state-before.json"
ok "release $RELEASE_ID is a $STATE; $(jq -r .url "$WORKDIR/view-before.json")"

section "§4 download every asset; exactly the 11 expected files"
gh release download "$TAG" --repo "$REPO" --pattern '*' --dir "$DL" >/dev/null
EXPECTED=(
    "omni-node-$TAG-x86_64-unknown-linux-gnu-default"
    "omni-node-$TAG-x86_64-unknown-linux-gnu-submit"
    "omni-node-$TAG-aarch64-unknown-linux-gnu-default"
    "omni-node-$TAG-aarch64-unknown-linux-gnu-submit"
    "omni-node-default-version.txt"
    "omni-node-submit-version.txt"
    "omni-node-default-help.txt"
    "omni-node-submit-help.txt"
    "SHA256SUMS"
    "SHA256SUMS.sig"
    "SHA256SUMS.cert"
)
for f in "${EXPECTED[@]}"; do
    [ -f "$DL/$f" ] || fail "MISSING: $f"
done
n=$(find "$DL" -maxdepth 1 -type f | wc -l | tr -d ' ')
[ "$n" -eq 11 ] || fail "expected exactly 11 files, found $n: $(cd "$DL" && find . -maxdepth 1 -type f | sort | tr '\n' ' ')"
ok "all 11 expected files present, and no others"
jq -e '[.assets[] | .digest | type == "string" and startswith("sha256:")] | all and length == 11' \
    "$WORKDIR/state-before.json" >/dev/null || fail "GitHub does not report a sha256 digest for every asset"
jq -r '.assets[] | "\(.digest | sub("^sha256:"; ""))  \(.name)"' "$WORKDIR/state-before.json" >"$WORKDIR/api-digests"
(cd "$DL" && "${SHA[@]}" -c "$WORKDIR/api-digests" >/dev/null) || fail "downloaded bytes differ from GitHub's asset digests"
ok "every downloaded file matches GitHub's asset digest"
SUMS_SHA256=$(sha256_of "$DL/SHA256SUMS")
if [ -n "$EXPECT_SUMS" ]; then
    [ "$SUMS_SHA256" = "$EXPECT_SUMS" ] ||
        fail "SHA256SUMS sha256 $SUMS_SHA256 is not the expected $EXPECT_SUMS"
    ok "SHA256SUMS sha256 equals the expected digest $EXPECT_SUMS"
fi

section "§5 cosign verify-blob + Fulcio SAN"
REF=$(gh api "repos/$REPO/git/ref/tags/$TAG")
OBJ_TYPE=$(jq -r .object.type <<<"$REF")
OBJ_SHA=$(jq -r .object.sha <<<"$REF")
if [ "$OBJ_TYPE" = tag ]; then
    COMMIT=$(gh api "repos/$REPO/git/tags/$OBJ_SHA" --jq .object.sha)
else
    COMMIT=$OBJ_SHA
fi
cosign_verify() {
    cosign verify-blob --certificate "$DL/SHA256SUMS.cert" --signature "$DL/SHA256SUMS.sig" \
        --certificate-oidc-issuer "$OIDC_ISSUER" "$@" "$DL/SHA256SUMS" >"$WORKDIR/cosign.out" 2>&1
}
cosign_verify --certificate-identity-regexp "$RUNBOOK_REGEX" ||
    fail "cosign (runbook regex): $(cat "$WORKDIR/cosign.out")"
ok "cosign verify-blob, runbook §5 regex: $(tail -n 1 "$WORKDIR/cosign.out")"
cosign_verify --certificate-identity-regexp "$RELEASE_YML_REGEX" ||
    fail "cosign (release.yml regex): $(cat "$WORKDIR/cosign.out")"
ok "cosign verify-blob, release.yml PROD_REGEX: $(tail -n 1 "$WORKDIR/cosign.out")"
cosign_verify \
    --certificate-identity "https://github.com/$REPO/.github/workflows/release.yml@refs/tags/$TAG" \
    --certificate-github-workflow-repository "$REPO" \
    --certificate-github-workflow-ref "refs/tags/$TAG" \
    --certificate-github-workflow-trigger push \
    --certificate-github-workflow-sha "$COMMIT" ||
    fail "cosign (exact identity, commit $COMMIT): $(cat "$WORKDIR/cosign.out")"
ok "cosign verify-blob, exact identity bound to $TAG -> $COMMIT"
tr -d '\r\n' <"$DL/SHA256SUMS.cert" | openssl base64 -d -A >"$WORKDIR/cert.pem"
SAN=$(openssl x509 -in "$WORKDIR/cert.pem" -noout -text |
    sed -n '/Subject Alternative Name/{n;p;}' | sed 's/^ *//')
WANT_SAN="URI:https://github.com/$REPO/.github/workflows/release.yml@refs/tags/$TAG"
[ "$SAN" = "$WANT_SAN" ] || fail "SAN is '$SAN', want '$WANT_SAN'"
ok "SAN $SAN"

section "§6.1 sha256sum -c SHA256SUMS"
(cd "$DL" && "${SHA[@]}" -c SHA256SUMS) >"$WORKDIR/sha256-c.txt" || {
    cat "$WORKDIR/sha256-c.txt" >&2
    fail "sha256 -c SHA256SUMS"
}
total=$(wc -l <"$DL/SHA256SUMS" | tr -d ' ')
passed=$(grep -c ': OK$' "$WORKDIR/sha256-c.txt" || true)
[ "$passed" -eq "$total" ] && [ "$total" -eq 8 ] || fail "expected 8 of 8 OK, got $passed of $total"
ok "$passed of $total OK"

section "§6.2 file -b on all 4 binaries"
for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
    case "$t" in x86_64*) re="x86-64" ;; aarch64*) re="ARM aarch64" ;; esac
    for v in default submit; do
        b="omni-node-$TAG-$t-$v"
        out=$(file -b "$DL/$b")
        grep -qE "^ELF 64-bit LSB .*$re, .*GNU/Linux" <<<"$out" || fail "wrong ELF for $b: $out"
        ok "$b: $out"
    done
done

section "§6.3-6.7 native --version / --help"
case "$(uname -s)/$(uname -m)" in
    Linux/x86_64) NATIVE=x86_64-unknown-linux-gnu ;;
    Linux/aarch64 | Linux/arm64) NATIVE=aarch64-unknown-linux-gnu ;;
    *) NATIVE="" ;;
esac
# The binaries under test never see a GitHub token.
run() { (cd "$DL" && env -u GH_TOKEN -u GITHUB_TOKEN "$@"); }
for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
    if [ "$t" != "$NATIVE" ]; then
        not_run "§6 --version/--help for $t: this $(uname -sm) host cannot run it natively"
        continue
    fi
    for v in default submit; do
        b="omni-node-$TAG-$t-$v"
        chmod +x "$DL/$b"
        got=$(run "./$b" --version)
        [ "$got" = "omni-node ${TAG#v}" ] || fail "$b --version printed '$got'"
        diff <(run "./$b" --version) "$DL/omni-node-$v-version.txt" >/dev/null ||
            fail "$b --version differs from omni-node-$v-version.txt"
        ok "$b --version: $got (matches omni-node-$v-version.txt)"
        # clap's Usage line embeds argv[0]'s basename; the snapshot was captured
        # from the x86_64 binary, so run it under that name.
        capname="omni-node-$TAG-$SNAPSHOT_TRIPLE-$v"
        diff <(run bash -c 'exec -a "$1" "./$2" --help' _ "$capname" "$b") \
            "$DL/omni-node-$v-help.txt" >/dev/null || fail "$b --help differs from omni-node-$v-help.txt"
        ok "$b --help matches omni-node-$v-help.txt"
    done
done

section "§7 after: release state unchanged"
gh release view "$TAG" --repo "$REPO" --json isDraft,isPrerelease,name,tagName,url >"$WORKDIR/view-after.json"
gh api "repos/$REPO/releases/$RELEASE_ID" | jq "$STATE_FILTER" >"$WORKDIR/state-after.json"
diff "$WORKDIR/view-before.json" "$WORKDIR/view-after.json" >/dev/null || fail "release view changed during verification"
diff "$WORKDIR/state-before.json" "$WORKDIR/state-after.json" >/dev/null ||
    fail "release metadata or asset digests changed during verification"
ok "release metadata and every asset digest identical before and after"

cat <<EOF

=== Evidence (paste into #73 §9; add the verifier handle and host) ===
Tag:                    $TAG
Tag commit / object:    $COMMIT / $OBJ_TYPE $OBJ_SHA
Release:                $RELEASE_ID, $STATE
Release URL:            $(jq -r .url "$WORKDIR/view-before.json")
Verifier host:          $(uname -sm)
Verification (UTC):     $(date -u +%Y-%m-%dT%H:%M:%SZ)
cosign:                 $COSIGN_VERSION; Verified OK (runbook regex, release.yml regex, exact identity)
Fulcio SAN:             $SAN
SHA256SUMS sha256:      $SUMS_SHA256
sha256 -c:              $passed of $total OK

SHA256SUMS:
$(cat "$DL/SHA256SUMS")
EOF
if [ ${#NOT_RUN[@]} -gt 0 ]; then
    printf '\nNOT RUN on this host (needs a native Linux host of that arch, or the hosted workflow):\n'
    printf '  %s\n' "${NOT_RUN[@]}"
fi
printf '\nResult: every check this host can run passed. Promotion is an owner decision (#73 §8).\n'
