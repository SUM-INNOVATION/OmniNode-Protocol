#!/usr/bin/env bash
# verify-release-signature.sh — the #73 runbook §5 for one downloaded
# release bundle: cosign verify-blob of SHA256SUMS against
#   1. the runbook §5 identity regex,
#   2. release.yml's PROD_REGEX, and
#   3. the exact identity bound to the commit the tag points at,
# then the Fulcio SAN check. Read-only.
#
# Used by .github/workflows/verify-release.yml: the `verify` job (the
# release under verification) and the `signature-selftest` job (a
# published release, run on pull requests that touch this tooling).
#
# Usage: scripts/verify-release-signature.sh <bundle-dir>
#   <bundle-dir> holds SHA256SUMS, SHA256SUMS.sig and SHA256SUMS.cert.
# Environment: REPO, TAG, OIDC_ISSUER, RUNBOOK_REGEX, RELEASE_YML_REGEX,
#   and GH_TOKEN (contents: read) for resolving the tag.
# Requires: cosign, gh, jq, openssl, base64. Exits non-zero on any failure.

set -euo pipefail

dir=${1:?usage: verify-release-signature.sh <bundle-dir>}
: "${REPO:?}" "${TAG:?}" "${OIDC_ISSUER:?}" "${RUNBOOK_REGEX:?}" "${RELEASE_YML_REGEX:?}"
cd "$dir"

cosign version
echo "--- runbook §5 regex ---"
printf '%s\n' "$RUNBOOK_REGEX"
cosign verify-blob \
    --certificate SHA256SUMS.cert \
    --signature SHA256SUMS.sig \
    --certificate-identity-regexp "$RUNBOOK_REGEX" \
    --certificate-oidc-issuer "$OIDC_ISSUER" \
    SHA256SUMS
echo "--- release.yml PROD_REGEX ---"
printf '%s\n' "$RELEASE_YML_REGEX"
cosign verify-blob \
    --certificate SHA256SUMS.cert \
    --signature SHA256SUMS.sig \
    --certificate-identity-regexp "$RELEASE_YML_REGEX" \
    --certificate-oidc-issuer "$OIDC_ISSUER" \
    SHA256SUMS
echo "--- extra: bind cert to the commit the tag points at ---"
REF=$(gh api "repos/$REPO/git/ref/tags/$TAG")
OBJ_TYPE=$(jq -r .object.type <<<"$REF")
OBJ_SHA=$(jq -r .object.sha <<<"$REF")
if [[ "$OBJ_TYPE" == "tag" ]]; then
    COMMIT=$(gh api "repos/$REPO/git/tags/$OBJ_SHA" --jq .object.sha)
else
    COMMIT="$OBJ_SHA"
fi
echo "tag object: $OBJ_TYPE $OBJ_SHA -> commit $COMMIT"
cosign verify-blob \
    --certificate SHA256SUMS.cert \
    --signature SHA256SUMS.sig \
    --certificate-identity "https://github.com/$REPO/.github/workflows/release.yml@refs/tags/$TAG" \
    --certificate-oidc-issuer "$OIDC_ISSUER" \
    --certificate-github-workflow-repository "$REPO" \
    --certificate-github-workflow-ref "refs/tags/$TAG" \
    --certificate-github-workflow-trigger push \
    --certificate-github-workflow-sha "$COMMIT" \
    SHA256SUMS
echo "--- runbook §5 SAN extraction ---"
base64 -d <SHA256SUMS.cert >cert.pem
openssl x509 -in cert.pem -noout -ext subjectAltName | tee san.txt
openssl x509 -in cert.pem -noout -subject -issuer -startdate -enddate -serial
openssl x509 -in cert.pem -noout -text | sed -n '/X509v3 extensions/,/Signature Algorithm/p'
WANT="URI:https://github.com/$REPO/.github/workflows/release.yml@refs/tags/$TAG"
if ! grep -qxF "    $WANT" san.txt && ! grep -qxF "$WANT" san.txt; then
    echo "::error::SAN does not equal $WANT"
    exit 1
fi
echo "SAN OK: $WANT"
echo "§5 result: cosign verify-blob succeeded for all three identity checks; SAN matches."
