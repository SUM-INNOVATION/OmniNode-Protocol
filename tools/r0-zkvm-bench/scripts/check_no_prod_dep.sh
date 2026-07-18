#!/usr/bin/env bash
# Dependency-isolation check for the R0 research crate (OmniNode #101).
#
# Asserts that NO production crate in the OmniNode workspace depends on
# `r0-zkvm-bench`, using `cargo metadata` on the workspace. Complements the
# in-crate `tests/dependency_isolation.rs`. Exit non-zero on any violation.
set -euo pipefail

CRATE="r0-zkvm-bench"
# Repo root = two levels up from this script's crate dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "==> Repo root: ${REPO_ROOT}"
echo "==> Checking that no workspace member depends on '${CRATE}'..."

# 1. The workspace must not even resolve the crate as a package/node.
if cargo metadata --manifest-path "${REPO_ROOT}/Cargo.toml" --format-version 1 \
     | grep -q "\"name\":\"${CRATE}\""; then
  echo "FAIL: '${CRATE}' appears in the workspace cargo metadata graph."
  exit 1
fi

# 2. No member manifest may reference the crate by name.
if grep -RIl --include=Cargo.toml -e "${CRATE}" -e "r0_zkvm_bench" \
     "${REPO_ROOT}/crates" 2>/dev/null | grep -q .; then
  echo "FAIL: a production crate manifest references '${CRATE}'."
  grep -RIn --include=Cargo.toml -e "${CRATE}" -e "r0_zkvm_bench" "${REPO_ROOT}/crates" || true
  exit 1
fi

# 3. The root manifest must exclude tools/.
if ! grep -q 'exclude = \["tools"\]' "${REPO_ROOT}/Cargo.toml"; then
  echo "FAIL: root Cargo.toml does not exclude tools/."
  exit 1
fi

echo "OK: '${CRATE}' is fully isolated — no production crate depends on it."
