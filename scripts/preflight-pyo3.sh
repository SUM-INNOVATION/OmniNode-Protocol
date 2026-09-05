#!/usr/bin/env bash
# Preflight for the PyO3 crate `omni-bridge`.
#
# WHY THIS EXISTS
# ---------------
# `omni-bridge` links against libpython. Where the development library is
# missing, the build fails at LINK time with a wall of undefined symbols:
#
#   Undefined symbols for architecture arm64:
#     "_PyBaseObject_Type", referenced from: ...
#     "_PyBytes_AsString",  referenced from: ...
#
# That failure names Python symbols but not the missing prerequisite, so it reads
# like a code error. It is not: it is an unsatisfied environment requirement, and
# it reproduces on a pristine checkout with no local changes.
#
# The historical response was to exclude the crate -- locally AND in CI -- which
# meant `omni-bridge` was tested nowhere. CI now installs Python and gates the
# crate; this script gives a developer the same check before they build.
#
# Usage:
#   scripts/preflight-pyo3.sh          # check only
#   scripts/preflight-pyo3.sh --test   # check, then run the crate's tests
set -euo pipefail

ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*" >&2; exit 1; }

echo "PyO3 preflight (omni-bridge):"

command -v python3 >/dev/null || fail "python3 not on PATH"
ok "python3 $(python3 --version 2>&1 | awk '{print $2}')"

# PyO3 needs a shared libpython to link against. A python built --without-shared
# (common in some pyenv installs) has no libpython*.so/.dylib and will fail at
# link time even though the interpreter runs fine.
read -r LIBDIR LDLIBRARY SHARED <<PY
$(python3 - <<'PYEOF'
import sysconfig
g = sysconfig.get_config_var
print(g('LIBDIR') or '-', g('LDLIBRARY') or '-', g('Py_ENABLE_SHARED') or '0')
PYEOF
)
PY

echo "  libdir=${LIBDIR}  lib=${LDLIBRARY}  shared=${SHARED}"

if [ "$SHARED" != "1" ]; then
  fail "this Python was built WITHOUT a shared library (Py_ENABLE_SHARED=0).
        PyO3 cannot link against it. Install a shared build, e.g.
          pyenv:   PYTHON_CONFIGURE_OPTS=\"--enable-shared\" pyenv install <ver>
          macOS:   brew install python@3.12
          debian:  apt-get install python3-dev"
fi
ok "shared libpython available"

if [ -n "${LIBDIR:-}" ] && [ -d "$LIBDIR" ]; then
  if ls "$LIBDIR"/${LDLIBRARY} >/dev/null 2>&1 || ls "$LIBDIR"/libpython*.{so,dylib} >/dev/null 2>&1; then
    ok "found libpython in $LIBDIR"
  else
    warn "no libpython file located in $LIBDIR — link may still fail"
  fi
fi

# PYO3_PYTHON pins which interpreter pyo3 builds against; report it so a
# mismatch between `python3` on PATH and the one pyo3 uses is visible.
if [ -n "${PYO3_PYTHON:-}" ]; then
  ok "PYO3_PYTHON=$PYO3_PYTHON (overrides PATH python3)"
else
  ok "PYO3_PYTHON unset — pyo3 will use python3 from PATH"
fi

echo "preflight passed."

if [ "${1:-}" = "--test" ]; then
  echo
  echo "running omni-bridge tests:"
  cargo test -p omni-bridge
fi
