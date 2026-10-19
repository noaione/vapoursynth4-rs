#!/usr/bin/env bash
# Verify `vapoursynth4-sys/src/vulkan.rs` against the C compiler.
#
# VSVulkan4.h hands a filter the core's Vulkan device, so the generated bindings
# are only correct if their `#[repr(C)]` layout is byte for byte what the C
# compiler produces from the same header. This compiles two probes that print
# `sizeof`, `_Alignof` and `offsetof` for every aggregate in the generated set
# and diffs their output.
#
# It also proves the checked-in module is current: the header is regenerated
# into a scratch directory and compared, so a stale binding fails the check
# rather than silently disagreeing with the file it claims to describe.
#
# Usage: scripts/check-vulkan-layout.sh [vulkan-include-dir] [vs-include-dir]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VULKAN_INC="${1:-${VULKAN_INC:-/usr/local/include}}"
VS_INC="${2:-${VS_INC:-}}"

if [ -z "$VS_INC" ]; then
    VS_INC="$(
        python3 -c 'import vapoursynth, os; print(os.path.dirname(vapoursynth.__file__))' 2>/dev/null || true
    )/include"
fi

for header in "$VULKAN_INC/vulkan/vulkan_core.h" "$VS_INC/VSVulkan4.h"; do
    if [ ! -f "$header" ]; then
        echo "missing header: $header" >&2
        echo "pass the include directories explicitly if they live elsewhere" >&2
        exit 1
    fi
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

check="vapoursynth4-sys/src/vulkan.rs"

# Regenerate into the scratch dir, so a stale checked-in file is caught.
python3 "$ROOT/scripts/gen-vulkan-bindings.py" \
    "$VULKAN_INC/vulkan/vulkan_core.h" "$VS_INC/VSVulkan4.h" >"$WORK/raw.rs"

# The generator emits plain text and the checked-in file is formatted, so the
# comparison has to be against formatted output; otherwise running `cargo fmt`
# alone would fail this check.
cp "$WORK/raw.rs" "$WORK/vulkan.rs"
rustfmt --edition 2024 "$WORK/vulkan.rs"

if ! diff -q "$WORK/vulkan.rs" "$ROOT/$check" >/dev/null 2>&1; then
    echo "FAIL: $check is out of date; regenerate it with scripts/gen-vulkan-bindings.py and run cargo fmt" >&2
    diff -u "$ROOT/$check" "$WORK/vulkan.rs" | head -60 >&2
    exit 1
fi
python3 "$ROOT/scripts/gen-vulkan-bindings.py" --layout "$WORK" \
    "$VULKAN_INC/vulkan/vulkan_core.h" "$VS_INC/VSVulkan4.h" "$WORK/raw.rs" >/dev/null

cc -std=c11 -I"$VULKAN_INC" -I"$VS_INC" -o "$WORK/layout_c" "$WORK/layout.c"
rustc --edition 2024 -O -o "$WORK/layout_rs" "$WORK/layout.rs"

"$WORK/layout_c" >"$WORK/c.txt"
"$WORK/layout_rs" >"$WORK/rs.txt"

if diff -u "$WORK/c.txt" "$WORK/rs.txt"; then
    printf 'OK: %s declarations match the C compiler\n' "$(wc -l <"$WORK/c.txt" | tr -d ' ')"
else
    echo "FAIL: generated Vulkan layout differs from C" >&2
    exit 1
fi
