#!/usr/bin/env python3
"""Generate the `vulkan` module of `vapoursynth4-sys` from the Khronos header.

`VSVulkan4.h` includes `<vulkan/vulkan_core.h>` and exposes the core's Vulkan
device to a filter: `VSVulkanFunctions` is a table of `PFN_vk*` members, and the
`VSVULKANAPI` entry points take `VkBuffer`, `VkMemoryRequirements` and friends.
Rust therefore needs those Vulkan declarations, but only the ones actually
reachable from `VSVulkan4.h`.

This script computes that reachable set and emits it. It is deliberately *not* a
build script: the output is checked in, so building the crate needs neither a
Vulkan SDK nor this script. Run it by hand when bumping the VapourSynth version:

    scripts/gen-vulkan-bindings.py \
        include/vulkan/vulkan_core.h include/VSVulkan4.h \
        > vapoursynth4-sys/src/vulkan.rs

The emitted layout is verified against the C compiler by
`scripts/check-vulkan-layout.sh`, which compiles a C harness and a Rust harness
and compares `sizeof`, `_Alignof` and `offsetof` for every type.
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict

# MARK: C scalar types -> Rust

SCALARS = {
    "void": "c_void",
    "char": "c_char",
    "signed char": "c_schar",
    "unsigned char": "c_uchar",
    "short": "c_short",
    "unsigned short": "c_ushort",
    "int": "c_int",
    "unsigned int": "c_uint",
    "unsigned": "c_uint",
    "long": "c_long",
    "unsigned long": "c_ulong",
    "long long": "c_longlong",
    "unsigned long long": "c_ulonglong",
    "float": "c_float",
    "double": "c_double",
    "int8_t": "i8",
    "uint8_t": "u8",
    "int16_t": "i16",
    "uint16_t": "u16",
    "int32_t": "i32",
    "uint32_t": "u32",
    "int64_t": "i64",
    "uint64_t": "u64",
    "size_t": "usize",
    "intptr_t": "isize",
    "uintptr_t": "usize",
    "VkBool32": "u32",
}

# Vulkan field and parameter names that collide with Rust keywords. `type` is
# the only one the headers actually use, but the whole set is cheap to carry.
RUST_KEYWORDS = frozenset("""
    type match ref in as box move use where if for loop break continue return
    self Self crate super fn let const static struct enum trait impl mod pub
    unsafe extern dyn async await try yield abstract become do final macro
    override priv typeof unsized virtual
""".split())


def rid(name: str) -> str:
    """Escape a C identifier that is a Rust keyword."""
    return f"r#{name}" if name in RUST_KEYWORDS else name


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def balanced_braces(text: str, start: int) -> int:
    """Index just past the `}` matching the `{` at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


# MARK: Declaration

class Decl:
    def __init__(self, kind, name, body="", text="", deps=None):
        self.kind = kind  # struct | union | enum | alias | pfn | handle_disp | handle_nd
        self.name = name
        self.body = body
        self.text = text
        self.deps = deps if deps is not None else set()


TYPE_RE = re.compile(r"\b(Vk[A-Za-z0-9_]+|PFN_vk\w+)\b")


def index_header(text: str) -> "OrderedDict[str, Decl]":
    decls: "OrderedDict[str, Decl]" = OrderedDict()
    i = 0
    n = len(text)
    while i < n:
        m = re.compile(r"\btypedef\b").search(text, i)
        if not m:
            break
        start = m.start()
        # find the terminating ';' at brace depth 0
        depth = 0
        j = m.end()
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            elif text[j] == ";" and depth == 0:
                break
            j += 1
        blob = text[start : j + 1]
        i = j + 1
        if "{" in blob:
            head = blob[: blob.index("{")]
            close = balanced_braces(blob, blob.index("{"))
            tail = blob[close:]
            names = re.findall(r"\b(\w+)\s*;", tail)
            if not names:
                continue
            name = names[-1]
            inner = blob[blob.index("{") + 1 : close - 1]
            kind = "struct"
            if re.search(r"\benum\b", head):
                kind = "enum"
            elif re.search(r"\bunion\b", head):
                kind = "union"
            decls[name] = Decl(kind, name, inner, blob, set(TYPE_RE.findall(inner)))
        else:
            body = blob[len("typedef") : -1].strip()
            pm = re.match(
                r"^(.+?)\(\s*VKAPI_PTR\s*\*\s*(PFN_vk\w+)\s*\)\s*\((.*)\)$", body, re.S
            )
            if pm:
                name = pm.group(2)
                decls[name] = Decl("pfn", name, pm.group(3), blob, set(TYPE_RE.findall(body)))
                continue
            name = re.findall(r"(\w+)$", body)
            if not name:
                continue
            name = name[0]
            base = body[: body.rfind(name)].strip()
            deps = set(TYPE_RE.findall(base))
            decls[name] = Decl("alias", name, base, blob, deps)
    for m in re.finditer(r"VK_DEFINE_HANDLE\((\w+)\)", text):
        decls.setdefault(m.group(1), Decl("handle_disp", m.group(1), "", m.group(0)))
    for m in re.finditer(r"VK_DEFINE_NON_DISPATCHABLE_HANDLE\((\w+)\)", text):
        decls.setdefault(m.group(1), Decl("handle_nd", m.group(1), "", m.group(0)))
    return decls


# MARK: closure


def compute_closure(core_text, vs_text, decls):
    fns = re.findall(r"FN\(VS_VK_\w+,\s*VS_VK_\w+,\s*(\w+)\)", vs_text)
    seed = set(re.findall(r"\b(Vk[A-Za-z0-9_]+)\b", vs_text))
    seed = {s for s in seed if not s.startswith(("VSVulkan", "VSGPU", "VS_"))}
    seed |= {f"PFN_vk{f}" for f in fns}
    # `VSVulkanCoreHandles` names `PFN_vkGetInstanceProcAddr` directly, and it is
    # not one of the listed entry points, so take every PFN the header mentions.
    seed |= set(re.findall(r"\b(PFN_vk\w+)\b", vs_text))
    # Callbacks the header only names indirectly.
    seed |= {"PFN_vkAllocationFunction", "PFN_vkReallocationFunction", "PFN_vkFreeFunction"}

    seen, stack = set(), list(seed)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        d = decls.get(name)
        if d:
            stack.extend(x for x in d.deps if x not in seen)
    missing = sorted(x for x in seen if x not in decls)
    # Names the closure reaches but the header never declares as a typedef: they
    # come from the `PFN_vkVoidFunction` family or are macros, so drop them.
    return seen, missing, fns


# MARK: emission

# `VkResult` etc. are C enums; anything the header defines with `enum` becomes a
# `c_int` alias plus constants, which stays valid for values a newer driver may
# report. Values in a `typedef enum` are i32 in the C ABI.
ENUM_BASE = "c_int"


def c_type_to_rust(ctype: str) -> str:
    ctype = ctype.strip()
    ctype = re.sub(r"\s+", " ", ctype)
    const = False
    ctype = ctype.replace("const", " ")
    ctype = re.sub(r"\s+", " ", ctype).strip()
    pointers = 0
    while ctype.endswith("*"):
        pointers += 1
        ctype = ctype[:-1].strip()
    ctype = ctype.strip()
    if ctype.startswith("struct ") or ctype.startswith("union ") or ctype.startswith("enum "):
        ctype = ctype.split(" ", 1)[1].strip()
    if ctype in SCALARS:
        base = SCALARS[ctype]
    elif ctype in ("VkBool32",):
        base = "u32"
    else:
        base = ctype
    out = base
    # Pointer const-ness follows the C: a trailing `const` before the `*` means
    # the pointee is const. We re-derive it from the original spelling.
    for k in range(pointers):
        out = f"*mut {out}"
    return out


FIELD_RE = re.compile(
    r"^\s*(?P<type>.*?)\s*(?P<stars>\**)\s*(?P<name>\w+)"
    r"(?P<arr>(\s*\[[^\]]+\])*)\s*$",
    re.S,
)


def split_fields(body: str):
    """Yield top-level `;`-terminated members, keeping nested braces intact."""
    depth = 0
    cur = []
    for ch in body:
        if ch == "{":
            depth += 1
            cur.append(ch)
        elif ch == "}":
            depth -= 1
            cur.append(ch)
        elif ch == ";" and depth == 0:
            yield "".join(cur)
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        yield "".join(cur)


def rust_array_dims(arr: str):
    return re.findall(r"\[([^\]]+)\]", arr)


def emit_struct(name, kind, body, union_names, indent="    "):
    lines = ["#[repr(C)]"]
    derives = ["Copy", "Clone"]
    # A union cannot derive `Debug`, and neither can anything holding one.
    if kind != "union" and not has_union(body, union_names):
        derives.append("Debug")
    lines.append("#[derive(" + ", ".join(derives) + ")]")
    kw = "union" if kind == "union" else "struct"
    if kind == "union":
        lines.append(f"pub union {name} {{")
    else:
        lines.append(f"pub struct {name} {{")
    pending = []
    for raw in split_fields(body):
        field = raw.strip()
        if not field:
            continue
        # Anonymous nested union/struct with a trailing member name.
        if re.match(r"^(struct|union)\s*\{", field):
            raise SystemExit(f"anonymous nested aggregate in {name}: {field[:60]!r}")
        m = re.match(r"^(.*?)\(\s*\*\s*(\w+)\s*\)\s*\((.*)\)$", field, re.S)
        if m:
            ret, fname, args = m.groups()
            sig = emit_fn_sig(ret.strip(), args.strip())
            lines.append(f"    pub {fname}: {sig},")
            continue
        fm = FIELD_RE.match(field)
        if not fm:
            raise SystemExit(f"cannot parse field in {name}: {field!r}")
        ctype = fm.group("type")
        stars = fm.group("stars")
        fname = fm.group("name")
        dims = rust_array_dims(fm.group("arr"))
        # `const T *` vs `T *`
        pointee_const = bool(re.search(r"\bconst\b", ctype))
        rust = c_type_to_rust(ctype + stars)
        if stars:
            n = len(stars)
            # rebuild with correct mutability: outer-most pointer const-ness is
            # not expressible here, so only the pointee is marked.
            rust = c_type_to_rust(ctype)
            if pointee_const:
                rust = f"*const {rust}"
            else:
                rust = f"*mut {rust}"
            for _ in range(n - 1):
                rust = f"*mut {rust}"
        for d in reversed(dims):
            rust = f"[{rust}; {d}]"
        lines.append(f"    pub {rid(fname)}: {rust},")
    lines.extend(pending)
    lines.append("}")
    return lines


def emit_fn_sig(ret: str, args: str) -> str:
    params = []
    if args.strip() and args.strip() != "void":
        for a in split_args(args):
            params.append(emit_param(a))
    ret = ret.strip()
    sig = "unsafe extern \"system\" fn(" + ", ".join(params) + ")"
    if ret and ret != "void":
        sig += f" -> {c_type_to_rust(ret)}"
    return sig


def split_args(args: str):
    depth = 0
    cur = []
    for ch in args:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            yield "".join(cur)
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        yield "".join(cur)


def emit_param(a: str) -> str:
    a = a.strip()
    if a == "...":
        return "..."
    # arrays decay to pointers in a parameter list
    a = re.sub(r"\[[^\]]*\]", "*", a)
    depth = 0
    for idx, ch in enumerate(a):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "(" and depth == 0:
            break
    m = re.match(r"^(.*?)\(\s*\*\s*(\w+)\s*\)\s*\((.*)\)$", a, re.S)
    if m:
        ret, fname, inner = m.groups()
        return (
            f"{rid(fname)}: Option<unsafe extern \"system\" fn({emit_args_only(inner)})"
            f" -> {c_type_to_rust(ret)}>"
        )
    # function pointer without a name
    m = re.match(r"^(.*?)\(\s*\*\s*\)\s*\((.*)\)$", a, re.S)
    if m:
        ret, inner = m.groups()
        return f"Option<unsafe extern \"system\" fn({emit_args_only(inner)}) -> {c_type_to_rust(ret)}>"
    stars = "*" * a.count("*")
    pointee_const = bool(re.search(r"\bconst\b", a[: a.index("*")])) if "*" in a else False
    base = a.replace("*", " ")
    base = re.sub(r"\bconst\b", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    parts = base.split(" ")
    if len(parts) == 1:
        return f"_: {c_type_to_rust(parts[0])}"
    name = parts[-1]
    ctype = " ".join(parts[:-1])
    if stars:
        rust = c_type_to_rust(ctype)
        rust = f"*const {rust}" if pointee_const else f"*mut {rust}"
        for _ in range(len(stars) - 1):
            rust = f"*mut {rust}"
    else:
        rust = c_type_to_rust(ctype)
    return f"{rid(name)}: {rust}"


def emit_args_only(args: str) -> str:
    out = []
    for a in split_args(args):
        out.append(emit_param(a))
    return ", ".join(out)


def has_union(body: str, union_names) -> bool:
    if re.search(r"\bunion\b", body):
        return True
    for t in TYPE_RE.findall(body):
        if t in union_names:
            return True
    return False


def split_commas(body: str):
    depth = 0
    cur = []
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            yield "".join(cur)
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        yield "".join(cur)


def parse_enum_constants(body: str):
    """Enum members are comma separated, and a member without `=` takes the
    previous one plus one."""
    out = []
    for raw in split_commas(body):
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            nm, val = item.split("=", 1)
            out.append((nm.strip(), val.strip()))
        else:
            out.append((item, None))
    return out


def collect_defines(text: str):
    """Object-like `#define VK_*` values the generated array extents need."""
    out = {}
    for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+(VK_[A-Za-z0-9_]+)[ \t]+([^\n\\]+)$", text, re.M):
        name, val = m.group(1), m.group(2).strip()
        if val and not val.startswith("("):
            out[name] = val
    return out


def strip_preprocessor(text: str) -> str:
    """Drop `#if`/`#endif` lines, which some enum bodies carry."""
    return re.sub(r"^[ \t]*#.*$", "", text, flags=re.M)


def main():
    argv = list(sys.argv[1:])
    layout_dir = None
    if "--layout" in argv:
        i = argv.index("--layout")
        layout_dir = argv[i + 1]
        del argv[i : i + 2]
    core_path, vs_path = argv[0], argv[1]
    raw_core = open(core_path).read()
    _hv = re.search(r"#define\s+VK_HEADER_VERSION\s+(\d+)", raw_core)
    header_version = _hv.group(1) if _hv else "unknown"
    defines = collect_defines(raw_core)
    core = strip_comments(strip_preprocessor(raw_core))
    vs = strip_comments(open(vs_path).read())
    # The handle macros sit on preprocessor lines, so re-add them verbatim.
    core += "\n" + "\n".join(
        re.findall(r"VK_DEFINE_(?:NON_DISPATCHABLE_)?HANDLE\(\w+\)", raw_core)
    )
    decls = index_header(core)
    needed, missing, fns = compute_closure(core, vs, decls)

    union_names = {n for n in needed if decls.get(n) and decls[n].kind == "union"}

    out = []
    w = out.append
    w("/*")
    w(" This Source Code Form is subject to the terms of the Mozilla Public")
    w(" License, v. 2.0. If a copy of the MPL was not distributed with this")
    w(" file, You can obtain one at http://mozilla.org/MPL/2.0/.")
    w("*/")
    w("")
    w("//! The subset of `<vulkan/vulkan_core.h>` that `VSVulkan4.h` reaches.")
    w(f"//! Generated from Vulkan headers `VK_HEADER_VERSION {header_version}`.")
    w("//!")
    w("//! Generated by `scripts/gen-vulkan-bindings.py`; do not edit by hand.")
    w("//! `VSVulkan4.h` requires Vulkan 1.4, so these are the 1.4 spellings.")
    w("//!")
    w("//! Enums are `c_int` aliases with associated constants rather than Rust")
    w("//! `enum`s: Vulkan enums are extensible, and a value the header does not")
    w("//! name is not an error a filter should turn into undefined behaviour.")
    w("")
    w("#![allow(non_camel_case_types)]")
    w("#![allow(non_snake_case)]")
    w("#![allow(dead_code)]")
    w("#![allow(non_upper_case_globals)]")
    w("// Vulkan constants are written as their C decimal literals on purpose: the")
    w("// header is the source of truth for them, so grouping them differently")
    w("// here would only make them harder to compare against it.")
    w("#![allow(clippy::unreadable_literal)]")
    w("")
    w("#![allow(unused_imports)]")
    w("use core::ffi::{c_char, c_double, c_float, c_int, c_long, c_longlong,")
    w("    c_schar, c_short, c_uchar, c_uint, c_ulong, c_ulonglong, c_ushort, c_void};");
    w("")

    order = ["handle_disp", "handle_nd", "alias", "enum", "struct", "union", "pfn"]
    by_kind = {k: [] for k in order}
    for name in sorted(needed):
        d = decls.get(name)
        if d:
            by_kind[d.kind].append(name)

    w("// MARK: Handles")
    for name in by_kind["handle_disp"]:
        w(f"pub type {name} = *mut c_void;")
    for name in by_kind["handle_nd"]:
        w(f"pub type {name} = u64;")
    w("")

    w("// MARK: Scalar aliases")
    for name in by_kind["alias"]:
        base = decls[name].body
        if base.startswith("enum ") or base.startswith("struct ") or base.startswith("union "):
            base = base.split(" ", 1)[1]
        w(f"pub type {name} = {c_type_to_rust(base)};")
    w("")

    w("// MARK: Enums")
    for name in by_kind["enum"]:
        body = decls[name].body
        w(f"pub type {name} = {ENUM_BASE};")
        last = None
        for cn, val in parse_enum_constants(body):
            if val is None:
                w(f"pub const {cn}: {name} = {last} + 1;")
            else:
                w(f"pub const {cn}: {name} = {_clean_const(val)};")
            last = cn
        w("")

    w("// MARK: Structs and unions")
    struct_lines = []
    for name in by_kind["struct"] + by_kind["union"]:
        d = decls[name]
        struct_lines.append("")
        struct_lines.append(f"/// `{name}`")
        struct_lines.extend(emit_struct(name, d.kind, d.body, union_names))
        struct_lines.append("")

    # Array extents in Vulkan structs are macros (`VK_UUID_SIZE`), so they have
    # to come along as `usize` constants for the array types to be nameable.
    dims = sorted(
        {
            d
            for d in re.findall(r"\[[^;\[\]]*;\s*(\w+)\s*\]", "\n".join(struct_lines))
            if not d.isdigit()
        }
        # `VSVulkan4.h` declares its own arrays over the same macros
        # (`VK_LUID_SIZE` reaches no Vulkan struct), so scan it as well.
        | set(re.findall(r"\[\s*([A-Z_][A-Z0-9_]*)\s*\]", vs))
    )
    if dims:
        w("// MARK: Array extents")
        for cname in dims:
            val = defines.get(cname)
            if val is None:
                raise SystemExit(f"array extent {cname} has no VK_ #define")
            w(f"pub const {cname}: usize = {_clean_const(val)};")
        w("")

    out.extend(struct_lines)

    w("// MARK: Function pointers")
    for name in by_kind["pfn"]:
        d = decls[name]
        w(f"pub type {name} = Option<{emit_fn_sig(_pfn_ret(d), d.body)}>;")
    w("")
    w("// MARK: VSVulkanFunctions")
    w("/// The core's ready loaded Vulkan dispatch table, from `VSVulkan4.h`.")
    w("///")
    w("/// Every member is an `Option`: the header marks the debug utils entries")
    w("/// optional, so a null one must be checked at the call site rather than")
    w("/// called. The rest are core Vulkan 1.4 and are always present.")
    w("///")
    w("/// The member order is the header's, and the struct is one flat run of")
    w("/// pointers, so a filter built against a different header would silently")
    w("/// call the neighbouring entry point. Rebuild when it changes.")
    w("#[repr(C)]")
    w("#[derive(Copy, Clone, Debug)]")
    w("pub struct VSVulkanFunctions {")
    for fn in fns:
        w(f"    pub vk{fn}: PFN_vk{fn},")
    w("}")
    w("")

    sys.stdout.write("\n".join(out) + "\n")
    print(
        f"// closure: {len(needed)} types; unresolved: {missing}",
        file=sys.stderr,
    )

    if layout_dir is not None:
        emit_layout(decls, needed, layout_dir, argv[2])


def _pfn_ret(d: Decl) -> str:
    m = re.search(r"typedef\s+(.*?)\(\s*VKAPI_PTR", d.text, re.S)
    return (m.group(1) if m else "void").strip()


def field_names(body: str):
    """Field names of a struct, for the offset comparison."""
    names = []
    for raw in split_fields(body):
        field = raw.strip()
        if not field or re.match(r"^(struct|union)\s*\{", field):
            continue
        m = re.match(r"^(.*?)\(\s*\*\s*(\w+)\s*\)\s*\((.*)\)$", field, re.S)
        if m:
            names.append(m.group(2))
            continue
        fm = FIELD_RE.match(field)
        if fm:
            names.append(fm.group("name"))
    return names


def emit_layout(decls, needed, outdir, vulkan_rs):
    """Write a C and a Rust probe that print the same size/align/offset lines.

    Diffing their output is what proves the generated `#[repr(C)]` layout
    matches what the C compiler does with the same header.
    """
    import os

    c, rs = [], []
    for name in sorted(needed):
        d = decls.get(name)
        if not d or d.kind not in ("struct", "union"):
            continue
        c.append(f'  printf("T {name} %zu %zu\\n", sizeof({name}), _Alignof({name}));')
        rs.append(
            f'    println!("T {name} {{}} {{}}", size_of::<{name}>(), align_of::<{name}>());'
        )
        if d.kind == "union":
            continue  # member offsets in a union are all zero and not meaningful
        for f in field_names(d.body):
            if f == "_":
                continue
            c.append(f'  printf("F {name}.{f} %zu\\n", offsetof({name}, {f}));')
            rs.append(f'    println!("F {name}.{f} {{}}", offset_of!({name}, {rid(f)}));')

    with open(os.path.join(outdir, "layout.c"), "w") as fh:
        fh.write("#include <stddef.h>\n#include <stdio.h>\n")
        fh.write("#include <vulkan/vulkan_core.h>\n#include <VSVulkan4.h>\n\n")
        fh.write("int main(void) {\n" + "\n".join(c) + "\n  return 0;\n}\n")

    with open(os.path.join(outdir, "layout.rs"), "w") as fh:
        fh.write("#[path = \"%s\"]\nmod vulkan;\n" % vulkan_rs)
        fh.write("use core::mem::{align_of, offset_of, size_of};\n")
        fh.write("#[allow(unused_imports)]\nuse vulkan::*;\n\n")
        fh.write("fn main() {\n" + "\n".join(rs) + "\n}\n")



def _clean_const(val: str) -> str:
    """Vulkan writes a few sentinels as C casts or bitwise negation."""
    v = val.strip()
    v = re.sub(r"\(\s*\w+\s*\)\s*", "", v)  # drop C casts
    v = re.sub(r"~0U?L{0,2}\b", "-1", v)
    # Integer suffixes only, and only in something that is entirely numeric:
    # a blanket `U` strip would mangle VK_STRUCTURE_TYPE_... into VK_STRCTRE.
    if re.fullmatch(r"[-0-9A-Fa-fxXUuLl~()\s|&+*<>]+", v):
        v = re.sub(r"\b(0[xX][0-9A-Fa-f]+|[0-9]+)(ULL|LL|UL|LU|U|L)\b", r"\1", v)
    return v

if __name__ == "__main__":
    main()
