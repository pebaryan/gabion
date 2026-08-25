"""Static checks over the WGSL kernel sources.

These guard against the class of bug where a kernel compiles (Tint is lenient)
but is silently wrong on real GPUs — something the transpiled-JS kernel smoke
cannot catch because it never compiles the WGSL itself. The first such bug was
found by the real-browser smoke test (tests/webgpu_smoke.html): reduce.wgsl
declared `var<workgroup> shared`, and `shared` is a reserved word in WGSL; the
shader produced all-zero output on an NVIDIA adapter while passing the
hand-transpiled checks.
"""
import re
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parent.parent / "gabion" / "web" / "kernels"

# Reserved words from the WGSL spec that Tint accepts as identifiers anyway.
# Using any of these as an identifier is undefined behavior on real drivers.
WGSL_RESERVED = {
    "NULL", "Self", "abstract", "active", "alignas", "alignof", "asm",
    "asm_fragment", "async", "attribute", "auto", "await", "become",
    "binding_array", "cast", "catch", "class", "co_await", "co_return",
    "co_yield", "coherent", "column_major", "common", "compile",
    "compile_fragment", "concept", "const_cast", "consteval", "constexpr",
    "constinit", "crate", "debugger", "decltype", "delete", "demote",
    "demote_to_helper", "do", "dynamic_cast", "enum", "explicit", "export",
    "extends", "extern", "external", "fallthrough", "filter", "final",
    "finally", "friend", "from", "fxgroup", "get", "goto", "groupshared",
    "highp", "impl", "implements", "import", "inline", "instanceof",
    "interface", "layout", "lowp", "macro", "macro_rules", "match",
    "mediump", "meta", "mod", "module", "move", "mut", "mutable",
    "namespace", "new", "nil", "noexcept", "noinline", "nointerpolation",
    "noperspective", "null", "nullptr", "of", "operator", "package",
    "packoffset", "partition", "pass", "patch", "pixelfragment", "precise",
    "precision", "premerge", "priv", "protected", "pub", "public",
    "readonly", "ref", "regardless", "register", "reinterpret_cast",
    "require", "resource", "restrict", "self", "set", "shared", "signed",
    "sizeof", "smooth", "snorm", "static", "static_assert", "static_cast",
    "std", "subgroup", "super", "target", "template", "this", "thread",
    "thread_local", "throw", "trait", "try", "type", "typedef", "typeid",
    "typename", "typeof", "union", "unknown", "unorm", "unsized",
    "unsigned", "use", "using", "varying", "virtual", "volatile", "wgsl",
    "where", "with", "writeonly", "yield",
}


def _identifiers_in_wgsl(text: str) -> list[str]:
    """All declared/used identifiers from var/let/struct/fn declarations.

    Comments are stripped first so prose like "mean/var from the stats pass"
    cannot produce phantom identifiers.
    """
    text = re.sub(r"//[^\n]*", "", text)
    ids = set()
    # var<address_space> name: type  (and var name: type)
    for m in re.finditer(r"\bvar\s*(?:<[^>]*>)?\s*([A-Za-z_][A-Za-z0-9_]*)", text):
        ids.add(m.group(1))
    # let name =
    for m in re.finditer(r"\blet\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        ids.add(m.group(1))
    # struct Name {
    for m in re.finditer(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        ids.add(m.group(1))
    # fn name(
    for m in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        ids.add(m.group(1))
    return sorted(ids)


def test_no_wgsl_reserved_words_as_identifiers():
    bad = []
    for path in sorted(KERNEL_DIR.glob("*.wgsl")):
        ids = _identifiers_in_wgsl(path.read_text(encoding="utf-8"))
        for ident in ids:
            if ident in WGSL_RESERVED:
                bad.append(f"{path.name}: '{ident}'")
    assert not bad, (
        "WGSL reserved words used as identifiers (Tint compiles these but "
        "real drivers produce undefined/zero results — see reduce.wgsl 'shared' "
        f"bug): {bad}"
    )


def test_workgroup_array_size_matches_workgroup_size():
    """var<workgroup> arrays must be sized to the kernel's workgroup_size."""
    for path in sorted(KERNEL_DIR.glob("*.wgsl")):
        text = path.read_text(encoding="utf-8")
        wg_sizes = re.findall(r"@compute\s+@workgroup_size\((\d+)\)", text)
        if not wg_sizes:
            continue
        size = int(wg_sizes[0])
        for m in re.finditer(
            r"var<workgroup>\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*array<f32,\s*(\d+)>",
            text,
        ):
            assert int(m.group(2)) >= size, (
                f"{path.name}: workgroup array '{m.group(1)}' has {m.group(2)} "
                f"slots but workgroup_size is {size}"
            )


def test_no_shared_identifier():
    """Directly assert the historical bug stays fixed: `shared` as an id."""
    for path in sorted(KERNEL_DIR.glob("*.wgsl")):
        text = path.read_text(encoding="utf-8")
        # strip comments
        no_comments = re.sub(r"//[^\n]*", "", text)
        for m in re.finditer(r"\bshared\b", no_comments):
            assert False, f"{path.name}: 'shared' used as an identifier at:\n  {no_comments[max(0,m.start()-60):m.end()+60]}"
