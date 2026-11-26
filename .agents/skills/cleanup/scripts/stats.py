"""Line statistics for a repository's checkout, for the audit step of a cleanup pass.

Usage: python3 -I stats.py [REPO] [--json OUT]

Prints a summary and, with --json, writes every per-file record and the
history to OUT for charts. Generated files are the ones .gitattributes marks
linguist-generated, and lockfiles are Cargo.lock, flake.lock and pnpm-lock.yaml. Rust lines
are classified by a small lexer: a line is code, a doc comment (/// or //!), a
plain comment or blank, and a line with code and a trailing comment counts as
code. Rust test code is whole files under tests/, tests.rs files, and modules
and inline items behind a test-only cfg: #[cfg(test)], #[cfg(all(test, ...))]
or #[cfg(any(test, feature = "test-util"))]. The counts of unwrap, expect and
the like read only production code lines, so doc examples are left out;
TODO, FIXME and XXX are counted in comments too. TypeScript and JavaScript
test code is files under a test/ or tests/ directory and *.spec.ts or
*.test.ts files. Areas are a workspace member (apps/x, crates/x, packages/x,
tools/x, tests/x) or another top-level directory. Other languages are counted
by cloc.
"""
import bisect, csv, io, json, os, re, subprocess, sys
from collections import defaultdict

args = sys.argv[1:]
out = None
if "--json" in args:
    i = args.index("--json")
    out = os.path.abspath(args[i + 1])
    del args[i:i + 2]
os.chdir(args[0] if args else ".")


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


if git("rev-parse", "--is-shallow-repository").strip() == "true":
    print("warning: shallow clone, so the growth history starts at its boundary", file=sys.stderr)
files = git("ls-files", "-z").split("\0")[:-1]
attrs = git("check-attr", "-z", "linguist-generated", "--", *files).split("\0")
marked = {attrs[i] for i in range(0, len(attrs) - 2, 3) if attrs[i + 2] in ("set", "true")}
LOCKFILE = re.compile(r"(^|/)(Cargo\.lock|flake\.lock|pnpm-lock\.yaml)$")
LOCKFILES = {p for p in files if LOCKFILE.search(p)}
GENERATED = marked - LOCKFILES
ROOT = "root & CI"
# An attribute compiling its item only for tests, or for other crates' tests
# through the `test-util` feature.
TEST_CFG = r'#\[cfg\(\s*(?:test|all\(\s*test\b[^\]]*|any\(\s*test\s*,\s*feature\s*=\s*"test-util"\s*\))\s*\)\]'


def area(p):
    parts = p.split("/")
    if parts[0] in ("apps", "crates", "packages", "tools", "tests") and len(parts) > 2:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) > 1 and not parts[0].startswith("."):
        return parts[0]
    return ROOT


def ts_test(p):
    """Whether p is TypeScript or JavaScript test code."""
    return p.endswith((".ts", ".js")) and (
        re.search(r"(^|/)tests?/", p) is not None or p.endswith((".spec.ts", ".test.ts")))


def module(p):
    """The top-level module or directory of an area holding p: apps/x/src/oidc, apps/x/tests."""
    a = area(p)
    if a == ROOT:
        return a
    parts = p[len(a) + 1:].split("/")
    head = parts[:2] if parts[0] == "src" and len(parts) > 2 else parts[:1]
    return a + "/" + "/".join(head)


RAW = re.compile(r'b?r(#*)"')


def lex_rust(src):
    """Per-line flags (code, doc, comment) and the offsets of braces in code."""
    n, lines = len(src), src.count("\n") + 1
    code, doc, com = [False] * lines, [False] * lines, [False] * lines
    braces, i, line = [], 0, 0
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1; i += 1; continue
        if c in " \t\r":
            i += 1; continue
        if src.startswith("//", i):
            is_doc = (src.startswith("///", i) and not src.startswith("////", i)) or src.startswith("//!", i)
            (doc if is_doc else com)[line] = True
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            is_doc = (src.startswith("/**", i) and not src.startswith("/**/", i)) or src.startswith("/*!", i)
            kind = doc if is_doc else com
            kind[line] = True
            depth, i = 1, i + 2
            while i < n and depth:
                if src.startswith("/*", i): depth += 1; i += 2
                elif src.startswith("*/", i): depth -= 1; i += 2
                else:
                    if src[i] == "\n":
                        line += 1; kind[line] = True
                    i += 1
            continue
        code[line] = True
        m = RAW.match(src, i)
        if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            end = '"' + m.group(1)
            j = src.find(end, m.end())
            j = n if j < 0 else j + len(end)
            for k in range(i, j):
                if src[k] == "\n": line += 1; code[line] = True
            i = j; continue
        if c == '"' or src.startswith('b"', i):
            i += 1 if c == '"' else 2
            while i < n and src[i] != '"':
                if src[i] == "\\": i += 1
                if i < n and src[i] == "\n": line += 1; code[line] = True
                i += 1
            i += 1; continue
        if c == "'":
            if i + 1 < n and src[i + 1] == "\\":
                i = src.find("'", i + 2) + 1; continue
            i += 3 if i + 2 < n and src[i + 2] == "'" else 1
            continue
        if c in "{}":
            braces.append((i, c))
        i += 1
    return code, doc, com, braces


def inline_test_lines(src, braces):
    """Line numbers of test-only items inside a file that isn't all test code."""
    test, offsets = set(), [b[0] for b in braces]
    for m in re.finditer(TEST_CFG, src):
        semi = src.find(";", m.end())
        k = bisect.bisect_left(offsets, m.end())
        brace = braces[k][0] if k < len(braces) else len(src)
        if k < len(braces) and braces[k][1] == "}":
            # A field, enum variant or match arm: it ends at its `,`, a last
            # statement at its `;`, and a last one without either just before
            # the `}` closing its parent.
            ends = [e for e in (src.find(",", m.end(), brace), src.find(";", m.end(), brace)) if e >= 0]
            end = min(ends, default=len(src[:brace].rstrip()) - 1)
        elif 0 <= semi < brace:
            end = semi
        else:
            depth, end = 0, len(src)
            for off, ch in braces[k:]:
                depth += 1 if ch == "{" else -1
                if depth == 0:
                    end = off; break
        test.update(range(src.count("\n", 0, m.start()), src.count("\n", 0, end) + 1))
    return test


def test_files(rs, sources):
    """Rust files that are test code as a whole."""
    tf = {p for p in rs if "/tests/" in "/" + p or p.endswith("/tests.rs")}
    for p in rs:
        for m in re.finditer(TEST_CFG + r"\s*(?:pub(?:\([a-z]+\))?\s+)?mod\s+(\w+)\s*;", sources[p]):
            d, base = os.path.split(p)
            stem = base[:-3]
            root = os.path.join(d if stem in ("mod", "lib", "main") else os.path.join(d, stem), m.group(1))
            tf.update(q for q in rs if q == root + ".rs" or q.startswith(root + "/"))
    return tf


others = [p for p in files if not p.endswith(".rs") and os.path.isfile(p)]
cloc = {}
report = subprocess.run(["cloc", "--by-file", "--csv", "--quiet", "--skip-uniqueness", "--list-file=-"],
                        input="\n".join(others), capture_output=True, text=True, check=True).stdout
for row in csv.reader(io.StringIO(report)):
    if len(row) >= 5 and row[0] not in ("language", "SUM"):
        cloc[row[1].removeprefix("./")] = (row[0], int(row[2]), int(row[3]), int(row[4]))

rs = [p for p in files if p.endswith(".rs")]
sources = {p: open(p, encoding="utf-8").read() for p in rs}
whole_tests = test_files(rs, sources)
FACTS = {
    "unsafe": r"\bunsafe\s*(\{|fn\b|impl\b|trait\b)", "#[allow": r"#!?\[allow\(",
    "#[expect": r"#!?\[expect\(", ".unwrap()": r"\.unwrap\(\)", ".expect(": r"\.expect\(",
    "TODO/FIXME": r"\b(TODO|FIXME|XXX)\b",
}
facts = defaultdict(int)
tests = defaultdict(int)
ignored = defaultdict(int)  # Rust tests that run only when asked, by reason
records = []
for p in files:
    kind = "generated" if p in GENERATED else "lockfile" if p in LOCKFILES else "handwritten"
    rec = dict(path=p, area=area(p), kind=kind, lang="Other", blank=0, code=0, doc=0, comment=0,
               test_code=0, test_comment=0)
    if p.endswith(".rs"):
        src = sources[p]
        code, doc, com, braces = lex_rust(src)
        lines = src.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        tl = set(range(len(lines))) if p in whole_tests else inline_test_lines(src, braces)
        for ln in range(len(lines)):
            t = ln in tl
            if code[ln]:
                rec["test_code" if t else "code"] += 1
            elif doc[ln] or com[ln]:
                rec["test_comment" if t else "doc" if doc[ln] else "comment"] += 1
            else:
                rec["blank"] += 1
        rec["lang"] = "Rust"
        if kind == "handwritten":
            prod = [i for i in range(len(lines)) if i not in tl]
            prod_code = "\n".join(lines[i] for i in prod if code[i])
            prod_all = "\n".join(lines[i] for i in prod)
            for k, rx in FACTS.items():
                facts[k] += len(re.findall(rx, prod_all if k == "TODO/FIXME" else prod_code, re.M))
            n = len(re.findall(r"#\[(tokio::|wasm_bindgen_)?test\b", src))
            tests["rust integration" if "/tests/" in "/" + p and "/src/" not in p else "rust unit"] += n
            for reason in re.findall(r'#\[ignore(?: = "([^"]*)")?\]', src):
                ignored[reason or "no reason"] += 1
    elif p in cloc:
        lang, b, c, k = cloc[p]
        rec.update(lang=lang, blank=b, comment=c, code=k)
        if ts_test(p):
            rec["test_code"], rec["code"], rec["test_comment"], rec["comment"] = k, 0, c, 0
            if p.endswith((".spec.ts", ".test.ts")) and "/screenshots/" not in p:
                tests["typescript"] += len(re.findall(r"^\s*test(\.only)?\(", open(p).read(), re.M))
    elif os.path.isfile(p):
        try:
            text = open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        ls = text.split("\n")
        if ls and ls[-1] == "":
            ls.pop()
        rec["blank"] = sum(1 for l in ls if not l.strip())
        rec["code"] = len(ls) - rec["blank"]
    if kind == "lockfile":
        rec["lang"] = "Lockfile"
    records.append(rec)


def total(r):
    return sum(r[k] for k in ("blank", "code", "doc", "comment", "test_code", "test_comment"))


def hist_cat(p):
    if p in GENERATED: return "generated"
    if p in LOCKFILES: return "lockfile"
    if p.endswith(".md"): return "docs"
    if (p.endswith(".rs") and ("/tests/" in "/" + p or p.endswith("tests.rs"))) or \
            ts_test(p):
        return "test"
    if p.endswith((".png", ".jpg", ".ico", ".woff2")): return "binary"
    return "source"


# Growth by category, one point per first-parent commit. Inline test modules
# count as source here, so "test" is lower than in the summary.
history, cur, running = [], None, defaultdict(int)
for l in git("log", "--reverse", "--first-parent", "--no-renames", "--numstat",
             "--format=@@%h %aI %s", "HEAD").split("\n"):
    if l.startswith("@@"):
        if cur: history.append(cur)
        sha, date, *subject = l[2:].split(" ")
        cur = {"sha": sha, "date": date, "subject": " ".join(subject)}
    elif l.strip():
        added, deleted, path = l.split("\t", 2)
        if added != "-":
            running[hist_cat(path)] += int(added) - int(deleted)
            cur.update(running)
if cur: history.append(cur)

# Summary.
hand = [r for r in records if r["kind"] == "handwritten"]
all_lines = sum(map(total, records))
print(f"HEAD {git('rev-parse', '--short', 'HEAD').strip()}: {len(files)} files, {all_lines:,} lines")
for kind in ("handwritten", "generated", "lockfile"):
    n = sum(total(r) for r in records if r["kind"] == kind)
    print(f"  {kind:<12} {n:>9,}  {n / all_lines:5.1%}")

print("\nHand-written by language (code / comments / blank):")
langs = defaultdict(lambda: [0, 0, 0])
for r in hand:
    l = langs[r["lang"]]
    l[0] += r["code"] + r["test_code"]; l[1] += r["doc"] + r["comment"] + r["test_comment"]; l[2] += r["blank"]
for lang, (c, m, b) in sorted(langs.items(), key=lambda kv: -kv[1][0])[:10]:
    print(f"  {lang:<16} {c:>8,} {m:>8,} {b:>8,}")

rust = [r for r in hand if r["lang"] == "Rust"]
src, tst = sum(r["code"] for r in rust), sum(r["test_code"] for r in rust)
doc, com = sum(r["doc"] for r in rust), sum(r["comment"] for r in rust)
print(f"\nRust: {src:,} source, {tst:,} test ({tst / (src + tst):.0%}), "
      f"{doc:,} doc comment and {com:,} plain comment lines outside tests")

print("\nBy area (source / test / test share):")
areas = defaultdict(lambda: [0, 0])
for r in hand:
    if r["lang"] not in ("Markdown", "Other"):
        areas[r["area"]][0] += r["code"]; areas[r["area"]][1] += r["test_code"]
for a, (s, t) in sorted(areas.items(), key=lambda kv: -sum(kv[1])):
    print(f"  {a:<24} {s:>8,} {t:>8,}  {t / (s + t) if s + t else 0:5.0%}")

modules = defaultdict(lambda: [0, 0])
for r in hand:
    if r["lang"] not in ("Markdown", "Other", "TOML") and sum(areas[r["area"]]) > 30000:
        modules[module(r["path"])][0] += r["code"]; modules[module(r["path"])][1] += r["test_code"]
if modules:
    print("\nModules of areas over 30,000 lines (source / test):")
    for m, (s, t) in sorted(modules.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {m:<40} {s:>8,} {t:>8,}")

print("\nHand-written files over 1,500 lines:")
for r in sorted((r for r in hand if r["lang"] not in ("Markdown", "Other") and total(r) > 1500),
                key=total, reverse=True):
    print(f"  {total(r):>6,}  {r['path']}")

print("\nTests:", ", ".join(f"{v} {k}" for k, v in tests.items()))
if ignored:
    print("Ignored unless asked:", ", ".join(f"{v} ({k})" for k, v in sorted(ignored.items(), key=lambda kv: -kv[1])))
print("Outside tests:", ", ".join(f"{v} {k}" for k, v in facts.items()))
print("(regex counts: close, not exact)")

# `foo.rs` next to `foo/` is caught by clippy::self_named_module_files. The
# other half of the layout rule is that a `mod.rs` needs children, a tests.rs
# at least. An integration test's shared `tests/<name>/mod.rs` is exempt.
lonely = [p for p in rs if p.endswith("/mod.rs") and not re.search(r"(^|/)tests/[^/]+/mod\.rs$", p)
          and not any(q != p and q.startswith(os.path.dirname(p) + "/") for q in rs)]
if lonely:
    print("\nmod.rs files with no child modules (should be a plain foo.rs):")
    for p in lonely:
        print(f"  {p}")

if out:
    with open(out, "w") as f:
        json.dump({"files": records, "facts": facts, "tests": tests, "ignored": ignored, "history": history,
                   "generated": sorted(GENERATED), "lockfiles": sorted(LOCKFILES)}, f, indent=1)
    print(f"\nWrote {out}")
