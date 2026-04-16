#!/usr/bin/env python3
"""
Walker for the claim-graph overlay on docs/proof-layer-analysis.md.

Spec: docs/spec/claims-convention.md.

Parses ``` ```claim ``` ``` fenced blocks from the reference manual,
validates the resulting graph, and reports UNJUSTIFIED leaves and the
upstream claims they transitively block. Exits non-zero on any
validation error or on a `justified` claim with an UNJUSTIFIED
transitive leaf.

No third-party dependencies (stdlib only).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MANUAL_PATH = REPO_ROOT / "docs" / "proof-layer-analysis.md"
CLAIMS_MD_PATH = REPO_ROOT / "docs" / "CLAIMS.md"

LEAF_TAGS = ("PAPER", "CODE", "ASSUMPTION", "UNJUSTIFIED")
COMBINATORS = ("AND_OF", "THEOREM", "SEQUENTIAL_COMPOSITION")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LeafEntry:
    tag: str            # one of LEAF_TAGS
    content: str        # inside-parens payload
    raw: str            # original string, e.g. "PAPER(Hyrax Theorem 7)"


@dataclass
class Measurement:
    hardware: Optional[str] = None
    commit: Optional[str] = None
    date: Optional[str] = None
    result: Optional[str] = None
    threshold: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Claim:
    id: str
    statement: Optional[str] = None
    combinator: Optional[str] = None        # None => leaf
    justifiedBy_ids: list[str] = field(default_factory=list)     # internal
    justifiedBy_leaves: list[LeafEntry] = field(default_factory=list)  # leaf
    status: str = "justified"
    superseded_by: Optional[str] = None
    measurement: Optional[Measurement] = None
    section_anchor: Optional[str] = None    # filled during parse
    source_lineno: int = 0                  # 1-based line of `id:` for error messages

    @property
    def is_leaf(self) -> bool:
        return self.combinator is None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

CLAIM_FENCE_OPEN = re.compile(r"^```claim\s*$")
CLAIM_FENCE_CLOSE = re.compile(r"^```\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LEAF_TAG_RE = re.compile(r"^\s*-\s*(%s)\((.*)\)\s*$" % "|".join(LEAF_TAGS))
ID_REF_RE = re.compile(r"^\s*-\s*([A-Z][A-Z0-9_-]*)\s*$")


def slugify_heading(text: str) -> str:
    """Rough GitHub-style slug for section anchors."""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s


def parse_manual(path: Path) -> list[Claim]:
    """Scan the manual for ``` ```claim ``` ``` blocks and parse each."""
    if not path.exists():
        die(f"manual not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    claims: list[Claim] = []
    cur_anchor: Optional[str] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        h = HEADING_RE.match(line)
        if h:
            cur_anchor = slugify_heading(h.group(2))
            i += 1
            continue
        if CLAIM_FENCE_OPEN.match(line):
            body_start = i + 1
            j = body_start
            while j < len(lines) and not CLAIM_FENCE_CLOSE.match(lines[j]):
                j += 1
            if j >= len(lines):
                die(f"{path}:{i + 1}: unterminated ```claim``` block")
            claim = parse_block(lines[body_start:j], path, body_start + 1)
            claim.section_anchor = cur_anchor
            claims.append(claim)
            i = j + 1
            continue
        i += 1
    return claims


def parse_block(body: list[str], path: Path, start_lineno: int) -> Claim:
    """Parse a claim block body (lines between the fences)."""
    claim = Claim(id="")
    i = 0
    in_justified = False
    in_measurement = False
    in_depends_on = False
    while i < len(body):
        raw = body[i]
        lineno = start_lineno + i
        if not raw.strip():
            i += 1
            continue
        # end of nested sections when a top-level key appears
        if re.match(r"^[A-Za-z_]+\s*:", raw):
            in_justified = False
            in_measurement = False
            in_depends_on = False

        m = re.match(r"^([A-Za-z_]+)\s*:\s*(.*?)\s*$", raw)
        if m and not raw.startswith(" "):
            key, val = m.group(1), m.group(2)
            if key == "id":
                claim.id = val
                claim.source_lineno = lineno
            elif key == "statement":
                claim.statement = val
            elif key == "combinator":
                if val not in COMBINATORS and not any(
                    val.startswith(c + "(") for c in COMBINATORS
                ):
                    die(f"{path}:{lineno}: unknown combinator {val!r}")
                claim.combinator = val
            elif key == "status":
                if val not in ("justified", "open"):
                    die(f"{path}:{lineno}: status must be justified|open, got {val!r}")
                claim.status = val
            elif key == "superseded_by":
                claim.superseded_by = val or None
            elif key == "justifiedBy":
                if val:
                    die(f"{path}:{lineno}: justifiedBy: must be empty and followed by a list")
                in_justified = True
                in_measurement = False
            elif key == "measurement":
                if val:
                    die(f"{path}:{lineno}: measurement: must be empty and followed by a block")
                in_measurement = True
                in_justified = False
                claim.measurement = Measurement()
            else:
                die(f"{path}:{lineno}: unknown top-level key {key!r}")
            i += 1
            continue

        # Indented content: list item or measurement sub-field.
        if in_justified:
            leaf = LEAF_TAG_RE.match(raw)
            idref = ID_REF_RE.match(raw)
            if leaf:
                claim.justifiedBy_leaves.append(
                    LeafEntry(tag=leaf.group(1), content=leaf.group(2), raw=raw.strip()[2:])
                )
            elif idref:
                claim.justifiedBy_ids.append(idref.group(1))
            else:
                die(f"{path}:{lineno}: malformed justifiedBy entry: {raw!r}")
            i += 1
            continue

        if in_measurement:
            # Sub-field of measurement, either "  key: value" or "  depends_on:" + list
            sub = re.match(r"^\s{2,}([a-z_]+)\s*:\s*(.*?)\s*$", raw)
            if sub:
                key, val = sub.group(1), sub.group(2)
                if key == "depends_on":
                    if val:
                        die(f"{path}:{lineno}: depends_on: must be empty then list")
                    in_depends_on = True
                else:
                    in_depends_on = False
                    if claim.measurement is None:
                        claim.measurement = Measurement()
                    setattr(claim.measurement, key, val)
                i += 1
                continue
            if in_depends_on:
                item = re.match(r"^\s{2,}-\s*(\S.*?)\s*$", raw)
                if item:
                    claim.measurement.depends_on.append(item.group(1))
                    i += 1
                    continue
            die(f"{path}:{lineno}: unexpected line inside measurement: {raw!r}")

        die(f"{path}:{lineno}: unexpected line: {raw!r}")

    if not claim.id:
        die(f"{path}:{start_lineno}: claim block missing `id:`")
    return claim


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unjustified_leaves: list[tuple[str, LeafEntry]] = field(default_factory=list)  # (claim_id, leaf)
    blocked_upstream: dict[str, list[str]] = field(default_factory=dict)  # uj_claim_id -> [upstream ids]
    stale_perf: list[tuple[str, str]] = field(default_factory=list)  # (claim_id, reason)
    claims_by_id: dict[str, Claim] = field(default_factory=dict)


def validate(claims: list[Claim]) -> Report:
    rep = Report()
    # 1. unique IDs
    by_id: dict[str, Claim] = {}
    for c in claims:
        if c.id in by_id:
            rep.errors.append(
                f"{MANUAL_PATH.name}:{c.source_lineno}: duplicate claim id {c.id!r} (first at line {by_id[c.id].source_lineno})"
            )
        else:
            by_id[c.id] = c
    rep.claims_by_id = by_id

    # 2. leaf/internal consistency + tag validity
    for c in claims:
        if c.is_leaf:
            if not c.justifiedBy_leaves:
                rep.errors.append(
                    f"{MANUAL_PATH.name}:{c.source_lineno}: leaf claim {c.id} has no leaf entries"
                )
            if c.justifiedBy_ids:
                rep.errors.append(
                    f"{MANUAL_PATH.name}:{c.source_lineno}: leaf claim {c.id} mixes claim-id references into justifiedBy; leaf claims take only leaf tags"
                )
        else:
            if not c.justifiedBy_ids:
                rep.errors.append(
                    f"{MANUAL_PATH.name}:{c.source_lineno}: internal claim {c.id} has no subclaim references"
                )
            if c.justifiedBy_leaves:
                rep.errors.append(
                    f"{MANUAL_PATH.name}:{c.source_lineno}: internal claim {c.id} mixes leaf tags into justifiedBy; promote leaf evidence to a sibling leaf claim"
                )

    # 3. referenced IDs resolve
    for c in claims:
        for ref in c.justifiedBy_ids:
            if ref not in by_id:
                rep.errors.append(
                    f"{MANUAL_PATH.name}:{c.source_lineno}: claim {c.id} references unknown id {ref!r}"
                )
        if c.superseded_by and c.superseded_by not in by_id:
            rep.errors.append(
                f"{MANUAL_PATH.name}:{c.source_lineno}: claim {c.id} superseded_by unknown id {c.superseded_by!r}"
            )

    # 4. no cycles (Tarjan-lite DFS)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {cid: WHITE for cid in by_id}
    stack_trace: list[str] = []

    def dfs(cid: str) -> None:
        color[cid] = GRAY
        stack_trace.append(cid)
        for ref in by_id[cid].justifiedBy_ids:
            if ref not in by_id:
                continue
            if color[ref] == GRAY:
                cyc = stack_trace[stack_trace.index(ref):] + [ref]
                rep.errors.append(f"cycle in claim graph: {' -> '.join(cyc)}")
            elif color[ref] == WHITE:
                dfs(ref)
        color[cid] = BLACK
        stack_trace.pop()

    for cid in by_id:
        if color[cid] == WHITE:
            dfs(cid)

    # 5. CODE file-exists soft check
    for c in claims:
        for leaf in c.justifiedBy_leaves:
            if leaf.tag != "CODE":
                continue
            for f in extract_code_files(leaf.content):
                p = REPO_ROOT / f
                if not p.exists():
                    rep.errors.append(
                        f"{MANUAL_PATH.name}:{c.source_lineno}: claim {c.id} CODE leaf references missing file {f!r}"
                    )

    # 6. no live claim references a superseded id
    superseded_ids = {c.id for c in claims if c.superseded_by}
    for c in claims:
        if c.id in superseded_ids:
            continue
        for ref in c.justifiedBy_ids:
            if ref in superseded_ids:
                rep.warnings.append(
                    f"{MANUAL_PATH.name}:{c.source_lineno}: claim {c.id} references superseded id {ref!r}"
                )

    # 7. stale perf check
    for c in claims:
        if c.measurement and c.measurement.commit and c.measurement.depends_on:
            stale = git_changed(c.measurement.commit, c.measurement.depends_on)
            if stale is not None and stale:
                reason = f"stale: {stale[0]} changed after benchmark commit {c.measurement.commit[:8]}"
                rep.stale_perf.append((c.id, reason))
                # inject a synthetic UNJUSTIFIED leaf
                c.justifiedBy_leaves.append(LeafEntry("UNJUSTIFIED", reason, reason))

    # 8. collect UNJUSTIFIED leaves + compute blocked upstream
    uj_owners: dict[str, list[LeafEntry]] = {}
    for c in claims:
        for leaf in c.justifiedBy_leaves:
            if leaf.tag == "UNJUSTIFIED":
                rep.unjustified_leaves.append((c.id, leaf))
                uj_owners.setdefault(c.id, []).append(leaf)

    blocked = compute_blocked(by_id)  # set of claim IDs that are transitively blocked

    for owner_id in uj_owners:
        # upstream = claims whose transitive support includes owner_id
        ups = [cid for cid, b in blocked.items() if owner_id in b and cid != owner_id]
        ups.sort()
        rep.blocked_upstream[owner_id] = ups

    # 9. justified claims that are blocked → error
    for cid, b in blocked.items():
        c = by_id[cid]
        if c.status == "justified" and b:
            block_ids = sorted(b)
            rep.errors.append(
                f"{MANUAL_PATH.name}:{c.source_lineno}: claim {c.id} marked `justified` but is blocked by UNJUSTIFIED leaves in: {', '.join(block_ids)}"
            )

    return rep


def extract_code_files(content: str) -> list[str]:
    """CODE leaf payload is `<file>[+<file>...]` (plus delim). Strip `:lines`."""
    parts = [p.strip() for p in content.split("+")]
    out = []
    for p in parts:
        # strip optional :L-L or :N
        if ":" in p:
            p = p.split(":", 1)[0].strip()
        if p:
            out.append(p)
    return out


def compute_blocked(by_id: dict[str, Claim]) -> dict[str, set[str]]:
    """For each claim, the set of claim IDs in its transitive support that
    own an UNJUSTIFIED leaf. Empty set => not blocked."""
    memo: dict[str, set[str]] = {}

    def visit(cid: str, path: set[str]) -> set[str]:
        if cid in memo:
            return memo[cid]
        if cid in path:
            return set()  # cycle; errored elsewhere
        path.add(cid)
        c = by_id[cid]
        out: set[str] = set()
        for leaf in c.justifiedBy_leaves:
            if leaf.tag == "UNJUSTIFIED":
                out.add(cid)
        for ref in c.justifiedBy_ids:
            if ref in by_id:
                out |= visit(ref, path)
        path.remove(cid)
        memo[cid] = out
        return out

    return {cid: visit(cid, set()) for cid in by_id}


def git_changed(commit: str, paths: list[str]) -> Optional[list[str]]:
    """Run `git log commit..HEAD -- paths`. Return list of changed paths, or None if git unavailable."""
    try:
        args = ["git", "log", "--name-only", "--pretty=format:", f"{commit}..HEAD", "--"] + paths
        out = subprocess.check_output(args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True)
        changed = sorted({ln.strip() for ln in out.splitlines() if ln.strip()})
        return changed
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(rep: Report) -> int:
    claims = rep.claims_by_id.values()
    n_total = len(rep.claims_by_id)
    n_leaves = sum(1 for c in claims if c.is_leaf)
    n_internal = n_total - n_leaves
    n_justified = sum(1 for c in claims if c.status == "justified")
    n_open = sum(1 for c in claims if c.status == "open")
    n_uj = len(rep.unjustified_leaves)

    print("=" * 72)
    print("Claims walker — docs/proof-layer-analysis.md")
    print("=" * 72)
    print(f"  claims total:     {n_total}  ({n_leaves} leaves, {n_internal} internal)")
    print(f"  status:           {n_justified} justified, {n_open} open")
    print(f"  UNJUSTIFIED leaves: {n_uj}")
    if rep.stale_perf:
        print(f"  stale perf claims:  {len(rep.stale_perf)}")
    print()

    if rep.unjustified_leaves:
        print("UNJUSTIFIED leaves:")
        for owner_id, leaf in rep.unjustified_leaves:
            print(f"  [{owner_id}] UNJUSTIFIED({leaf.content})")
            ups = rep.blocked_upstream.get(owner_id, [])
            if ups:
                print(f"    blocks upstream: {', '.join(ups)}")
        print()

    if rep.warnings:
        print("Warnings:")
        for w in rep.warnings:
            print(f"  ! {w}")
        print()

    if rep.errors:
        print("Errors:")
        for e in rep.errors:
            print(f"  ✗ {e}")
        print()
        return 1

    print("OK — walker green.")
    return 0


def emit_claims_md(rep: Report) -> None:
    lines = [
        "# Claims Index",
        "",
        "Auto-generated from `docs/proof-layer-analysis.md` by `tools/check_claims.py`.",
        "Do not edit by hand.",
        "",
        "| ID | Section | Status | Statement |",
        "|---|---|---|---|",
    ]
    blocked = compute_blocked(rep.claims_by_id)
    for cid in sorted(rep.claims_by_id):
        c = rep.claims_by_id[cid]
        is_blocked = bool(blocked.get(cid))
        if c.status == "justified" and is_blocked:
            disp = "⚠ justified+blocked"
        elif c.status == "justified":
            disp = "justified"
        elif is_blocked:
            disp = "open (blocked)"
        else:
            disp = "open"
        anchor = c.section_anchor or ""
        section = f"[`§`](proof-layer-analysis.md#{anchor})" if anchor else ""
        stmt = (c.statement or "").replace("|", "\\|")
        lines.append(f"| `{cid}` | {section} | {disp} | {stmt} |")
    CLAIMS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {CLAIMS_MD_PATH.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def die(msg: str) -> None:
    print(f"check_claims: {msg}", file=sys.stderr)
    sys.exit(2)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--emit-claims-md",
        action="store_true",
        help="regenerate docs/CLAIMS.md from the parsed graph",
    )
    ap.add_argument(
        "--manual",
        default=str(MANUAL_PATH),
        help="path to the reference manual (default: docs/proof-layer-analysis.md)",
    )
    args = ap.parse_args(argv)

    claims = parse_manual(Path(args.manual))
    rep = validate(claims)
    rc = print_report(rep)
    if args.emit_claims_md:
        emit_claims_md(rep)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
