#!/usr/bin/env python3
"""
Self-test for tools/check_claims.py. Exercises the error paths on
synthetic claim blocks to ensure the walker actually fails on the
failure modes we care about.

Run: python3 tools/test_check_claims.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_claims  # noqa: E402


def run(md: str) -> check_claims.Report:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tf:
        tf.write(md)
        path = tf.name
    claims = check_claims.parse_manual(Path(path))
    return check_claims.validate(claims)


def assert_has_error(rep: check_claims.Report, needle: str, label: str) -> None:
    joined = "\n".join(rep.errors)
    assert needle in joined, (
        f"[{label}] expected error containing {needle!r}\n"
        f"got errors:\n{joined or '(none)'}"
    )


def test_green_minimal():
    md = """## 2. Foo
```claim
id: C-FOO-HIDE
statement: x
justifiedBy:
  - ASSUMPTION(x)
status: justified
```
"""
    rep = run(md)
    assert not rep.errors, rep.errors
    assert len(rep.claims_by_id) == 1


def test_duplicate_id():
    md = """## X
```claim
id: C-X
justifiedBy:
  - ASSUMPTION(a)
status: justified
```
```claim
id: C-X
justifiedBy:
  - ASSUMPTION(b)
status: justified
```
"""
    rep = run(md)
    assert_has_error(rep, "duplicate claim id 'C-X'", "duplicate_id")


def test_unresolved_reference():
    md = """## X
```claim
id: C-A
combinator: AND_OF
justifiedBy:
  - C-MISSING
status: justified
```
"""
    rep = run(md)
    assert_has_error(rep, "unknown id 'C-MISSING'", "unresolved_ref")


def test_cycle():
    md = """## X
```claim
id: C-A
combinator: AND_OF
justifiedBy:
  - C-B
status: justified
```
```claim
id: C-B
combinator: AND_OF
justifiedBy:
  - C-A
status: justified
```
"""
    rep = run(md)
    assert_has_error(rep, "cycle in claim graph", "cycle")


def test_leaf_with_id_reference():
    md = """## X
```claim
id: C-A
justifiedBy:
  - C-B
status: justified
```
```claim
id: C-B
justifiedBy:
  - ASSUMPTION(x)
status: justified
```
"""
    rep = run(md)
    assert_has_error(rep, "mixes claim-id references", "leaf_with_id")


def test_internal_with_leaf_tag():
    md = """## X
```claim
id: C-A
combinator: AND_OF
justifiedBy:
  - PAPER(foo)
status: justified
```
"""
    rep = run(md)
    assert_has_error(rep, "mixes leaf tags", "internal_with_leaf")


def test_missing_code_file():
    md = """## X
```claim
id: C-A
justifiedBy:
  - CODE(src/does/not/exist.cu + test/also_missing.cu)
status: justified
```
"""
    rep = run(md)
    assert_has_error(rep, "missing file 'src/does/not/exist.cu'", "missing_code")


def test_justified_blocked_by_unjustified():
    md = """## X
```claim
id: C-LEAF
justifiedBy:
  - UNJUSTIFIED(todo)
status: open
```
```claim
id: C-ROOT
combinator: AND_OF
justifiedBy:
  - C-LEAF
status: justified
```
"""
    rep = run(md)
    assert_has_error(rep, "C-ROOT", "justified_blocked")
    assert_has_error(rep, "blocked by UNJUSTIFIED", "justified_blocked_msg")
    # and unjustified leaf shows up in the report
    assert any(cid == "C-LEAF" for cid, _ in rep.unjustified_leaves)
    # and upstream blocked set lists C-ROOT
    assert "C-ROOT" in rep.blocked_upstream["C-LEAF"]


def test_open_blocked_is_fine():
    md = """## X
```claim
id: C-LEAF
justifiedBy:
  - UNJUSTIFIED(todo)
status: open
```
```claim
id: C-ROOT
combinator: AND_OF
justifiedBy:
  - C-LEAF
status: open
```
"""
    rep = run(md)
    # no errors — open claim is allowed to be blocked
    assert not rep.errors, rep.errors


def test_unknown_top_level_key_dies():
    md = """## X
```claim
id: C-A
bogus: whatever
justifiedBy:
  - ASSUMPTION(x)
status: justified
```
"""
    try:
        run(md)
    except SystemExit as e:
        assert e.code == 2
        return
    raise AssertionError("expected SystemExit on unknown key")


def main() -> int:
    tests = [
        test_green_minimal,
        test_duplicate_id,
        test_unresolved_reference,
        test_cycle,
        test_leaf_with_id_reference,
        test_internal_with_leaf_tag,
        test_missing_code_file,
        test_justified_blocked_by_unjustified,
        test_open_blocked_is_fine,
        test_unknown_top_level_key_dies,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
