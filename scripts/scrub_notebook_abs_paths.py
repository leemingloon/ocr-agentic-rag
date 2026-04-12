"""
Replace absolute-path prefixes in *.ipynb JSON with empty string (repo-relative remainder)
or a short placeholder for temp / Python install dirs.

Uses Path.home() so it works on the machine that runs it (no hardcoded username).
"""
from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    home = Path.home()

    def json_escaped(path: Path, trailing: str) -> str:
        """Notebook JSON doubles each backslash; match that form."""
        s = str(path) + trailing
        return s.replace("\\", "\\\\")

    replace_empty: list[str] = []
    for rel in (
        ("OneDrive", "Desktop", "Job_search_2026", "ocr-agentic-rag"),
        ("Desktop", "Job_search_2026", "ocr-agentic-rag"),
    ):
        p = home.joinpath(*rel)
        replace_empty.append(json_escaped(p, os.sep))
        replace_empty.append(p.as_posix() + "/")
        # Some tools emit lowercase drive letter in JSON.
        ps = str(p)
        if len(ps) > 1 and ps[1] == ":":
            p2 = Path(ps[0].swapcase() + ps[1:])
            replace_empty.append(json_escaped(p2, os.sep))

    py311 = home / "AppData" / "Local" / "Programs" / "Python" / "Python311"
    replace_empty.extend([json_escaped(py311, os.sep), py311.as_posix() + "/"])
    ps = str(py311)
    if len(ps) > 1 and ps[1] == ":":
        p2 = Path(ps[0].swapcase() + ps[1:])
        replace_empty.extend([json_escaped(p2, os.sep)])

    tmp = home / "AppData" / "Local" / "Temp"
    replace_tmp = [(json_escaped(tmp, os.sep), "__TEMP__/"), (tmp.as_posix() + "/", "__TEMP__/")]
    ts = str(tmp)
    if len(ts) > 1 and ts[1] == ":":
        tmp2 = Path(ts[0].swapcase() + ts[1:])
        replace_tmp.append((json_escaped(tmp2, os.sep), "__TEMP__/"))

    replace_empty = list(dict.fromkeys(replace_empty))

    n_changed = 0
    for nb in (repo / "notebooks").rglob("*.ipynb"):
        text = nb.read_text(encoding="utf-8")
        orig = text
        for pre in replace_empty:
            text = text.replace(pre, "")
        for old, new in replace_tmp:
            text = text.replace(old, new)
        if text != orig:
            nb.write_text(text, encoding="utf-8")
            n_changed += 1
            print("scrubbed", nb.relative_to(repo))
    print("done,", n_changed, "notebooks updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
