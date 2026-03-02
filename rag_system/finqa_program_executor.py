"""
FinQA-style program executor for multi-hop numerical reasoning.

Used by the RAG pipeline to execute model-generated programs (add/subtract/multiply/divide)
so the final answer is a precise number instead of free-form text. Aligns with:
- FinQA benchmark: execution accuracy (run program -> compare to exe_ans)
- Kaggle/industry: program synthesis + execute for numerical QA

Program format (single or multi-step):
  divide(9896, 23.6%)        -> 9896 / 0.236
  subtract(19201, 23280), divide(#0, 23280)  -> step0 = 19201-23280, step1 = step0/23280

Percentages: 23.6% -> 0.236. References: #0 = result of step 0, #1 = result of step 1.

Execution rules (gold-blind, no dataset leakage):
- Each operation produces one output; stored sequentially as #0, #1, #2, ... (0-based, immutable).
- References #k refer strictly to the k-th operation output. No overwrite.
- All arithmetic uses floating-point; no rounding or integer coercion until final answer.
- subtract(a,b) = a - b, divide(a,b) = a / b. Final answer = last step output only.
"""

import os
import re
from typing import Optional


def _looks_like_nested_program(s: str) -> bool:
    """True if s is a single op(...) call (e.g. subtract(959.2, 991.1)), so we can execute it as nested."""
    s = s.strip()
    for op in ("add", "subtract", "multiply", "divide"):
        if s.startswith(op + "(") and s.endswith(")"):
            inner = s[len(op) + 1 : -1]
            depth = 0
            for c in inner:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
            if depth == 0:
                return True
    return False


def _normalize_arg(s: str, step_results: list[float], _execute_finqa_program=None) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    # Reference to previous step: #0, #1, ... (0-based, immutable; never overwritten)
    ref = re.match(r"#(\d+)$", s)
    if ref:
        idx = int(ref.group(1))
        if idx < 0 or idx >= len(step_results):
            return None
        assert idx < len(step_results), "invariant: #k must reference existing step"
        val = step_results[idx]
        assert isinstance(val, (int, float)) and not isinstance(val, bool), "stored result must be numeric"
        return float(val)
    # Nested op call (e.g. divide(subtract(new,old), old)): execute and use result
    if _looks_like_nested_program(s) and _execute_finqa_program is not None:
        return _execute_finqa_program(s)
    # Percentage: 23.6% or 23.6
    s_clean = s.replace(",", "").replace("$", "").strip()
    if "%" in s_clean:
        try:
            num = float(re.sub(r"%\s*$", "", s_clean).strip())
            return num / 100.0
        except ValueError:
            return None
    try:
        return float(s_clean)
    except ValueError:
        return None


def _split_steps(program: str) -> list[str]:
    """Split program into steps by top-level comma or newline. Preserves paren depth."""
    steps_str = []
    depth = 0
    start = 0
    for i, c in enumerate(program):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and (c == "," or c == "\n"):
            part = program[start:i].strip()
            if part:
                steps_str.append(part)
            start = i + 1
    part = program[start:].strip()
    if part:
        steps_str.append(part)
    return steps_str


def _parse_one_step(step_str: str) -> Optional[tuple[str, str, str]]:
    """Parse one step like 'divide(9896, 23.6%)' into (op, arg1, arg2)."""
    step_str = step_str.strip()
    for op in ("add", "subtract", "multiply", "divide"):
        if step_str.startswith(op + "(") and step_str.endswith(")"):
            inner = step_str[len(op) + 1 : -1]
            parts = []
            depth = 0
            start = 0
            for i, c in enumerate(inner):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif c == "," and depth == 0:
                    parts.append(inner[start:i].strip())
                    start = i + 1
            parts.append(inner[start:].strip())
            if len(parts) == 2:
                return (op, parts[0], parts[1])
            return None
    return None


def execute_finqa_program(program: str) -> Optional[float]:
    """
    Execute a FinQA-style program string. Returns the final numerical result or None.

    Steps are split by top-level comma or newline. Each step output is stored as #0, #1, ...
    (0-based, immutable). All arithmetic is floating-point.

    Examples:
      execute_finqa_program("divide(9896, 23.6%)")  -> 41932.203389...
      execute_finqa_program("subtract(19201, 23280), divide(#0, 23280)")  -> -0.175...
    """
    if not program or not isinstance(program, str):
        return None
    program = program.strip()
    steps_str = _split_steps(program)

    step_results: list[float] = []
    debug = os.environ.get("RAG_DEBUG") == "1"
    for i, step_str in enumerate(steps_str):
        if not step_str:
            continue
        parsed = _parse_one_step(step_str)
        if not parsed:
            continue
        op, a_str, b_str = parsed
        a = _normalize_arg(a_str, step_results, _execute_finqa_program=execute_finqa_program)
        b = _normalize_arg(b_str, step_results, _execute_finqa_program=execute_finqa_program)
        if a is None or b is None:
            return None
        # Force float for all arithmetic (no int coercion / integer division)
        a, b = float(a), float(b)
        if op == "add":
            res = a + b
        elif op == "subtract":
            res = a - b
        elif op == "multiply":
            res = a * b
        elif op == "divide":
            if abs(b) < 1e-12:
                return None
            res = a / b
        else:
            return None
        assert isinstance(res, (int, float)), f"result must be numeric, got {type(res)}"
        assert not isinstance(res, bool), "result must not be bool (0/1 used as number)"
        res = float(res)
        step_results.append(res)
        step_idx = len(step_results) - 1
        if debug:
            print(f"[DEBUG] executor step {step_idx}: {step_str} -> #{step_idx}={res}")

    return float(step_results[-1]) if step_results else None


def _find_program_candidates(text: str) -> list[str]:
    """Find substrings that look like op(arg1, arg2) with balanced parens."""
    candidates = []
    for op in ("add", "subtract", "multiply", "divide"):
        start = 0
        while True:
            i = text.lower().find(op + "(", start)
            if i < 0:
                break
            j = i + len(op) + 1
            depth = 1
            while j < len(text) and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                candidates.append(text[i : j])
            start = i + 1
    return candidates


def _find_program_candidates_with_positions(text: str) -> list[tuple[str, int, int]]:
    """Find each op(...) substring with (program, start, end) so we can detect adjacent chains."""
    result = []
    text_lower = text.lower()
    for op in ("add", "subtract", "multiply", "divide"):
        start = 0
        while True:
            i = text_lower.find(op + "(", start)
            if i < 0:
                break
            j = i + len(op) + 1
            depth = 1
            while j < len(text) and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                result.append((text[i:j], i, j))
            start = i + 1
    return sorted(result, key=lambda x: x[1])


def _find_last_multistep_program(text: str) -> Optional[str]:
    """
    Find the last contiguous multi-step program (e.g. subtract(a,b), divide(#0,b))
    so we execute the full chain and return the final step result, not the first.
    """
    if not text:
        return None
    candidates = _find_program_candidates_with_positions(text)
    if not candidates:
        return None
    # Start from the last candidate (by position in text)
    last_prog, last_start, last_end = candidates[-1]
    # Extend left: include any candidate that ends with ") " or ")," immediately before our start
    parts = [last_prog]
    pos_before = last_start
    while True:
        # Find the candidate that ends immediately before our current span (gap is only ", " or ",")
        found = None
        best_e = -1
        for prog, s, e in candidates:
            if e >= pos_before:
                continue
            gap = text[e:pos_before]
            gap_clean = gap.strip().replace("\n", " ").strip()
            # Allow comma or newline/whitespace as step separator (model may output steps on new lines)
            if gap == "," or (gap.startswith(",") and gap.replace(",", "").strip() == "") or gap_clean == "":
                if e > best_e:
                    best_e = e
                    found = (prog, s, e)
        if not found:
            break
        prog, s, e = found
        parts.insert(0, prog)
        pos_before = s
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts)


def extract_and_execute_program(text: str) -> Optional[float]:
    """
    Find a FinQA-style program in model output and execute it.
    Looks for patterns like divide(9896, 23.6%) or subtract(a,b), divide(#0,b).
    For multi-step programs, executes the full chain and returns the **final** step result.
    """
    if not text:
        return None

    def _has_top_level_comma(s: str) -> bool:
        depth = 0
        for c in s:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "," and depth == 0:
                return True
        return False

    debug = os.environ.get("RAG_DEBUG") == "1"
    # If model output has a comma-separated chain (e.g. subtract(a,b), divide(#0,b)), run that first for final result
    multistep = _find_last_multistep_program(text)
    if multistep and _has_top_level_comma(multistep):
        if debug:
            print(f"[DEBUG] executor: extracted multistep program ({len(multistep)} chars): {multistep[:300]!r}{'...' if len(multistep) > 300 else ''}")
        result = execute_finqa_program(multistep)
        if result is not None:
            if debug:
                print(f"[DEBUG] executor: result={result}")
            return result
    candidates = _find_program_candidates(text)
    # Prefer longer (nested) expressions so divide(subtract(new,old), old) beats inner subtract(new,old)
    for program in sorted(candidates, key=len, reverse=True):
        if debug:
            print(f"[DEBUG] executor: trying candidate ({len(program)} chars): {program[:200]!r}{'...' if len(program) > 200 else ''}")
        result = execute_finqa_program(program)
        if result is not None:
            if debug:
                print(f"[DEBUG] executor: result={result}")
            return result
    if multistep:
        result = execute_finqa_program(multistep)
        if result is not None:
            return result
    result = execute_finqa_program(text.strip())
    if debug and result is not None:
        print(f"[DEBUG] executor: result (from text.strip())={result}")
    return result
