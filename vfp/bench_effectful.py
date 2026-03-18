"""Bench workflow using effectful's LLM module with algebraic effects.

Represents sketcher calls as effectful Tools, and the LLM lemma-implementation
loop as an effectful Template.  The LLM can invoke the sketcher tools during
its proof search.

Usage:
    python bench_effectful.py
    python bench_effectful.py --file bench/binary_search_solution.dfy
    python bench_effectful.py --file bench/reverse_solution.dfy --lemma reverseLength
    python bench_effectful.py --model openai/gpt-5
    USE_SKETCHERS=true python bench_effectful.py

Environment variables:
    USE_SKETCHERS: Set to 'false' to disable sketcher tools (default: true)
    LLM_MODEL: litellm model string (default: from LLM_MODEL env or "vertex_ai/gemini-2.5-flash-lite")
    MAX_VERIFICATION_ATTEMPTS: Max execute() failures per lemma before stopping (default: 5)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import bench_driver
import driver
import litellm
import sketcher
driver.XP_DEBUG = False
from dotenv import load_dotenv
from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# Monkey-patch pydantic.create_model to fix effectful bug where bare types
# (e.g., `value=str` or `value=ThinkLemma`) are passed as kwargs instead of
# tuple format `(type, ...)`. Pydantic interprets bare types as default values.
import pydantic as _pydantic
_orig_create_model = _pydantic.create_model
def _fixed_create_model(__model_name, **kwargs):
    fixed = {}
    for k, v in kwargs.items():
        if k.startswith('__'):
            fixed[k] = v
        elif isinstance(v, type):
            fixed[k] = (v, ...)
        else:
            fixed[k] = v
    return _orig_create_model(__model_name, **fixed)
_pydantic.create_model = _fixed_create_model
from error_parser import parse_dafny_errors, extract_proof_obligations, format_errors_structured, format_proof_obligations
from fine import format_errors
from pydantic import BaseModel, Field

_repo_root = Path(__file__).resolve().parent.parent

load_dotenv(_repo_root / ".env")
if not os.environ.get("DAFNY"):
    _dafny_dll = _repo_root / "dafny" / "Binaries" / "Dafny.dll"
    if _dafny_dll.exists():
        os.environ["DAFNY"] = str(_dafny_dll)

_orig_completion = litellm.completion
def _traced_completion(*args, **kwargs):
    global _llm_call_count
    model = kwargs.get("model", args[0] if args else "unknown")
    kwargs.setdefault("timeout", 600)  # 10-minute timeout for structured output + multi-turn tool calls
    print(f"[LLM] Querying {model} ...", flush=True)
    _llm_call_count += 1
    result = _orig_completion(*args, **kwargs)
    print(f"[LLM] Response received from {model}", flush=True)
    return result
litellm.completion = _traced_completion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


USE_SKETCHERS = os.environ.get('USE_SKETCHERS', 'true').lower() != 'false'
LLM_MODEL = os.environ.get('LLM_MODEL', 'vertex_ai/gemini-2.5-flash-lite')
MAX_VERIFICATION_ATTEMPTS = int(os.environ.get('MAX_VERIFICATION_ATTEMPTS', '5'))
FORCE_LLM = False
DEFAULT_OUT_PATH = str(_repo_root / "vfp" / "bench_effectful_latest.json")
DEFAULT_PERSISTENCE_OUT_PATH = str(_repo_root / "vfp" / "bench_effectful_persistence_latest.jsonl")
DEFAULT_SAMPLES_OUT_PATH = str(_repo_root / "vfp" / "bench_effectful_samples_latest.jsonl")
OUT_PATH = DEFAULT_OUT_PATH
PERSISTENCE_OUT_PATH = DEFAULT_PERSISTENCE_OUT_PATH
SAMPLES_OUT_PATH = DEFAULT_SAMPLES_OUT_PATH

provider = LiteLLMProvider(model=LLM_MODEL)

# Per-lemma counter: failed execute() calls; reset at start of each lemma
_execute_attempt_count = 0

# Per-lemma sampling state used to write per-iteration traces.
_current_lemma_name: Optional[str] = None
_current_sample_index = 0
_current_last_code: str = ""

# Per-lemma tool call counter: tool_name -> count
_tool_calls: dict[str, int] = {}
_llm_call_count = 0

def _track_tool(tool_name: str) -> None:
    _tool_calls[tool_name] = _tool_calls.get(tool_name, 0) + 1


def _append_sample_event(event_type: str, **payload: Any) -> None:
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "lemma": _current_lemma_name,
        "sample_index": _current_sample_index,
        "llm_model": LLM_MODEL,
        "persistence_memory": list(persistence_memory),
        **payload,
    }
    try:
        with open(SAMPLES_OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception as e:
        print(f"failed to append sample event to {SAMPLES_OUT_PATH}: {e}")

# ---------------------------------------------------------------------------
# Tools – effectful wrappers around the sketcher / driver helpers
# ---------------------------------------------------------------------------

class VerificationError(Exception):
    """Exception raised when Dafny verification fails."""
    def __init__(self, errors: str):
        self.errors = errors
        super().__init__(errors)


# DafnyTools - Tools for working with Dafny programs and proofs
# These tools are grouped together conceptually for the LLM to discover and use

@Tool.define
def execute(dafny_program: str) -> str:
    """Execute/verify a Dafny program.
    
    This tool verifies the entire program and raises an error if verification fails.
    When used with RetryLLMHandler, the LLM will be retried automatically
    when verification errors occur. After MAX_VERIFICATION_ATTEMPTS failed attempts
    for this lemma, returns a message instead of raising so the loop stops.

    Args:
        program: Full Dafny program source.

    Returns:
        "Verification succeeded" if the program verifies without errors.
        
    Raises:
        VerificationError: If there are verification errors (until attempt limit reached).
    """
    global _execute_attempt_count, _current_sample_index
    print("[TOOL] execute()", flush=True)
    _track_tool("execute")
    _current_sample_index += 1
    errs = sketcher.list_errors_for_method(dafny_program, None)
    if errs:
        _execute_attempt_count += 1
        error_msg = format_errors(errs)
        _append_sample_event(
            "execute_failed",
            execute_attempt=_execute_attempt_count,
            code=_current_last_code,
            program=dafny_program,
            errors=error_msg,
        )
        if _execute_attempt_count >= MAX_VERIFICATION_ATTEMPTS:
            print(f"[TOOL] execute() max attempts ({MAX_VERIFICATION_ATTEMPTS}) reached, stopping retries", flush=True)
            return (
                f"Max verification attempts ({MAX_VERIFICATION_ATTEMPTS}) reached. "
                f"Last errors:\n{error_msg}\n\n"
                "Return your current best proof body now: a single block starting with // BEGIN DAFNY and ending with // END DAFNY."
            )
        raise VerificationError(error_msg)
    _append_sample_event(
        "execute_succeeded",
        execute_attempt=_execute_attempt_count + 1,
        code=_current_last_code,
        program=dafny_program,
    )
    return "Verification succeeded"


@Tool.define
def induction_sketch(dafny_program: str, method_name: str) -> str:
    """Generate an induction proof sketch for a lemma.

    Args:
        program: Full Dafny program source (the lemma body should be empty).
        method_name: Name of the lemma.

    Returns:
        The induction sketch body (Dafny code to place inside the lemma).
    """
    print(f"[TOOL] induction_sketch(method_name={method_name!r})", flush=True)
    _track_tool("induction_sketch")
    result = sketcher.sketch_induction(dafny_program, method_name)
    return result or ""


@Tool.define
def insert_body(lemma_name: str, original_dafny_program: str, body: str) -> str:
    """Insert a proof body into a lemma and return the full program.

    The lemma is identified by *lemma_name* inside *original_program*.
    *body* is the code to place inside the lemma braces.

    Args:
        lemma_name: Name of the lemma to fill in.
        original_program: The original Dafny program (with empty lemma).
        body: The proof body to insert.

    Returns:
        The full Dafny program with the body inserted, or an error message.
    """
    print(f"[TOOL] insert_body(lemma_name={lemma_name!r})", flush=True)
    _track_tool("insert_body")
    global _current_last_code
    _current_last_code = body
    done = sketcher.sketch_done(original_dafny_program)
    if done is None:
        out = "Error: could not resolve program metadata"
        _append_sample_event("insert_body_error", code=body, message=out)
        return out
    lemma = next((x for x in done if x['name'] == lemma_name), None)
    if lemma is None:
        out = f"Error: lemma '{lemma_name}' not found in program"
        _append_sample_event("insert_body_error", code=body, message=out)
        return out
    result = driver.insert_program_todo(lemma, original_dafny_program, body)
    _append_sample_event("insert_body", code=body, program=result)
    return result


@Tool.define
def verify_method(dafny_program: str, method_name: str) -> str:
    """Verify ONLY a specific method/lemma (preferred over execute for single lemmas).

    Args:
        program: Full Dafny program source.
        method_name: Name of the method/lemma to verify.

    Returns:
        Success or failure message with errors.
    """
    print(f"[TOOL] verify_method(method_name={method_name!r})", flush=True)
    _track_tool("verify_method")
    errs = sketcher.list_errors_for_method(dafny_program, method_name)
    if errs:
        return f"Verification failed for {method_name}:\n{format_errors(errs)}"
    return f"Verification succeeded for {method_name}"


@Tool.define
def verify_isolated(dafny_program: str, lemma_name: str) -> str:
    """Verify a lemma in isolation by stubbing all other lemma bodies with 'assume false;'.

    Use this when other unproved lemmas cause cascading verification failures.

    Args:
        program: Full Dafny program source.
        lemma_name: Name of the lemma to verify in isolation.

    Returns:
        Success or failure message.
    """
    print(f"[TOOL] verify_isolated(lemma_name={lemma_name!r})", flush=True)
    _track_tool("verify_isolated")
    done = sketcher.sketch_done(dafny_program)
    if not done:
        errs = sketcher.list_errors_for_method(dafny_program, lemma_name)
        if errs:
            return f"Verification failed for {lemma_name}:\n{format_errors(errs)}"
        return f"Verification succeeded for {lemma_name}"

    lines = dafny_program.splitlines(keepends=True)
    for item in sorted(done, key=lambda x: x.get('startLine', 0), reverse=True):
        if item.get('name') == lemma_name:
            continue
        if item.get('type') != 'lemma' or item.get('status') != 'done':
            continue
        start = item.get('insertLine', 0)
        end = item.get('endLine', 0)
        if start <= 0 or end <= 0 or start > end:
            continue
        body_start = start - 1
        body_end = end
        joined = ''.join(lines[body_start:body_end])
        brace_open = joined.find('{')
        if brace_open == -1:
            continue
        prefix = joined[:brace_open]
        lines[body_start:body_end] = [prefix + "{ assume false; }\n"]

    stubbed = ''.join(lines)
    errs = sketcher.list_errors_for_method(stubbed, lemma_name)
    if errs:
        return f"Verification failed for {lemma_name} (isolated):\n{format_errors(errs)}"
    return f"Verification succeeded for {lemma_name} (isolated)"


@Tool.define
def detect_axiom(dafny_program: str, lemma_name: str) -> str:
    """Check if a lemma depends on bodyless (uninterpreted) functions and is thus an axiom.

    Args:
        program: Full Dafny program source.
        lemma_name: Name of the lemma to check.

    Returns:
        JSON with is_axiom flag and reason.
    """
    import re as _re
    print(f"[TOOL] detect_axiom(lemma_name={lemma_name!r})", flush=True)
    _track_tool("detect_axiom")
    done = sketcher.sketch_done(dafny_program)
    if not done:
        return json.dumps({"lemma": lemma_name, "is_axiom": "unknown", "reason": "could not parse declarations"})

    lemma = next((x for x in done if x.get('name') == lemma_name), None)
    if lemma is None:
        return json.dumps({"lemma": lemma_name, "is_axiom": "unknown", "reason": "lemma not found"})

    bodyless_fns = [item['name'] for item in done
                    if item.get('type') == 'function' and item.get('status') != 'done']

    lines = dafny_program.splitlines()
    start = lemma.get('startLine', 1) - 1
    end = lemma.get('endLine', lemma.get('insertLine', start + 1))
    lemma_sig = '\n'.join(lines[start:end])

    is_axiom_attr = '{:axiom}' in lemma_sig
    referenced_bodyless = [fn for fn in bodyless_fns if fn in lemma_sig]

    if is_axiom_attr:
        result = {"lemma": lemma_name, "is_axiom": True,
                  "reason": "Has {:axiom} attribute"}
    elif referenced_bodyless:
        result = {"lemma": lemma_name, "is_axiom": True,
                  "reason": f"Depends on bodyless function(s): {', '.join(referenced_bodyless)}"}
    else:
        result = {"lemma": lemma_name, "is_axiom": False,
                  "reason": "All referenced functions have bodies. This lemma should be provable."}
    return json.dumps(result, indent=2)


@Tool.define
def parse_errors_tool(dafny_program: str, method_name: str = "") -> str:
    """Parse Dafny verification errors into structured format with categories and suggestions.

    Categories: postcondition, precondition, assertion, decreases, timeout, calc, exists, forall.
    Each error includes an actionable suggestion for fixing it.

    Args:
        program: Full Dafny program source.
        method_name: Optional method/lemma name to filter errors for.

    Returns:
        Structured error report with proof obligations.
    """
    print(f"[TOOL] parse_errors(method_name={method_name!r})", flush=True)
    _track_tool("parse_errors_tool")
    raw = sketcher._show_errors_for_method_core(dafny_program, method_name or None)
    if not raw:
        return "No errors found."
    errors = parse_dafny_errors(raw)
    if not errors:
        return "Verification succeeded (no errors parsed)."
    parts = ["=== ERRORS ===", format_errors_structured(errors)]
    obligations = extract_proof_obligations(errors, method_name)
    if obligations:
        parts.extend(["", "=== PROOF OBLIGATIONS ===", format_proof_obligations(obligations)])
    return "\n".join(parts)


@Tool.define
def counterexamples(dafny_program: str, method_name: str) -> str:
    """Get counterexamples for a lemma to understand why it fails.

    Shows concrete input values that make the lemma's postcondition false.

    Args:
        program: Full Dafny program source.
        method_name: Name of the lemma.

    Returns:
        List of counterexample conditions, or a message if none found.
    """
    print(f"[TOOL] counterexamples(method_name={method_name!r})", flush=True)
    _track_tool("counterexamples")
    results = sketcher.sketch_counterexamples(dafny_program, method_name)
    if isinstance(results, str):
        return results
    if results:
        return "Counterexamples (conditions where the lemma fails):\n" + \
               "\n".join(f"  {i}. {ce}" for i, ce in enumerate(results, 1))
    return "No counterexamples found (lemma may be correct)."


@Tool.define
def find_relevant(dafny_program: str, lemma_name: str) -> str:
    """Find declarations (functions, lemmas, predicates) relevant to proving a lemma.

    Analyzes the lemma's signature and ensures clauses to find referenced declarations.

    Args:
        program: Full Dafny program source.
        lemma_name: Name of the lemma to analyze.

    Returns:
        List of relevant declarations with their types and body status.
    """
    import re as _re
    print(f"[TOOL] find_relevant(lemma_name={lemma_name!r})", flush=True)
    _track_tool("find_relevant")
    done = sketcher.sketch_done(dafny_program)
    lines = dafny_program.splitlines()
    target = next((x for x in (done or []) if x.get('name') == lemma_name), None)
    if not target:
        return f"Lemma '{lemma_name}' not found."

    start = target.get('startLine', 1) - 1
    end = target.get('endLine', start + 1)
    sig_region = '\n'.join(lines[start:end])
    identifiers = set(_re.findall(r'\b([A-Za-z_]\w*)\b', sig_region))
    keywords = {'lemma', 'function', 'method', 'predicate', 'requires', 'ensures',
                'decreases', 'modifies', 'reads', 'returns', 'var', 'if', 'else',
                'then', 'match', 'case', 'forall', 'exists', 'true', 'false',
                'int', 'nat', 'bool', 'string', 'seq', 'set', 'map', 'multiset',
                'ghost', 'opaque', 'old', 'fresh', 'allocated', 'null', 'this',
                'assert', 'assume', 'calc', 'reveal', 'while', 'for', 'return',
                lemma_name}
    identifiers -= keywords

    relevant = []
    for item in (done or []):
        name = item.get('name', '')
        if name == lemma_name or name in keywords:
            continue
        if name in identifiers:
            kind = item.get('type', '?')
            has_body = item.get('status') == 'done'
            s = item.get('startLine', 1) - 1
            sig = lines[s].strip() if s < len(lines) else ''
            relevant.append(f"  [{kind}] {name} ({'has body' if has_body else 'NO BODY'}): {sig}")

    if relevant:
        return f"Declarations relevant to '{lemma_name}':\n" + "\n".join(relevant)
    return f"No directly relevant declarations found for '{lemma_name}'."


@Tool.define
def search_lemmas(dafny_program: str, pattern: str) -> str:
    """Search for lemmas/functions by name pattern or keyword in a program.

    Args:
        program: Full Dafny program source.
        pattern: Search pattern (case-insensitive substring match).

    Returns:
        Matching declarations with line numbers.
    """
    import re as _re
    print(f"[TOOL] search_lemmas(pattern={pattern!r})", flush=True)
    _track_tool("search_lemmas")
    pat = pattern.lower()
    results = []
    lines = dafny_program.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _re.match(r'^(lemma|function|predicate|ghost\s+function|ghost\s+method|method)\b', stripped):
            sig_lines = [stripped]
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j].strip()
                if next_line.startswith(('requires', 'ensures', 'decreases', 'modifies', 'reads')):
                    sig_lines.append(next_line)
                elif next_line == '{' or next_line == '' or _re.match(r'^(lemma|function|predicate|ghost|method|class|module|datatype)\b', next_line):
                    break
                else:
                    sig_lines.append(next_line)
                    if '{' in next_line:
                        break
            full_sig = ' '.join(sig_lines)
            if pat in full_sig.lower():
                results.append(f"  Line {i+1}: {full_sig.rstrip('{').strip()}")

    if results:
        return f"Found {len(results)} matching declaration(s):\n" + "\n".join(results)
    return f"No declarations matching '{pattern}' found."


@Tool.define
def analyze_induction(dafny_program: str, lemma_name: str) -> str:
    """Analyze the induction structure of a lemma: what to induct on, base/recursive cases.

    Combines parameter analysis with the sketcher's induction sketch.

    Args:
        program: Full Dafny program source.
        lemma_name: Name of the lemma to analyze.

    Returns:
        Induction analysis with candidates and sketch.
    """
    import re as _re
    print(f"[TOOL] analyze_induction(lemma_name={lemma_name!r})", flush=True)
    _track_tool("analyze_induction")

    sketch = sketcher.sketch_induction(dafny_program, lemma_name)
    if not sketch or "Error" in sketch:
        sketch = sketcher.sketch_induction(dafny_program, lemma_name, shallow=True)

    done = sketcher.sketch_done(dafny_program)
    lines = dafny_program.splitlines()
    lemma = next((x for x in (done or []) if x.get('name') == lemma_name), None)

    analysis = [f"=== Induction Analysis for {lemma_name} ==="]
    if lemma:
        start = lemma.get('startLine', 1) - 1
        insert = lemma.get('insertLine', start + 1)
        sig = '\n'.join(lines[start:insert])
        analysis.append(f"\nSignature:\n{sig}")
        params = _re.findall(r'(\w+)\s*:\s*(\w[\w<>,._ ]*)', sig)
        if params:
            analysis.append(f"\nParameters: {', '.join(f'{n}: {t}' for n, t in params)}")
            datatypes = {_re.match(r'^\s*datatype\s+(\w+)', l).group(1)
                         for l in lines if _re.match(r'^\s*datatype\s+(\w+)', l)}
            candidates = [(n, t) for n, t in params if t in datatypes or t in ('nat', 'Nat')]
            if candidates:
                analysis.append(f"Induction candidates: {', '.join(f'{n} ({t})' for n, t in candidates)}")

    if sketch:
        analysis.append(f"\nInduction sketch:\n{sketch}")
    else:
        analysis.append("\nNo induction sketch available. Try manual case analysis.")

    return '\n'.join(analysis)


@Tool.define
def check_calc(dafny_program: str, lemma_name: str) -> str:
    """Check each step of a calc block individually to find which step fails.

    Isolates each transition (A op B) in a calc block and verifies it separately.

    Args:
        program: Full Dafny program source.
        lemma_name: Name of the lemma containing the calc block.

    Returns:
        Per-step pass/fail report.
    """
    print(f"[TOOL] check_calc(lemma_name={lemma_name!r})", flush=True)
    _track_tool("check_calc")
    from calc_checker import check_calc_steps
    return check_calc_steps(dafny_program, lemma_name)


@Tool.define
def dependency_order(dafny_program: str) -> str:
    """Show the optimal order to solve lemmas based on their dependencies.

    Analyzes which lemmas depend on which and returns a topological ordering.

    Args:
        program: Full Dafny program source.

    Returns:
        Ordered list of lemma names to solve.
    """
    print("[TOOL] dependency_order()", flush=True)
    _track_tool("dependency_order")
    from dependency_graph import get_solve_order
    order = get_solve_order(dafny_program)
    if order:
        return "Recommended solve order:\n" + "\n".join(f"  {i}. {n}" for i, n in enumerate(order, 1))
    return "No lemmas found to order."


@Tool.define
def dependency_info(dafny_program: str, lemma_name: str) -> str:
    """Show what a specific lemma depends on (functions, other lemmas, predicates).

    Args:
        program: Full Dafny program source.
        lemma_name: Name of the lemma to analyze.

    Returns:
        List of dependencies with their types and body status.
    """
    print(f"[TOOL] dependency_info(lemma_name={lemma_name!r})", flush=True)
    _track_tool("dependency_info")
    from dependency_graph import format_dependency_info
    return format_dependency_info(dafny_program, lemma_name)


@Tool.define
def inspect_function(dafny_program: str, name: str) -> str:
    """Check if a function/lemma is opaque, axiom, or has a body.

    Args:
        program: Full Dafny program source.
        name: Name of the declaration to inspect.

    Returns:
        JSON with properties: has_body, is_axiom, is_opaque, is_ghost, etc.
    """
    print(f"[TOOL] inspect_function(name={name!r})", flush=True)
    _track_tool("inspect_function")
    done = sketcher.sketch_done(dafny_program)
    item = next((x for x in (done or []) if x.get('name') == name), None)
    if item is None:
        return f"Declaration '{name}' not found"

    lines = dafny_program.splitlines()
    start_line = item.get('startLine', 0) - 1
    end_line = item.get('endLine', 0) - 1
    decl_text = '\n'.join(lines[start_line:end_line+1]) if start_line >= 0 else ''

    info = {
        'name': name,
        'type': item.get('type', '?'),
        'status': item.get('status', '?'),
        'has_body': item.get('status') == 'done',
        'is_axiom': '{:axiom}' in decl_text,
        'is_opaque': 'opaque' in decl_text.split(name)[0] if name in decl_text else False,
        'is_ghost': 'ghost' in decl_text.split(name)[0] if name in decl_text else False,
    }
    if info['type'] == 'function' and not info['has_body']:
        info['note'] = 'No body (uninterpreted). Lemmas about it may be axioms.'
    if info['is_axiom']:
        info['note'] = 'This is an axiom — cannot be proved, only assumed.'
    if info['is_opaque']:
        info['note'] = f'Opaque. Need "reveal {name}();" in proof to use its definition.'
    return json.dumps(info, indent=2)


@Tool.define
def list_declarations(dafny_program: str) -> str:
    """List all declarations (lemmas, functions, methods) with their signatures and flags.

    Args:
        program: Full Dafny program source.

    Returns:
        List of declarations with [axiom]/[no-body] flags.
    """
    import re as _re
    print("[TOOL] list_declarations()", flush=True)
    _track_tool("list_declarations")
    done = sketcher.sketch_done(dafny_program)
    lines = dafny_program.splitlines()
    if not done:
        results = []
        for line in lines:
            stripped = line.strip()
            if _re.match(r'^(lemma|function|method|predicate|ghost\s+function|ghost\s+method)\b', stripped):
                results.append(stripped.rstrip('{').strip())
        return '\n'.join(results) if results else "No declarations found."

    parts = []
    for item in done:
        kind = item.get('type', '?')
        name = item.get('name', '?')
        has_body = item.get('status') == 'done'
        start = item.get('startLine', 0)
        axiom = '{:axiom}' in lines[start-1] if start > 0 and start <= len(lines) else False
        flags = []
        if axiom:
            flags.append('axiom')
        if not has_body:
            flags.append('no-body')
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        insert = item.get('insertLine', start)
        sig_lines = lines[start-1:insert] if start > 0 else []
        sig = ' '.join(l.strip() for l in sig_lines).rstrip('{').strip()
        parts.append(f"[{kind}] {name}{flag_str}: {sig}")
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Template – the LLM-powered lemma implementer
# ---------------------------------------------------------------------------

class ThinkLemma(BaseModel):
    think: str = Field(description="Your thoughts and plans for the lemma.")
    code: str = Field(description="The code to insert into the lemma.")

@Template.define
def implement_lemma(dafny_source: str, lemma_name: str, errors: str) -> ThinkLemma:
    """You are implementing a lemma in a Dafny program.

The current program is:
{dafny_source}

The lemma to implement is {lemma_name}.

The current verification errors are:
{errors}

Available tools:
- VERIFICATION: `execute`, `verify_method`, `verify_isolated`, `parse_errors_tool`
- ANALYSIS: `detect_axiom`, `inspect_function`, `list_declarations`, `counterexamples`, `find_relevant`, `search_lemmas`
- PROOF: `induction_sketch`, `analyze_induction`, `insert_body`, `check_calc`
- DEPENDENCIES: `dependency_order`, `dependency_info`
- PERSISTENCE: `read_persistence_memory`, `write_to_persistence_memory`

Your goal:
1. Read persistence memory for insights from previous lemmas.
2. Use `detect_axiom` to check if the lemma is provable.
3. Use `find_relevant` and `inspect_function` to understand available functions/lemmas.
4. Use `analyze_induction` or `induction_sketch` for a proof sketch.
5. Use `insert_body` to insert your proposed proof body.
6. Use `verify_method` (preferred) or `execute` to verify. If it fails, use `parse_errors_tool` for structured error analysis.
7. If other unproved lemmas cause cascading failures, use `verify_isolated`.
8. If your proof uses a calc block, use `check_calc` to find which step fails.
9. Use `counterexamples` if you need to understand what cases fail.
10. When verification succeeds, return ONLY the final proof body (without the outer braces), starting with "// BEGIN DAFNY" and ending with "// END DAFNY".
11. If execute() returns a message saying "Max verification attempts reached", return your current best proof body in the same format.
12. If you discover the lemma is an axiom / unprovable, return "// AXIOM".

Write useful insights to persistence memory for future lemmas.
"""
    raise NotHandled


persistence_memory: List[str] = []


def _append_persistence_event(memory: str):
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "memory": memory,
    }
    try:
        with open(PERSISTENCE_OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception as e:
        print(f"failed to append persistence memory to {PERSISTENCE_OUT_PATH}: {e}")

@Tool.define
def read_persistence_memory() -> str:
    """Read the persistence memory. Find things that previous run found to be useful."""
    print("[TOOL] Reading from persistence memory: ", persistence_memory, flush=True)
    _track_tool("read_persistence_memory")
    return "\n".join(persistence_memory)

@Tool.define
def write_to_persistence_memory(memory: str) -> str:
    """Write a memory to the persistence memory, write things that are useful to remember."""
    print("[TOOL] Writing to persistence memory: ", memory, flush=True)
    _track_tool("write_to_persistence_memory")
    persistence_memory.append(memory)
    _append_persistence_event(memory)
    return "Memory written to persistence memory"


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def lemma1(lemma, p, stats):
    """Process a single lemma using effectful tools + LLM template."""
    global _current_lemma_name, _current_sample_index, _current_last_code
    init_p = p
    name = lemma['name']

    _current_lemma_name = name
    _current_sample_index = 0
    _current_last_code = ""
    _tool_calls.clear()
    _llm_call_count = 0
    print('lemma', name)

    # Step -1: auto-detect axioms (bodyless functions)
    try:
        import re as _re
        done = sketcher.sketch_done(init_p)
        if done:
            bodyless_fns = [x['name'] for x in done
                            if x.get('type') == 'function' and x.get('status') != 'done']
            _done_names = {x['name'] for x in done}
            _lines = init_p.splitlines()
            for _i, _line in enumerate(_lines):
                _m = _re.match(r'^\s*(?:ghost\s+)?function\s+(\w+)', _line)
                if _m and _m.group(1) not in bodyless_fns and _m.group(1) not in _done_names:
                    _has_body = False
                    _depth = 0
                    for _j in range(_i, min(_i + 20, len(_lines))):
                        for _ch in _lines[_j]:
                            if _ch == '{': _depth += 1
                            elif _ch == '}': _depth -= 1
                        if _depth > 0:
                            _has_body = True
                            break
                        if _j > _i and _lines[_j].strip() == '':
                            break
                        if _j > _i and _re.match(r'^\s*(lemma|method|function|predicate|class|module)\b', _lines[_j]):
                            break
                    if not _has_body:
                        bodyless_fns.append(_m.group(1))
            lines = init_p.splitlines()
            start = lemma.get('startLine', 1) - 1
            end = lemma.get('endLine', lemma.get('insertLine', start + 1))
            lemma_sig = '\n'.join(lines[start:end])
            referenced = [fn for fn in bodyless_fns if fn in lemma_sig]
            if referenced:
                print(f"[AXIOM] {name} depends on bodyless function(s): {', '.join(referenced)} — skipping")
                stats[name] = -2
                save_run_state(stats)
                return
    except Exception as e:
        print(f"[AXIOM] Detection failed for {name}: {e}")

    # Step 0: try empty proof
    xp = driver.insert_program_todo(lemma, init_p, "")
    e = sketcher.list_errors_for_method(xp, name)
    if not e and not FORCE_LLM:
        print("empty proof works")
        stats[name] = -1
        return
    if not e and FORCE_LLM:
        print("empty proof works (continuing due to --force-llm)")

    # Step 1: try induction sketch (non-LLM)
    if USE_SKETCHERS:
        ix = sketcher.sketch_induction(xp, name)
        p_ind = driver.insert_program_todo(lemma, init_p, ix)
        e_ind = sketcher.list_errors_for_method(p_ind, name)
        if not e_ind and not FORCE_LLM:
            print("inductive proof sketch works")
            stats[name] = 0
            return
        if not e_ind and FORCE_LLM:
            print("inductive proof sketch works (continuing due to --force-llm)")
        # Start LLM loop from the induction-sketched version
        p, e = p_ind, e_ind
    else:
        p = xp

    # Step 2: LLM repair mediated by effectful (retries until success or MAX_VERIFICATION_ATTEMPTS)
    global _execute_attempt_count
    _execute_attempt_count = 0
    with handler(provider), handler(RetryLLMHandler()):
        r = implement_lemma(p, name, format_errors(e))

    x = r.code
    _current_last_code = x or ""
    _append_sample_event("llm_returned", code=_current_last_code, think=r.think,
                         tool_calls=dict(_tool_calls), llm_calls=_llm_call_count)
    if x is None:
        print("LLM did not return valid Dafny")
        stats[name] = 2
        return

    # Verify the final result
    p_final = driver.insert_program_todo(lemma, init_p, x)
    e_final = sketcher._list_errors_for_method_core(p_final, name)
    if not e_final:
        print("LLM repair succeeded")
        stats[name] = 1
        stats['proof_' + name] = x
        stats['calls_' + name] = {"tool_calls": dict(_tool_calls), "llm_calls": _llm_call_count}
        _append_sample_event("lemma_solved", code=x, program=p_final,
                             tool_calls=dict(_tool_calls), llm_calls=_llm_call_count)
    else:
        print("LLM repair failed - still has errors")
        stats[name] = 2
        stats['failed_proof_' + name] = x
        stats['calls_' + name] = {"tool_calls": dict(_tool_calls), "llm_calls": _llm_call_count}
        _append_sample_event("lemma_unsolved", code=x, program=p_final, errors=format_errors(e_final),
                             tool_calls=dict(_tool_calls), llm_calls=_llm_call_count)


# ---------------------------------------------------------------------------
# Stats reporting (same as bench_feedback.py)
# ---------------------------------------------------------------------------

def print_summary_stats(stats):
    print('total for empty proof works:',
          len([v for v in stats.values() if isinstance(v, int) and v == -1]))
    print('total for inductive proof sketch works:',
          len([v for v in stats.values() if isinstance(v, int) and v == 0]))
    print('total for LLM repair loop works:',
          len([v for v in stats.values() if isinstance(v, int) and v == 1]))
    print('total for unsolved:',
          len([v for v in stats.values() if isinstance(v, int) and v == 2]))


def _summary_counts(stats):
    return {
        "empty_proof_works": len([v for v in stats.values() if isinstance(v, int) and v == -1]),
        "inductive_proof_sketch_works": len([v for v in stats.values() if isinstance(v, int) and v == 0]),
        "llm_repair_loop_works": len([v for v in stats.values() if isinstance(v, int) and v == 1]),
        "unsolved": len([v for v in stats.values() if isinstance(v, int) and v == 2]),
    }


def save_run_state(stats):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "llm_model": LLM_MODEL,
            "use_sketchers": USE_SKETCHERS,
            "max_verification_attempts": MAX_VERIFICATION_ATTEMPTS,
            "force_llm": FORCE_LLM,
            "persistence_out": PERSISTENCE_OUT_PATH,
            "samples_out": SAMPLES_OUT_PATH,
        },
        "summary": _summary_counts(stats),
        "persistence_memory": persistence_memory,
        "stats": stats,
    }
    try:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        print(f"saved run state to {OUT_PATH}")
    except Exception as e:
        print(f"failed to save run state to {OUT_PATH}: {e}")
    try:
        memory_snapshot_path = str(Path(OUT_PATH).with_suffix("")) + "_persistence.json"
        with open(memory_snapshot_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "persistence_memory": persistence_memory,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
        print(f"saved persistence snapshot to {memory_snapshot_path}")
    except Exception as e:
        print(f"failed to save persistence snapshot: {e}")


def print_stats(stats):
    print('FINISHED RUNNING THE BENCH')
    print(stats)
    print_summary_stats(stats)
    print('lemmas')
    for k, v in stats.items():
        if not isinstance(v, int):
            print(k)
            print(v)
    print_summary_stats(stats)
    save_run_state(stats)


def _load_persistence_from_jsonl(path: str) -> list[str]:
    """Load persistence memories from a JSONL file."""
    memories = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    mem = entry.get("memory", "")
                    if mem:
                        memories.append(mem)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return memories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default=LLM_MODEL,
                        help='litellm model to use (overrides LLM_MODEL env)')
    parser.add_argument('--force-llm', action='store_true',
                        help='Always run LLM stage even if empty/sketch proof succeeds')
    parser.add_argument('--out', type=str, default=DEFAULT_OUT_PATH,
                        help='Path to JSON file where final run state is saved')
    parser.add_argument('--persistence-out', type=str, default=DEFAULT_PERSISTENCE_OUT_PATH,
                        help='Path to JSONL file where persistence memory is appended during run')
    parser.add_argument('--persistence-in', type=str, default=None,
                        help='Path to JSONL file to load persistence memory from previous runs')
    parser.add_argument('--samples-out', type=str, default=DEFAULT_SAMPLES_OUT_PATH,
                        help='Path to JSONL file where per-iteration samples are appended')
    args, remaining = parser.parse_known_args()
    LLM_MODEL = args.model
    provider = LiteLLMProvider(model=LLM_MODEL)
    OUT_PATH = args.out
    PERSISTENCE_OUT_PATH = args.persistence_out
    SAMPLES_OUT_PATH = args.samples_out
    FORCE_LLM = args.force_llm

    # Load persistence memory from previous runs
    persistence_in = args.persistence_in or PERSISTENCE_OUT_PATH
    loaded = _load_persistence_from_jsonl(persistence_in)
    if loaded:
        persistence_memory.extend(loaded)
        print(f"Loaded {len(loaded)} persistence memories from {persistence_in}")

    # Hand remaining args to bench_driver parser.
    sys.argv = [sys.argv[0]] + remaining
    bench_driver.run(lemma1, print_stats)
