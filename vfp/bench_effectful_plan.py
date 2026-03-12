"""Bench workflow using effectful's LLM planning/staging pattern.

Instead of the LLM making sequential tool calls, it generates a Python function
(a "plan") that orchestrates tool calls programmatically. The plan can use loops,
conditionals, error handling, and can recursively call make_plan() for sub-problems.

This is the effectful "higher-order function" pattern from llm.ipynb: the LLM
writes code that calls tools as normal Python functions.

Usage:
    python bench_effectful_plan.py
    python bench_effectful_plan.py --model gpt-5.4
    python bench_effectful_plan.py --glob-pattern 'DafnyBench/*.dfy'
"""

import argparse
import json
import os
import re
import sys
from collections.abc import Callable
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
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# Monkey-patch pydantic.create_model to fix effectful bug where bare types
# are passed as kwargs instead of tuple format (type, ...).
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

from error_parser import (
    parse_dafny_errors, extract_proof_obligations,
    format_errors_structured, format_proof_obligations,
)
from fine import format_errors

_repo_root = Path(__file__).resolve().parent.parent

load_dotenv(_repo_root / ".env")
if not os.environ.get("DAFNY"):
    _dafny_dll = _repo_root / "dafny" / "Binaries" / "Dafny.dll"
    if _dafny_dll.exists():
        os.environ["DAFNY"] = str(_dafny_dll)

_orig_completion = litellm.completion
def _traced_completion(*args, **kwargs):
    model = kwargs.get("model", args[0] if args else "unknown")
    kwargs.setdefault("timeout", 120)
    print(f"[LLM] Querying {model} ...", flush=True)
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
DEFAULT_OUT_PATH = str(_repo_root / "vfp" / "bench_effectful_plan_latest.json")
DEFAULT_PERSISTENCE_OUT_PATH = str(_repo_root / "vfp" / "bench_effectful_plan_persistence_latest.jsonl")
DEFAULT_SAMPLES_OUT_PATH = str(_repo_root / "vfp" / "bench_effectful_plan_samples_latest.jsonl")
OUT_PATH = DEFAULT_OUT_PATH
PERSISTENCE_OUT_PATH = DEFAULT_PERSISTENCE_OUT_PATH
SAMPLES_OUT_PATH = DEFAULT_SAMPLES_OUT_PATH

provider = LiteLLMProvider(model=LLM_MODEL)

_execute_attempt_count = 0
_current_lemma_name: Optional[str] = None
_current_sample_index = 0
_current_last_code: str = ""


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
# Plain-function tools (NOT @Tool.define — these are called directly by plans)
# ---------------------------------------------------------------------------
# The LLM-generated plan calls these as normal Python functions.
# They're injected into the plan's execution environment.


def verify_program(program: str) -> str:
    """Verify a full Dafny program. Returns 'OK' or error string."""
    global _execute_attempt_count, _current_sample_index
    _current_sample_index += 1
    print("[PLAN-TOOL] verify_program()", flush=True)
    errs = sketcher.list_errors_for_method(program, None)
    if errs:
        _execute_attempt_count += 1
        msg = format_errors(errs)
        _append_sample_event("verify_failed", execute_attempt=_execute_attempt_count, errors=msg)
        return f"ERRORS:\n{msg}"
    _append_sample_event("verify_succeeded")
    return "OK"


def verify_method(program: str, method_name: str) -> str:
    """Verify a specific method/lemma. Returns 'OK' or error string."""
    print(f"[PLAN-TOOL] verify_method({method_name!r})", flush=True)
    errs = sketcher.list_errors_for_method(program, method_name)
    if errs:
        return f"ERRORS for {method_name}:\n{format_errors(errs)}"
    return "OK"


def verify_isolated(program: str, lemma_name: str) -> str:
    """Verify a lemma in isolation (other lemma bodies stubbed with 'assume false;')."""
    print(f"[PLAN-TOOL] verify_isolated({lemma_name!r})", flush=True)
    done = sketcher.sketch_done(program)
    if not done:
        errs = sketcher.list_errors_for_method(program, lemma_name)
        if errs:
            return f"ERRORS for {lemma_name}:\n{format_errors(errs)}"
        return "OK"

    lines = program.splitlines(keepends=True)
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
        return f"ERRORS for {lemma_name} (isolated):\n{format_errors(errs)}"
    return "OK"


def insert_body(program: str, lemma_name: str, body: str) -> str:
    """Insert a proof body into a lemma. Returns the full program with body inserted."""
    global _current_last_code
    _current_last_code = body
    print(f"[PLAN-TOOL] insert_body({lemma_name!r})", flush=True)
    done = sketcher.sketch_done(program)
    if done is None:
        return "Error: could not parse program"
    lemma = next((x for x in done if x['name'] == lemma_name), None)
    if lemma is None:
        return f"Error: lemma '{lemma_name}' not found"
    result = driver.insert_program_todo(lemma, program, body)
    _append_sample_event("insert_body", code=body, program=result)
    return result


def get_induction_sketch(program: str, lemma_name: str) -> str:
    """Get an induction proof sketch for a lemma."""
    print(f"[PLAN-TOOL] get_induction_sketch({lemma_name!r})", flush=True)
    result = sketcher.sketch_induction(program, lemma_name)
    return result or ""


def get_counterexamples(program: str, lemma_name: str) -> str:
    """Get counterexamples showing why a lemma fails."""
    print(f"[PLAN-TOOL] get_counterexamples({lemma_name!r})", flush=True)
    results = sketcher.sketch_counterexamples(program, lemma_name)
    if isinstance(results, str):
        return results
    if results:
        return "\n".join(f"  {i}. {ce}" for i, ce in enumerate(results, 1))
    return "No counterexamples found."


def parse_errors(program: str, method_name: str = "") -> str:
    """Parse verification errors into structured format with categories and suggestions."""
    print(f"[PLAN-TOOL] parse_errors({method_name!r})", flush=True)
    raw = sketcher._show_errors_for_method_core(program, method_name or None)
    if not raw:
        return "No errors."
    errors = parse_dafny_errors(raw)
    if not errors:
        return "No errors parsed."
    parts = [format_errors_structured(errors)]
    obligations = extract_proof_obligations(errors, method_name)
    if obligations:
        parts.append(format_proof_obligations(obligations))
    return "\n".join(parts)


def detect_axiom(program: str, lemma_name: str) -> bool:
    """Check if a lemma is an axiom (depends on bodyless functions). Returns True/False."""
    print(f"[PLAN-TOOL] detect_axiom({lemma_name!r})", flush=True)
    done = sketcher.sketch_done(program)
    if not done:
        return False
    lemma = next((x for x in done if x.get('name') == lemma_name), None)
    if lemma is None:
        return False
    bodyless_fns = [item['name'] for item in done
                    if item.get('type') == 'function' and item.get('status') != 'done']
    lines = program.splitlines()
    start = lemma.get('startLine', 1) - 1
    end = lemma.get('endLine', lemma.get('insertLine', start + 1))
    lemma_sig = '\n'.join(lines[start:end])
    if '{:axiom}' in lemma_sig:
        return True
    return any(fn in lemma_sig for fn in bodyless_fns)


def find_relevant(program: str, lemma_name: str) -> str:
    """Find declarations relevant to proving a lemma."""
    print(f"[PLAN-TOOL] find_relevant({lemma_name!r})", flush=True)
    done = sketcher.sketch_done(program)
    lines = program.splitlines()
    target = next((x for x in (done or []) if x.get('name') == lemma_name), None)
    if not target:
        return f"Lemma '{lemma_name}' not found."
    start = target.get('startLine', 1) - 1
    end = target.get('endLine', start + 1)
    sig_region = '\n'.join(lines[start:end])
    identifiers = set(re.findall(r'\b([A-Za-z_]\w*)\b', sig_region))
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
        return f"Relevant declarations:\n" + "\n".join(relevant)
    return "No directly relevant declarations found."


def analyze_induction(program: str, lemma_name: str) -> str:
    """Analyze induction structure: candidates, base/recursive cases, and sketch."""
    print(f"[PLAN-TOOL] analyze_induction({lemma_name!r})", flush=True)
    sketch = sketcher.sketch_induction(program, lemma_name)
    if not sketch or "Error" in sketch:
        sketch = sketcher.sketch_induction(program, lemma_name, shallow=True)
    done = sketcher.sketch_done(program)
    lines = program.splitlines()
    lemma = next((x for x in (done or []) if x.get('name') == lemma_name), None)
    parts = [f"=== Induction Analysis for {lemma_name} ==="]
    if lemma:
        start = lemma.get('startLine', 1) - 1
        insert = lemma.get('insertLine', start + 1)
        sig = '\n'.join(lines[start:insert])
        parts.append(f"\nSignature:\n{sig}")
        params = re.findall(r'(\w+)\s*:\s*(\w[\w<>,._ ]*)', sig)
        if params:
            parts.append(f"\nParameters: {', '.join(f'{n}: {t}' for n, t in params)}")
            datatypes = set()
            for l in lines:
                m = re.match(r'^\s*datatype\s+(\w+)', l)
                if m:
                    datatypes.add(m.group(1))
            candidates = [(n, t) for n, t in params if t in datatypes or t in ('nat', 'Nat')]
            if candidates:
                parts.append(f"Induction candidates: {', '.join(f'{n} ({t})' for n, t in candidates)}")
    if sketch:
        parts.append(f"\nInduction sketch:\n{sketch}")
    else:
        parts.append("\nNo induction sketch available.")
    return '\n'.join(parts)


def inspect_function(program: str, name: str) -> str:
    """Inspect a function/lemma: has_body, is_axiom, is_opaque, is_ghost."""
    print(f"[PLAN-TOOL] inspect_function({name!r})", flush=True)
    done = sketcher.sketch_done(program)
    item = next((x for x in (done or []) if x.get('name') == name), None)
    if item is None:
        return f"'{name}' not found"
    lines = program.splitlines()
    sl = item.get('startLine', 0) - 1
    el = item.get('endLine', 0) - 1
    decl = '\n'.join(lines[sl:el+1]) if sl >= 0 else ''
    info = {
        'name': name,
        'type': item.get('type', '?'),
        'has_body': item.get('status') == 'done',
        'is_axiom': '{:axiom}' in decl,
        'is_opaque': 'opaque' in decl.split(name)[0] if name in decl else False,
    }
    if info['is_opaque']:
        info['note'] = f'Opaque — use "reveal {name}();" in proof.'
    if info['type'] == 'function' and not info['has_body']:
        info['note'] = 'No body (uninterpreted). Lemmas about it may be axioms.'
    return json.dumps(info, indent=2)


def list_declarations(program: str) -> str:
    """List all declarations with signatures and flags."""
    print("[PLAN-TOOL] list_declarations()", flush=True)
    done = sketcher.sketch_done(program)
    lines = program.splitlines()
    if not done:
        results = []
        for line in lines:
            stripped = line.strip()
            if re.match(r'^(lemma|function|method|predicate|ghost\s+function|ghost\s+method)\b', stripped):
                results.append(stripped.rstrip('{').strip())
        return '\n'.join(results) if results else "No declarations found."
    parts = []
    for item in done:
        kind = item.get('type', '?')
        name = item.get('name', '?')
        has_body = item.get('status') == 'done'
        start = item.get('startLine', 0)
        flags = []
        if start > 0 and start <= len(lines) and '{:axiom}' in lines[start-1]:
            flags.append('axiom')
        if not has_body:
            flags.append('no-body')
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        insert = item.get('insertLine', start)
        sig_lines = lines[start-1:insert] if start > 0 else []
        sig = ' '.join(l.strip() for l in sig_lines).rstrip('{').strip()
        parts.append(f"[{kind}] {name}{flag_str}: {sig}")
    return '\n'.join(parts)


def check_calc(program: str, lemma_name: str) -> str:
    """Check each step of a calc block individually."""
    print(f"[PLAN-TOOL] check_calc({lemma_name!r})", flush=True)
    from calc_checker import check_calc_steps
    return check_calc_steps(program, lemma_name)


def dependency_order(program: str) -> str:
    """Get optimal lemma solve order based on dependencies."""
    print("[PLAN-TOOL] dependency_order()", flush=True)
    from dependency_graph import get_solve_order
    order = get_solve_order(program)
    if order:
        return "\n".join(f"  {i}. {n}" for i, n in enumerate(order, 1))
    return "No lemmas found."


def dependency_info(program: str, lemma_name: str) -> str:
    """Show what a lemma depends on."""
    print(f"[PLAN-TOOL] dependency_info({lemma_name!r})", flush=True)
    from dependency_graph import format_dependency_info
    return format_dependency_info(program, lemma_name)


# Persistence memory (shared across plans)
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
        print(f"failed to append persistence event: {e}")


def read_memory() -> str:
    """Read persistence memory from previous lemmas."""
    print(f"[PLAN-TOOL] read_memory() -> {len(persistence_memory)} entries", flush=True)
    return "\n".join(persistence_memory) if persistence_memory else "(empty)"


def write_memory(memory: str) -> str:
    """Write an insight to persistence memory for future lemmas."""
    print(f"[PLAN-TOOL] write_memory({memory[:60]!r}...)", flush=True)
    persistence_memory.append(memory)
    _append_persistence_event(memory)
    return "Saved."


# ---------------------------------------------------------------------------
# The planning template: LLM writes a Python function that IS the plan
# ---------------------------------------------------------------------------

# All tools available to the generated plan code
_PLAN_ENV = {
    # Verification
    "verify_program": verify_program,
    "verify_method": verify_method,
    "verify_isolated": verify_isolated,
    # Program manipulation
    "insert_body": insert_body,
    # Analysis
    "detect_axiom": detect_axiom,
    "find_relevant": find_relevant,
    "inspect_function": inspect_function,
    "list_declarations": list_declarations,
    "analyze_induction": analyze_induction,
    "get_induction_sketch": get_induction_sketch,
    "get_counterexamples": get_counterexamples,
    "parse_errors": parse_errors,
    "check_calc": check_calc,
    "dependency_order": dependency_order,
    "dependency_info": dependency_info,
    # Persistence
    "read_memory": read_memory,
    "write_memory": write_memory,
    # Stdlib
    "print": print,
    "json": json,
    "re": re,
}


@Template.define
def make_plan(
    dafny_source: str, lemma_name: str, errors: str
) -> Callable[[str, str, str], str]:
    """You are a Dafny proof engineer. Given a program, a lemma name, and
verification errors, write a Python function called `solve` that attempts
to prove the lemma.

The program is:
{dafny_source}

The lemma to implement is: {lemma_name}

Current verification errors:
{errors}

Your function signature MUST be:
    def solve(program: str, lemma_name: str, errors: str) -> str:

It should return the proof body (Dafny code for inside the lemma braces),
or "AXIOM" if the lemma is unprovable.

Available functions (DO NOT REDEFINE THEM):

VERIFICATION:
- verify_program(program: str) -> str:  Verify full program. Returns "OK" or "ERRORS:..."
- verify_method(program: str, method_name: str) -> str:  Verify one method. Returns "OK" or "ERRORS:..."
- verify_isolated(program: str, lemma_name: str) -> str:  Verify in isolation (stubs other lemmas).

PROGRAM MANIPULATION:
- insert_body(program: str, lemma_name: str, body: str) -> str:  Insert proof body, returns full program.

ANALYSIS:
- detect_axiom(program: str, lemma_name: str) -> bool:  True if lemma is an axiom.
- find_relevant(program: str, lemma_name: str) -> str:  Find relevant declarations.
- inspect_function(program: str, name: str) -> str:  Check if function is opaque/axiom/has body.
- list_declarations(program: str) -> str:  List all declarations.
- analyze_induction(program: str, lemma_name: str) -> str:  Analyze induction structure and get sketch.
- get_induction_sketch(program: str, lemma_name: str) -> str:  Get raw induction sketch code.
- get_counterexamples(program: str, lemma_name: str) -> str:  Get counterexamples.
- parse_errors(program: str, method_name: str) -> str:  Structured error analysis.
- check_calc(program: str, lemma_name: str) -> str:  Check calc block steps individually.
- dependency_order(program: str) -> str:  Optimal solve order.
- dependency_info(program: str, lemma_name: str) -> str:  Show lemma dependencies.

PERSISTENCE:
- read_memory() -> str:  Read insights from previous lemmas.
- write_memory(memory: str) -> str:  Save an insight for future lemmas.

RECURSIVE PLANNING:
- make_plan(program: str, lemma_name: str, errors: str) -> Callable:
    Ask the LLM to generate a new plan function for a sub-problem.
    Returns a callable `solve(program, lemma_name, errors) -> str`.
    Use this when your initial approach fails and you want a fresh strategy.

STRATEGY GUIDE:
1. Start by reading persistence memory for insights.
2. Check if the lemma is an axiom (return "AXIOM" if so).
3. Analyze the lemma: find_relevant, analyze_induction, inspect_function.
4. Try the induction sketch first — it often works.
5. If not, craft a proof body and verify it with verify_method.
6. On failure, use parse_errors for structured feedback and iterate.
7. Use verify_isolated if other lemmas cause cascading failures.
8. Try up to 5 different proof strategies before giving up.
9. Write useful insights to memory before returning.
10. If all else fails, call make_plan() for a fresh approach with different strategy.
"""
    raise NotHandled


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def lemma1(lemma, p, stats):
    """Process a single lemma using LLM-generated plan."""
    global _current_lemma_name, _current_sample_index, _current_last_code
    global _execute_attempt_count
    init_p = p
    name = lemma['name']

    _current_lemma_name = name
    _current_sample_index = 0
    _current_last_code = ""
    _execute_attempt_count = 0
    print('lemma', name)

    # Step -1: auto-detect axioms
    try:
        done = sketcher.sketch_done(init_p)
        if done:
            bodyless_fns = [x['name'] for x in done
                            if x.get('type') == 'function' and x.get('status') != 'done']
            _done_names = {x['name'] for x in done}
            _lines = init_p.splitlines()
            for _i, _line in enumerate(_lines):
                _m = re.match(r'^\s*(?:ghost\s+)?function\s+(\w+)', _line)
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
                        if _j > _i and re.match(r'^\s*(lemma|method|function|predicate|class|module)\b', _lines[_j]):
                            break
                    if not _has_body:
                        bodyless_fns.append(_m.group(1))
            lines = init_p.splitlines()
            start = lemma.get('startLine', 1) - 1
            end = lemma.get('endLine', lemma.get('insertLine', start + 1))
            lemma_sig = '\n'.join(lines[start:end])
            referenced = [fn for fn in bodyless_fns if fn in lemma_sig]
            if referenced:
                print(f"[AXIOM] {name} depends on bodyless function(s): {', '.join(referenced)}")
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
        p, e = p_ind, e_ind
    else:
        p = xp

    # Step 2: LLM generates a plan (Python function) that orchestrates tools
    error_str = format_errors(e) if e else "none"

    # Build env with make_plan available for recursive planning
    plan_env = dict(_PLAN_ENV)
    plan_env["make_plan"] = make_plan

    try:
        with (
            handler(provider),
            handler(RetryLLMHandler(num_retries=3)),
            handler(UnsafeEvalProvider()),
        ):
            plan_fn = make_plan(p, name, error_str)

        print(f"[PLAN] Generated plan function, executing...", flush=True)
        _append_sample_event("plan_generated")

        # Execute the plan with tools in scope
        # Inject tools into the function's globals so it can call them
        if hasattr(plan_fn, '__globals__'):
            plan_fn.__globals__.update(plan_env)

        result = plan_fn(p, name, error_str)

    except Exception as ex:
        print(f"[PLAN] Plan generation/execution failed: {ex}")
        _append_sample_event("plan_error", error=str(ex))
        stats[name] = 2
        return

    _current_last_code = result or ""
    _append_sample_event("plan_returned", code=_current_last_code)

    if result is None or result.strip() == "":
        print("Plan returned empty result")
        stats[name] = 2
        return

    if result.strip() == "AXIOM":
        print("Plan determined lemma is an axiom")
        stats[name] = -2
        return

    # Verify the final result
    x = result
    p_final = driver.insert_program_todo(lemma, init_p, x)
    e_final = sketcher._list_errors_for_method_core(p_final, name)
    if not e_final:
        print("Plan succeeded — proof verified")
        stats[name] = 1
        stats['proof_' + name] = x
        _append_sample_event("lemma_solved", code=x, program=p_final)
    else:
        print("Plan failed — proof has errors")
        stats[name] = 2
        stats['failed_proof_' + name] = x
        _append_sample_event("lemma_unsolved", code=x, program=p_final,
                             errors=format_errors(e_final))


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------

def print_summary_stats(stats):
    print('total for empty proof works:',
          len([v for v in stats.values() if isinstance(v, int) and v == -1]))
    print('total for inductive proof sketch works:',
          len([v for v in stats.values() if isinstance(v, int) and v == 0]))
    print('total for LLM plan works:',
          len([v for v in stats.values() if isinstance(v, int) and v == 1]))
    print('total for unsolved:',
          len([v for v in stats.values() if isinstance(v, int) and v == 2]))
    print('total for axioms:',
          len([v for v in stats.values() if isinstance(v, int) and v == -2]))


def _summary_counts(stats):
    return {
        "empty_proof_works": len([v for v in stats.values() if isinstance(v, int) and v == -1]),
        "inductive_proof_sketch_works": len([v for v in stats.values() if isinstance(v, int) and v == 0]),
        "llm_plan_works": len([v for v in stats.values() if isinstance(v, int) and v == 1]),
        "unsolved": len([v for v in stats.values() if isinstance(v, int) and v == 2]),
        "axioms": len([v for v in stats.values() if isinstance(v, int) and v == -2]),
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
        print(f"failed to save run state: {e}")
    try:
        memory_path = str(Path(OUT_PATH).with_suffix("")) + "_persistence.json"
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "persistence_memory": persistence_memory}, f, indent=2)
        print(f"saved persistence snapshot to {memory_path}")
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
    parser.add_argument('--model', type=str, default=LLM_MODEL)
    parser.add_argument('--force-llm', action='store_true')
    parser.add_argument('--out', type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument('--persistence-out', type=str, default=DEFAULT_PERSISTENCE_OUT_PATH)
    parser.add_argument('--persistence-in', type=str, default=None)
    parser.add_argument('--samples-out', type=str, default=DEFAULT_SAMPLES_OUT_PATH)
    args, remaining = parser.parse_known_args()
    LLM_MODEL = args.model
    provider = LiteLLMProvider(model=LLM_MODEL)
    OUT_PATH = args.out
    PERSISTENCE_OUT_PATH = args.persistence_out
    SAMPLES_OUT_PATH = args.samples_out
    FORCE_LLM = args.force_llm

    persistence_in = args.persistence_in or PERSISTENCE_OUT_PATH
    loaded = _load_persistence_from_jsonl(persistence_in)
    if loaded:
        persistence_memory.extend(loaded)
        print(f"Loaded {len(loaded)} persistence memories from {persistence_in}")

    sys.argv = [sys.argv[0]] + remaining
    bench_driver.run(lemma1, print_stats)
