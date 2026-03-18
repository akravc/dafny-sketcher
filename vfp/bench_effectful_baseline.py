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
    kwargs.setdefault("timeout", 600)  # 10-minute timeout
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

# Per-lemma tool call counter and LLM call counter
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
        dafny_program: Full Dafny program source.

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
        dafny_program: Full Dafny program source (the lemma body should be empty).
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

    The lemma is identified by *lemma_name* inside *original_dafny_program*.
    *body* is the code to place inside the lemma braces.

    Args:
        lemma_name: Name of the lemma to fill in.
        original_dafny_program: The original Dafny program (with empty lemma).
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

Your goal:
1. Use the `induction_sketch` tool to get a proof sketch if it seems useful.
2. Use the `insert_body` tool to insert your proposed proof body.
3. Use the `execute` tool to verify the proof. If it raises an error, the system
   will automatically retry, giving you a chance to refine your solution.
4. Iterate: if verification fails (error is raised), refine the body and try again.
5. When verification succeeds (execute returns successfully), return ONLY the final proof body (without the outer braces), starting with "// BEGIN DAFNY" and ending with "// END DAFNY".
6. If execute() returns a message saying "Max verification attempts reached", return your current best proof body in the same format (// BEGIN DAFNY ... // END DAFNY).

Think, and provie your implementation of the lemma. If you find something useful, write it to the persistence memory using the `write_to_persistence_memory` tool, and read previous runs' persistence memory using the `read_persistence_memory` tool.
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
        stats['calls_' + name] = {"tool_calls": dict(_tool_calls), "llm_calls": _llm_call_count}
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
