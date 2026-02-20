"""Bench workflow using effectful's LLM module with algebraic effects.

Represents sketcher calls as effectful Tools, and the LLM lemma-implementation
loop as an effectful Template.  The LLM can invoke the sketcher tools during
its proof search.

Usage:
    python bench_effectful.py
    python bench_effectful.py --file bench/binary_search_solution.dfy
    python bench_effectful.py --file bench/reverse_solution.dfy --lemma reverseLength
    USE_SKETCHERS=true python bench_effectful.py

Environment variables:
    USE_SKETCHERS: Set to 'false' to disable sketcher tools (default: true)
    LLM_MODEL: litellm model string (default: from LLM_MODEL env or "gpt-4o")
    MAX_VERIFICATION_ATTEMPTS: Max execute() failures per lemma before stopping (default: 5)
"""

import sys
from pathlib import Path

# Use .venv at repo root first
_repo_root = Path(__file__).resolve().parent.parent
_venv_site = _repo_root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if _venv_site.exists():
    sys.path.insert(0, str(_venv_site))

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
if not os.environ.get("DAFNY"):
    _dafny_dll = Path(__file__).resolve().parent.parent / "dafny" / "Binaries" / "Dafny.dll"
    if _dafny_dll.exists():
        os.environ["DAFNY"] = str(_dafny_dll)

from pydantic import BaseModel, Field

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled
import driver
import sketcher

import litellm
_orig_completion = litellm.completion
def _traced_completion(*args, **kwargs):
    model = kwargs.get("model", args[0] if args else "unknown")
    kwargs.setdefault("timeout", 60)  # 1-minute timeout to avoid infinite hangs
    print(f"[LLM] Querying {model} ...", flush=True)
    result = _orig_completion(*args, **kwargs)
    print(f"[LLM] Response received from {model}", flush=True)
    return result
litellm.completion = _traced_completion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# DUPLICATE CODE NEEDEDTO NOT DEPEND on FINE.py, which in turns, on llm.py
# Which is buggy
def format_errors(errs):
    r = ""
    for row,col,err,snippet in errs:
        r += f"{row},{col}: {err} -- {snippet}\n"
    return r

USE_SKETCHERS = os.environ.get('USE_SKETCHERS', 'true').lower() != 'false'
LLM_MODEL = os.environ.get('LLM_MODEL', 'vertex_ai/gemini-2.5-flash-lite')
MAX_VERIFICATION_ATTEMPTS = int(os.environ.get('MAX_VERIFICATION_ATTEMPTS', '5'))
FORCE_LLM = False

provider = LiteLLMProvider(model=LLM_MODEL)

# Per-lemma counter: failed execute() calls; reset at start of each lemma
_execute_attempt_count = 0

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
def execute(program: str) -> str:
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
    global _execute_attempt_count
    print(f"[TOOL] execute()", flush=True)
    errs = sketcher.list_errors_for_method(program, None)
    if errs:
        _execute_attempt_count += 1
        error_msg = format_errors(errs)
        if _execute_attempt_count >= MAX_VERIFICATION_ATTEMPTS:
            print(f"[TOOL] execute() max attempts ({MAX_VERIFICATION_ATTEMPTS}) reached, stopping retries", flush=True)
            return (
                f"Max verification attempts ({MAX_VERIFICATION_ATTEMPTS}) reached. "
                f"Last errors:\n{error_msg}\n\n"
                "Return your current best proof body now: a single block starting with // BEGIN DAFNY and ending with // END DAFNY."
            )
        raise VerificationError(error_msg)
    return "Verification succeeded"


@Tool.define
def induction_sketch(program: str, method_name: str) -> str:
    """Generate an induction proof sketch for a lemma.

    Args:
        program: Full Dafny program source (the lemma body should be empty).
        method_name: Name of the lemma.

    Returns:
        The induction sketch body (Dafny code to place inside the lemma).
    """
    print(f"[TOOL] induction_sketch(method_name={method_name!r})", flush=True)
    result = sketcher.sketch_induction(program, method_name)
    return result or ""


@Tool.define
def insert_body(lemma_name: str, original_program: str, body: str) -> str:
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
    done = sketcher.sketch_done(original_program)
    if done is None:
        return "Error: could not resolve program metadata"
    lemma = next((x for x in done if x['name'] == lemma_name), None)
    if lemma is None:
        return f"Error: lemma '{lemma_name}' not found in program"
    return driver.insert_program_todo(lemma, original_program, body)


# ---------------------------------------------------------------------------
# Template – the LLM-powered lemma implementer
# ---------------------------------------------------------------------------

class ThinkLemma(BaseModel):
    think: str = Field(description="Your thoughts and plans for the lemma.")
    code: str = Field(description="The code to insert into the lemma.")

@Template.define
def implement_lemma(program: str, lemma_name: str, errors: str) -> ThinkLemma:
    """You are implementing a lemma in a Dafny program.

The current program is:
{program}

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

Think, and provie your implementation of the lemma.
"""
    raise NotHandled


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def lemma1(lemma, p, stats):
    """Process a single lemma using effectful tools + LLM template."""
    init_p = p
    name = lemma['name']
    print('lemma', name)

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
        p = p_ind
        e = e_ind
    else:
        p = xp

    # Step 2: LLM repair mediated by effectful (retries until success or MAX_VERIFICATION_ATTEMPTS)
    global _execute_attempt_count
    _execute_attempt_count = 0
    with handler(provider), handler(RetryLLMHandler()):
        r = implement_lemma(p, name, format_errors(e))

    x = r.code
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
    else:
        print("LLM repair failed - still has errors")
        stats[name] = 2
        stats['failed_proof_' + name] = x


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--force-llm', action='store_true',
                        help='Always run LLM stage even if empty/sketch proof succeeds')
    args, remaining = parser.parse_known_args()
    FORCE_LLM = args.force_llm

    # Hand remaining args to bench_driver parser.
    sys.argv = [sys.argv[0]] + remaining
    import bench_driver
    bench_driver.run(lemma1, print_stats)
