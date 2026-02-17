"""Bench workflow using effectful's LLM module with algebraic effects.

Represents sketcher calls as effectful Tools, and the LLM lemma-implementation
loop as an effectful Template.  The LLM can invoke the sketcher tools during
its proof search.

Usage:
    python bench_effectful.py
    python bench_effectful.py --file bench/binary_search_solution.dfy
    python bench_effectful.py --file bench/reverse_solution.dfy --lemma reverseLength
    USE_SKETCHERS=true python bench_effectful.py
    MAX_ITERS=5 python bench_effectful.py

Environment variables:
    USE_SKETCHERS: Set to 'false' to disable sketcher tools (default: true)
    MAX_ITERS: Maximum LLM repair iterations (default: 3)
    LLM_MODEL: litellm model string (default: from LLM_MODEL env or "gpt-4o")
"""

import os
from typing import Optional

import driver
import sketcher
from fine import format_errors
from driver import prompt_begin_dafny, extract_dafny_program

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler

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

USE_SKETCHERS = os.environ.get('USE_SKETCHERS', 'true').lower() != 'false'
MAX_ITERS = int(os.environ.get('MAX_ITERS', '3'))
LLM_MODEL = os.environ.get('LLM_MODEL', 'vertex_ai/gemini-2.5-flash-lite')

provider = LiteLLMProvider(model=LLM_MODEL)

# ---------------------------------------------------------------------------
# Tools – effectful wrappers around the sketcher / driver helpers
# ---------------------------------------------------------------------------

@Tool.define
def list_errors(program: str, method_name: str) -> str:
    """Check Dafny verification errors for a specific method.

    Args:
        program: Full Dafny program source.
        method_name: Name of the method to verify.

    Returns:
        A human-readable summary of the errors, or "No errors" when clean.
    """
    print(f"[TOOL] list_errors(method_name={method_name!r})", flush=True)
    errs = sketcher.list_errors_for_method(program, method_name)
    if not errs:
        return "No errors"
    return format_errors(errs)


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

@Template.define
def implement_lemma(program: str, lemma_name: str, errors: str) -> str:
    """You are implementing a lemma in a Dafny program.

The current program is:
{program}

The lemma to implement is {lemma_name}.

The current verification errors are:
{errors}

Your goal:
1. Use the `induction_sketch` tool to get a proof sketch if it seems useful.
2. Use the `insert_body` tool to insert your proposed proof body.
3. Use the `list_errors` tool to check whether the proof verifies.
4. Iterate: if there are still errors, refine the body and try again.
5. When the proof verifies (no errors), return ONLY the final proof body
   (without the outer braces), starting with "// BEGIN DAFNY" and ending
   with "// END DAFNY".

Please just provide the body of the lemma (without the outer braces),
starting with a line "// BEGIN DAFNY", ending with a line "// END DAFNY".
"""
    from effectful.ops.types import NotHandled
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
    if not e:
        print("empty proof works")
        stats[name] = -1
        return

    # Step 1: try induction sketch (non-LLM)
    if USE_SKETCHERS:
        ix = sketcher.sketch_induction(xp, name)
        p_ind = driver.insert_program_todo(lemma, init_p, ix)
        e_ind = sketcher.list_errors_for_method(p_ind, name)
        if not e_ind:
            print("inductive proof sketch works")
            stats[name] = 0
            return
        # Start LLM loop from the induction-sketched version
        p = p_ind
        e = e_ind
    else:
        p = xp
        print('Not using sketchers!')

    # Step 2: LLM repair loop mediated by effectful
    for i in range(MAX_ITERS):
        with handler(provider): #, handler(RetryLLMHandler()):
            r = implement_lemma(p, name, format_errors(e))

        x = extract_dafny_program(r)
        if x is None:
            print(f"  iteration {i}: LLM did not return valid Dafny")
            continue

        p = driver.insert_program_todo(lemma, init_p, x)
        e = sketcher._list_errors_for_method_core(p, name)
        if not e:
            print("LLM repair loop works " + str(i))
            stats[name] = 1
            stats['proof_' + name] = x
            return

    print("all failed :(")
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
    import bench_driver
    bench_driver.run(lemma1, print_stats)
