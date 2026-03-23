"""
Benchmark testing the "process supervision" hypothesis for the scaffolding paradox.

Hypothesis: giving the LLM the correct sketch (outcome) doesn't help, but
simulating the *thinking process* that would generate the sketch might help.

For each lemma with a case/if structure:
  1. Mode A: repair from empty (baseline)
  2. Mode B: repair from skeleton (original paradox)
  3. Mode C: ask LLM to explain WHY this sketch is correct (process),
             then repair with that explanation in context

This tests whether process > outcome for Dafny proof generation.

Usage:
    # Run all three modes (A, B, C) with Opus 4.6:
    python bench_paradox_process.py --model anthropic/claude-opus-4-6 --glob-pattern 'DafnyBench/*.dfy'

    # Run only Mode C (process supervision):
    python bench_paradox_process.py --model anthropic/claude-opus-4-6 --process-only --glob-pattern 'DafnyBench/*.dfy'

    # Limit LLM calls: 2 = 1 explanation + 1 repair attempt:
    python bench_paradox_process.py --model anthropic/claude-opus-4-6 --process-only --max-llm-calls 2 --glob-pattern 'DafnyBench/*.dfy'
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import bench_driver
import driver
import litellm
import sketcher
from fine import format_errors
from driver import prompt_begin_dafny, extract_dafny_program
from llm import default_generate as generate
from bench_paradox import extract_skeleton, repair_loop, prompt_lemma_implementer

_repo_root = Path(__file__).resolve().parent.parent
load_dotenv(_repo_root / ".env")
load_dotenv(_repo_root / "vfp" / ".env")


# ---------------------------------------------------------------------------
# Configuration (set from CLI args in __main__)
# ---------------------------------------------------------------------------

_config = {
    "llm_model": None,
    "n_iterations": 3,
    "max_llm_calls": None,
    "process_only": False,
    "out_path": str(_repo_root / "vfp" / "bench_paradox_process_latest.json"),
    "temperature": 0.0,
}


def _litellm_generate(prompt, max_tokens=4000, temperature=None, model=None):
    model = model or _config["llm_model"]
    temp = temperature if temperature is not None else _config["temperature"]
    print(f"[LLM] Querying {model} ...", flush=True)
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temp,
        timeout=600,
    )
    result = resp.choices[0].message.content
    print(f"[LLM] Response received", flush=True)
    return result


def _get_generate():
    """Return the appropriate generate function based on config."""
    return _litellm_generate if _config["llm_model"] else generate


# ---------------------------------------------------------------------------
# Process supervision: ask LLM to explain the skeleton
# ---------------------------------------------------------------------------

def generate_process_explanation(program: str, name: str, skeleton: str) -> str:
    """Ask the LLM to explain the reasoning behind a proof skeleton.

    The prompt receives ONLY the skeleton (case structure with empty branches),
    NOT the correct proof body. This simulates "process supervision" — asking
    the model to reconstruct the thinking that would lead to the sketch.
    """
    prompt = f"""You are a Dafny proof expert. A lemma needs to be proved, and someone has
proposed the following proof skeleton (just the case structure with empty branches):

Program:
{program}

Lemma to prove: {name}

The skeleton (case structure with empty branches) is:
{skeleton}

Please explain step by step:
1. WHY this particular case analysis is the right approach for this lemma
2. What property or structure of the data/function definitions makes this case split necessary
3. For each branch, what the key proof obligation is and what technique (induction, helper lemma call, assertion, etc.) would discharge it
4. Any invariants or key insights that connect the preconditions to the postconditions through this case structure

Be specific about the Dafny functions and predicates involved. This explanation will be used to guide a proof search."""

    return _get_generate()(prompt, max_tokens=4000)


def repair_loop_with_process(lemma, init_p, start_body: str, name: str,
                              process_explanation: str, n_iterations: int = 3):
    """Run LLM repair with process explanation prepended to the prompt.

    Like bench_paradox.repair_loop but each repair prompt includes the
    process explanation as context.
    """
    p = driver.insert_program_todo(lemma, init_p, start_body)
    e = sketcher._list_errors_for_method_core(p, name)
    if not e:
        return (-1, start_body)

    last_x = start_body
    gen = _get_generate()
    for i in range(n_iterations):
        base_prompt = prompt_lemma_implementer(p, name, e)
        prompt = f"""Before attempting the proof, here is an analysis of the proof strategy that should work for this lemma:

{process_explanation}

Now, using this analysis to guide your proof:

{base_prompt}"""
        r = gen(prompt)
        x = extract_dafny_program(r)
        if x is None:
            continue
        last_x = x
        p = driver.insert_program_todo(lemma, init_p, x)
        e = sketcher._list_errors_for_method_core(p, name)
        if not e:
            return (i, x)

    return (None, last_x)


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def lemma1(lemma, p, stats):
    init_p = p
    name = lemma['name']
    n_iterations = _config["n_iterations"]
    process_only = _config["process_only"]
    max_llm_calls = _config["max_llm_calls"]

    print('lemma', name)

    # Extract the skeleton from the solution body
    lines = init_p.splitlines()
    body_text = '\n'.join(lines[lemma['insertLine']:lemma['endLine']-1])
    skeleton = extract_skeleton(body_text)
    if skeleton is None:
        print("  solution has no case/if structure, skipping")
        stats[name] = {'empty': None, 'skeleton': None, 'process': None, 'skipped': True}
        return

    print(f"  skeleton:\n{skeleton}")

    # Check empty proof first
    xp = driver.insert_program_todo(lemma, init_p, "")
    e = sketcher.list_errors_for_method(xp, name)
    if not e:
        print("  empty proof works (trivial)")
        stats[name] = {'empty': -1, 'skeleton': -1, 'process': -1, 'skipped': False}
        return

    # --- Mode A & B: empty and skeleton repair ---
    if not process_only:
        print("  [empty] starting repair loop...")
        empty_iter, empty_proof = repair_loop(lemma, init_p, "", name, n_iterations)
        if empty_iter is not None:
            print(f"  [empty] solved at iteration {empty_iter}")
        else:
            print("  [empty] failed")

        print("  [skeleton] starting repair loop...")
        skel_iter, skel_proof = repair_loop(lemma, init_p, skeleton, name, n_iterations)
        if skel_iter is not None:
            print(f"  [skeleton] solved at iteration {skel_iter}")
        else:
            print("  [skeleton] failed")
    else:
        empty_iter, empty_proof = None, ""
        skel_iter, skel_proof = None, ""

    # --- Mode C: process explanation + repair ---
    print("  [process] generating explanation...")
    explanation = ""
    explanation_error = None
    try:
        explanation = generate_process_explanation(init_p, name, skeleton)
        print(f"  [process] explanation generated ({len(explanation)} chars)")
    except Exception as ex:
        explanation_error = str(ex)
        print(f"  [process] explanation failed: {ex}")

    if explanation:
        n_repair = (max_llm_calls - 1) if max_llm_calls else n_iterations
        print(f"  [process] starting repair loop (max {n_repair} repairs)...")
        proc_iter, proc_proof = repair_loop_with_process(
            lemma, init_p, "", name, explanation, n_repair
        )
        if proc_iter is not None:
            print(f"  [process] solved at iteration {proc_iter}")
        else:
            print("  [process] failed")
    else:
        proc_iter, proc_proof = None, ""

    result = {
        'empty': empty_iter,
        'skeleton': skel_iter,
        'process': proc_iter,
        'skipped': False,
    }
    if explanation_error:
        result['process_error'] = explanation_error

    stats[name] = result
    if empty_iter is not None:
        stats['proof_empty_' + name] = empty_proof
    if skel_iter is not None:
        stats['proof_skeleton_' + name] = skel_proof
    if proc_iter is not None:
        stats['proof_process_' + name] = proc_proof
    if explanation:
        stats['explanation_' + name] = explanation

    save_run_state(stats)


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------

def print_summary_stats(stats):
    entries = {k: v for k, v in stats.items() if isinstance(v, dict)}
    total = len(entries)
    skipped = len([v for v in entries.values() if v.get('skipped')])
    considered = total - skipped

    empty_solved = len([v for v in entries.values()
                        if not v.get('skipped') and v.get('empty') is not None])
    skel_solved = len([v for v in entries.values()
                       if not v.get('skipped') and v.get('skeleton') is not None])
    proc_solved = len([v for v in entries.values()
                       if not v.get('skipped') and v.get('process') is not None])

    neither_es = len([v for v in entries.values()
                      if not v.get('skipped')
                      and v.get('empty') is None and v.get('skeleton') is None])

    print(f'\ntotal lemmas: {total}')
    print(f'skipped (no case/if structure): {skipped}')
    print(f'considered: {considered}')
    print(f'  empty solved:    {empty_solved}')
    print(f'  skeleton solved: {skel_solved}')
    print(f'  process solved:  {proc_solved}')
    print(f'  neither (empty/skel): {neither_es}')

    # Process vs skeleton comparison
    # NOTE: when --process-only is used, skeleton=None for all non-trivial
    # lemmas, so "process-only solved" counts all process successes.
    proc_only = 0
    skel_only = 0
    both_ps = 0
    for v in entries.values():
        if v.get('skipped'):
            continue
        s = v.get('skeleton')
        p = v.get('process')
        if p is not None and s is None:
            proc_only += 1
        elif s is not None and p is None:
            skel_only += 1
        elif p is not None and s is not None:
            both_ps += 1

    print(f'\n  === Process vs Skeleton ===')
    print(f'  process-only solved: {proc_only}')
    print(f'  skeleton-only solved: {skel_only}')
    if both_ps > 0:
        # Compare iteration counts (NOTE: process iteration 0 costs 2 LLM
        # calls while skeleton iteration 0 costs 1, so this comparison
        # reflects proof-attempt count, not total LLM cost)
        proc_faster = sum(1 for v in entries.values()
                          if not v.get('skipped')
                          and v.get('process') is not None
                          and v.get('skeleton') is not None
                          and v['process'] < v['skeleton'])
        skel_faster = sum(1 for v in entries.values()
                          if not v.get('skipped')
                          and v.get('process') is not None
                          and v.get('skeleton') is not None
                          and v['skeleton'] < v['process'])
        same = both_ps - proc_faster - skel_faster
        print(f'  both solved: {both_ps}')
        print(f'    process fewer iterations: {proc_faster}')
        print(f'    skeleton fewer iterations: {skel_faster}')
        print(f'    same iterations: {same}')

    # Process vs empty comparison
    proc_only_ve = 0
    for v in entries.values():
        if v.get('skipped'):
            continue
        if v.get('process') is not None and v.get('empty') is None:
            proc_only_ve += 1

    print(f'\n  === Process vs Empty ===')
    print(f'  process-only solved: {proc_only_ve}')


def save_run_state(stats):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": _config,
        "stats": stats,
    }
    try:
        with open(_config["out_path"], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
    except Exception as e:
        print(f"failed to save: {e}")


def print_stats(stats):
    print('\nFINISHED RUNNING THE BENCH')
    print_summary_stats(stats)

    entries = {k: v for k, v in stats.items() if isinstance(v, dict)}
    print('\n--- per-lemma results ---')
    for name, v in sorted(entries.items()):
        if v.get('skipped'):
            continue
        err = f"  error={v['process_error'][:50]}" if v.get('process_error') else ""
        print(f'  {name}: empty={v.get("empty")}  skeleton={v.get("skeleton")}  process={v.get("process")}{err}')

    print_summary_stats(stats)
    save_run_state(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default=None,
                        help='litellm model (e.g. anthropic/claude-opus-4-6)')
    parser.add_argument('--out', type=str, default=_config["out_path"])
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--process-only', action='store_true',
                        help='Only run Mode C (process), skip A and B')
    parser.add_argument('--max-llm-calls', type=int, default=None,
                        help='Max LLM calls per lemma for process mode '
                             '(e.g. 2=explain+1repair, 3=explain+2repairs)')
    args, remaining = parser.parse_known_args()

    _config["llm_model"] = args.model
    _config["out_path"] = args.out
    _config["n_iterations"] = args.iterations
    _config["process_only"] = args.process_only
    _config["max_llm_calls"] = args.max_llm_calls

    if args.model:
        # Override the generate function used by repair_loop (from bench_paradox)
        import bench_paradox
        bench_paradox.generate = _litellm_generate

    sys.argv = [sys.argv[0]] + remaining
    bench_driver.run(lemma1, print_stats)
