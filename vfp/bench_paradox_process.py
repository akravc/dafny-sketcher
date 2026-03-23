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
    python bench_paradox_process.py --model anthropic/claude-opus-4-6 \\
        --glob-pattern 'DafnyBench/*.dfy'

    # Run only Mode C (process supervision):
    python bench_paradox_process.py --model anthropic/claude-opus-4-6 \\
        --process-only --glob-pattern 'DafnyBench/*.dfy'

    # Limit LLM calls: 2 = 1 explanation + 1 repair attempt:
    python bench_paradox_process.py --model anthropic/claude-opus-4-6 \\
        --process-only --max-llm-calls 2 --glob-pattern 'DafnyBench/*.dfy'
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import bench_driver
import driver
import litellm
import sketcher
from fine import format_errors
from driver import extract_dafny_program
from bench_paradox import extract_skeleton, prompt_lemma_implementer

_repo_root = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    llm_model: str | None = None
    n_iterations: int = 3
    max_llm_calls: int | None = None
    process_only: bool = False
    out_path: str = str(_repo_root / "vfp" / "bench_paradox_process_latest.json")
    temperature: float = 0.0


def make_generate(config: Config) -> Callable[..., str]:
    """Build the LLM generate function from config.

    If config.llm_model is set, returns a litellm-based generator.
    Otherwise, falls back to the default generator from llm.py.
    """
    if not config.llm_model:
        from llm import default_generate
        return default_generate

    def generate(prompt, max_tokens=4000, temperature=None, model=None):
        m = model or config.llm_model
        t = temperature if temperature is not None else config.temperature
        print(f"[LLM] Querying {m} ...", flush=True)
        resp = litellm.completion(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=t,
            timeout=600,
        )
        result = resp.choices[0].message.content
        print(f"[LLM] Response received", flush=True)
        return result

    return generate


# ---------------------------------------------------------------------------
# Process supervision: ask LLM to explain the skeleton
# ---------------------------------------------------------------------------

def generate_process_explanation(
    program: str, name: str, skeleton: str, generate: Callable[..., str],
) -> str:
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

    return generate(prompt, max_tokens=4000)


def repair_loop_with_process(
    lemma: dict,
    init_p: str,
    start_body: str,
    name: str,
    process_explanation: str,
    generate: Callable[..., str],
    n_iterations: int = 3,
) -> tuple[int | None, str]:
    """Run LLM repair with process explanation prepended to the prompt.

    Like bench_paradox.repair_loop but each repair prompt includes the
    process explanation as context.

    Returns (iteration, proof) on success or (None, last_attempt) on failure.
    """
    p = driver.insert_program_todo(lemma, init_p, start_body)
    e = sketcher.list_errors_for_method(p, name)
    if not e:
        return (-1, start_body)

    last_x = start_body
    for i in range(n_iterations):
        base_prompt = prompt_lemma_implementer(p, name, e)
        prompt = f"""Before attempting the proof, here is an analysis of the proof strategy that should work for this lemma:

{process_explanation}

Now, using this analysis to guide your proof:

{base_prompt}"""
        r = generate(prompt)
        x = extract_dafny_program(r)
        if x is None:
            continue
        last_x = x
        p = driver.insert_program_todo(lemma, init_p, x)
        e = sketcher.list_errors_for_method(p, name)
        if not e:
            return (i, x)

    return (None, last_x)


def repair_loop(
    lemma: dict,
    init_p: str,
    start_body: str,
    name: str,
    generate: Callable[..., str],
    n_iterations: int = 3,
) -> tuple[int | None, str]:
    """Run LLM repair starting from start_body (no process explanation).

    Returns (iteration, proof) on success, (-1, start_body) if start_body
    already verifies, or (None, last_attempt) on failure.
    """
    p = driver.insert_program_todo(lemma, init_p, start_body)
    e = sketcher.list_errors_for_method(p, name)
    if not e:
        return (-1, start_body)

    last_x = start_body
    for i in range(n_iterations):
        prompt = prompt_lemma_implementer(p, name, e)
        r = generate(prompt)
        x = extract_dafny_program(r)
        if x is None:
            continue
        last_x = x
        p = driver.insert_program_todo(lemma, init_p, x)
        e = sketcher.list_errors_for_method(p, name)
        if not e:
            return (i, x)

    return (None, last_x)


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def lemma1(lemma: dict, p: str, stats: dict) -> None:
    init_p = p
    name = lemma['name']
    generate = _generate
    config = _run_config

    print('lemma', name)

    # Extract the skeleton from the solution body
    lines = init_p.splitlines()
    insert = lemma.get('insertLine', 0)
    end = lemma.get('endLine', insert)
    if insert >= len(lines) or end > len(lines):
        print(f"  bad line range ({insert}:{end}), skipping")
        stats[name] = {'empty': None, 'skeleton': None, 'process': None, 'skipped': True}
        return

    body_text = '\n'.join(lines[insert:end-1])
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
    if not config.process_only:
        print("  [empty] starting repair loop...")
        empty_iter, empty_proof = repair_loop(
            lemma, init_p, "", name, generate, config.n_iterations)
        if empty_iter is not None:
            print(f"  [empty] solved at iteration {empty_iter}")
        else:
            print("  [empty] failed")

        print("  [skeleton] starting repair loop...")
        skel_iter, skel_proof = repair_loop(
            lemma, init_p, skeleton, name, generate, config.n_iterations)
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
        explanation = generate_process_explanation(init_p, name, skeleton, generate)
        print(f"  [process] explanation generated ({len(explanation)} chars)")
    except Exception as ex:
        explanation_error = str(ex)
        print(f"  [process] explanation failed: {ex}")

    if explanation:
        n_repair = (config.max_llm_calls - 1) if config.max_llm_calls else config.n_iterations
        print(f"  [process] starting repair loop (max {n_repair} repairs)...")
        proc_iter, proc_proof = repair_loop_with_process(
            lemma, init_p, "", name, explanation, generate, n_repair)
        if proc_iter is not None:
            print(f"  [process] solved at iteration {proc_iter}")
        else:
            print("  [process] failed")
    else:
        proc_iter, proc_proof = None, ""

    result: dict = {
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

def _count_stats(stats: dict) -> dict:
    """Compute all summary counts in a single pass."""
    entries = {k: v for k, v in stats.items() if isinstance(v, dict)}
    counts = {
        'total': len(entries),
        'skipped': 0, 'considered': 0,
        'empty_solved': 0, 'skel_solved': 0, 'proc_solved': 0,
        'neither_es': 0,
        'proc_only_vs_skel': 0, 'skel_only_vs_proc': 0, 'both_ps': 0,
        'proc_fewer_iter': 0, 'skel_fewer_iter': 0, 'same_iter': 0,
        'proc_only_vs_empty': 0,
    }
    for v in entries.values():
        if v.get('skipped'):
            counts['skipped'] += 1
            continue
        counts['considered'] += 1
        e, s, p = v.get('empty'), v.get('skeleton'), v.get('process')
        if e is not None: counts['empty_solved'] += 1
        if s is not None: counts['skel_solved'] += 1
        if p is not None: counts['proc_solved'] += 1
        if e is None and s is None: counts['neither_es'] += 1
        # process vs skeleton
        if p is not None and s is None: counts['proc_only_vs_skel'] += 1
        elif s is not None and p is None: counts['skel_only_vs_proc'] += 1
        elif p is not None and s is not None:
            counts['both_ps'] += 1
            if p < s: counts['proc_fewer_iter'] += 1
            elif s < p: counts['skel_fewer_iter'] += 1
            else: counts['same_iter'] += 1
        # process vs empty
        if p is not None and e is None: counts['proc_only_vs_empty'] += 1
    return counts


def print_summary_stats(stats: dict) -> None:
    c = _count_stats(stats)

    print(f'\ntotal lemmas: {c["total"]}')
    print(f'skipped (no case/if structure): {c["skipped"]}')
    print(f'considered: {c["considered"]}')
    print(f'  empty solved:    {c["empty_solved"]}')
    print(f'  skeleton solved: {c["skel_solved"]}')
    print(f'  process solved:  {c["proc_solved"]}')
    print(f'  neither (empty/skel): {c["neither_es"]}')

    # NOTE: when --process-only is used, skeleton=None for all non-trivial
    # lemmas, so "process-only solved" counts all process successes.
    print(f'\n  === Process vs Skeleton ===')
    print(f'  process-only solved: {c["proc_only_vs_skel"]}')
    print(f'  skeleton-only solved: {c["skel_only_vs_proc"]}')
    if c['both_ps'] > 0:
        # NOTE: process iteration 0 costs 2 LLM calls (explain + 1 repair)
        # while skeleton iteration 0 costs 1 LLM call. This comparison
        # reflects proof-attempt count, not total LLM cost.
        print(f'  both solved: {c["both_ps"]}')
        print(f'    process fewer iterations: {c["proc_fewer_iter"]}')
        print(f'    skeleton fewer iterations: {c["skel_fewer_iter"]}')
        print(f'    same iterations: {c["same_iter"]}')

    print(f'\n  === Process vs Empty ===')
    print(f'  process-only solved: {c["proc_only_vs_empty"]}')


def save_run_state(stats: dict) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(_run_config),
        "stats": stats,
    }
    try:
        with open(_run_config.out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
    except Exception as e:
        print(f"failed to save: {e}")


def print_stats(stats: dict) -> None:
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


# Module-level state set by __main__ and read by lemma1.
_run_config: Config = Config()
_generate: Callable[..., str] = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("generate not initialized"))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(_repo_root / ".env")
    load_dotenv(_repo_root / "vfp" / ".env")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default=None,
                        help='litellm model (e.g. anthropic/claude-opus-4-6)')
    parser.add_argument('--out', type=str, default=Config.out_path)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--process-only', action='store_true',
                        help='Only run Mode C (process), skip A and B')
    parser.add_argument('--max-llm-calls', type=int, default=None,
                        help='Max LLM calls per lemma for process mode '
                             '(e.g. 2=explain+1repair, 3=explain+2repairs)')
    args, remaining = parser.parse_known_args()

    _run_config = Config(
        llm_model=args.model,
        n_iterations=args.iterations,
        max_llm_calls=args.max_llm_calls,
        process_only=args.process_only,
        out_path=args.out,
    )
    _generate = make_generate(_run_config)

    sys.argv = [sys.argv[0]] + remaining
    bench_driver.run(lemma1, print_stats)
