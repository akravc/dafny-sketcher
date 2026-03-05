"""
Benchmark script with case-by-case inductive repair strategy.

Like bench_induction_on.py up to the induction sketch phase.
Then uses a narrowing repair loop:
  1. Try fixing the first failing inductive case (whole case)
  2. Then try fixing individual failing statements within the case
  3. After a case verifies, move to the next failing case

Also provides available lemma signatures as context to the LLM.
"""

import re
from llm import default_generate as generate
import driver
import sketcher
import os
from fine import format_errors
from driver import prompt_begin_dafny, extract_dafny_program
from bench_induction_on import (
    USE_SKETCHERS,
    extract_induction_on,
    insert_induction_on_attribute,
    prompt_induction_on,
)


# ---------------------------------------------------------------------------
# Helpers: lemma signatures, body/case parsing, replacement
# ---------------------------------------------------------------------------

def extract_lemma_signatures(program: str) -> str:
    """Return a formatted string of every lemma signature in *program*."""
    lines = program.splitlines()
    sigs: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('lemma ') or stripped.startswith('lemma{'):
            sig_lines = [lines[i].rstrip()]
            j = i + 1
            while j < len(lines):
                ns = lines[j].strip()
                if ns.startswith(('requires', 'ensures', 'decreases')):
                    sig_lines.append(lines[j].rstrip())
                    j += 1
                else:
                    break
            sigs.append('\n'.join(sig_lines))
            i = j
        else:
            i += 1
    return '\n'.join(sigs) if sigs else '(none)'


def find_lemma_body(program: str, name: str):
    """
    Locate the lemma body in *program*.

    Returns ``(body_open_line, body_close_line, body_text)`` where the line
    numbers are **1-based** and *body_text* is the content between ``{`` and
    ``}``.  Returns ``None`` when the lemma cannot be found.
    """
    lines = program.splitlines()
    lemma_idx = None
    for i, line in enumerate(lines):
        if re.search(rf'\blemma\b', line) and re.search(rf'\b{re.escape(name)}\b', line):
            lemma_idx = i
            break
    if lemma_idx is None:
        return None

    # Skip past requires/ensures/decreases clauses before looking for the
    # body opening brace, so that attribute braces like {:induction_on …}
    # on the declaration line are not mistaken for the body.
    scan_from = lemma_idx
    for i in range(lemma_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith(('requires', 'ensures', 'decreases')):
            scan_from = i
        else:
            break

    depth = 0
    body_start = body_end = None
    for i in range(scan_from, len(lines)):
        for j, ch in enumerate(lines[i]):
            if ch == '{':
                # Skip attribute braces like {:induction_on …}
                if depth == 0 and j + 1 < len(lines[i]) and lines[i][j + 1] == ':':
                    continue
                if depth == 0:
                    body_start = i
                depth += 1
            elif ch == '}':
                if depth == 0:
                    # Closing an attribute brace we skipped – ignore
                    continue
                depth -= 1
                if depth == 0:
                    body_end = i
                    break
        if body_end is not None:
            break

    if body_start is None or body_end is None:
        return None
    body_text = '\n'.join(lines[body_start + 1 : body_end])
    return (body_start + 1, body_end + 1, body_text)


def find_top_level_cases(body_text: str, body_open_line: int) -> list[dict]:
    """
    Parse top-level proof branches inside a lemma body.

    Handles two flavours:
      * **match/case** – ``case X => { … }``
      * **if/else**    – ``if cond { … } else { … }``

    *body_open_line* is the **1-based** line of the opening ``{``.

    Each returned dict contains:
      * ``header``       – the stripped branch header
      * ``text``         – full text of the branch (header + body)
      * ``start`` / ``end`` – 1-based program line range (inclusive)
      * ``sketch_start`` / ``sketch_end`` – 0-based line indices in *body_text*
    """
    cases = _find_match_cases(body_text, body_open_line)
    if cases:
        print(f"  ## DEBUG: found {len(cases)} match/case branch(es)")
        return cases
    branches = _find_if_else_branches(body_text, body_open_line)
    if branches:
        print(f"  ## DEBUG: found {len(branches)} if/else branch(es)")
    return branches


def _find_match_cases(body_text: str, body_open_line: int) -> list[dict]:
    """Find ``case … =>`` branches (braced or unbraced match)."""
    lines = body_text.splitlines()
    depth = 0
    case_depth = None
    case_starts: list[int] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'case\s+', stripped) and '=>' in stripped:
            if case_depth is None:
                case_depth = depth
            if depth == case_depth:
                case_starts.append(i)
        for ch in line:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1

    if not case_starts:
        return []

    cases = []
    for idx, start in enumerate(case_starts):
        end = case_starts[idx + 1] - 1 if idx + 1 < len(case_starts) else len(lines) - 1
        while end > start and not lines[end].strip():
            end -= 1
        cases.append({
            'header': lines[start].strip(),
            'text': '\n'.join(lines[start : end + 1]),
            'start': body_open_line + 1 + start,
            'end': body_open_line + 1 + end,
            'sketch_start': start,
            'sketch_end': end,
        })
    return cases


def _find_if_else_branches(body_text: str, body_open_line: int) -> list[dict]:
    """
    Find ``if / else if / else`` branches.

    For a body like::

        if cond {        ← branch 0  (lines 0-2, includes ``} else {``)
          body1;
        } else {         ← branch 1  (lines 2-4)
          body2;
        }

    Adjacent branches *overlap* on the ``} else …`` transition line so that
    each branch is self-contained text the LLM can read and reproduce.
    """
    lines = body_text.splitlines()
    branch_starts: list[int] = []
    if_depth = None

    depth = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if if_depth is None:
            if stripped.startswith('if ') and '{' in stripped:
                if_depth = depth
                branch_starts.append(i)
        else:
            if depth == if_depth + 1 and re.match(r'\}\s*else\b', stripped):
                branch_starts.append(i)
        for ch in line:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1

    if len(branch_starts) < 2:
        return []

    # Find the closing line of the whole if/else chain
    depth = if_depth
    chain_end = len(lines) - 1
    for i in range(branch_starts[0], len(lines)):
        for ch in lines[i]:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
        if i > branch_starts[0] and depth == if_depth:
            chain_end = i
            break

    branches = []
    for idx, start in enumerate(branch_starts):
        # Each branch runs up to (and including) the next transition line,
        # so adjacent branches share the ``} else {`` boundary.
        if idx + 1 < len(branch_starts):
            end = branch_starts[idx + 1]
        else:
            end = chain_end
        while end > start and not lines[end].strip():
            end -= 1
        branches.append({
            'header': lines[start].strip(),
            'text': '\n'.join(lines[start : end + 1]),
            'start': body_open_line + 1 + start,
            'end': body_open_line + 1 + end,
            'sketch_start': start,
            'sketch_end': end,
        })
    return branches


def errors_in_range(errors, start_line, end_line):
    """Filter errors whose line falls within [start_line, end_line] (1-based)."""
    return [(l, c, m, s) for (l, c, m, s) in errors if start_line <= l <= end_line]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def prompt_case_repair(program: str, name: str, case_text: str,
                       errors: list, lemma_sigs: str) -> str:
    return f"""You are fixing a specific branch in an inductive proof of lemma {name} in a Dafny program.

Program:
{program}

Available lemma signatures in the file (you may call any of these as helper lemmas):
{lemma_sigs}

The specific proof branch that needs fixing is:
{case_text}

The errors in this branch are:
{format_errors(errors)}

Please provide the full fixed body of lemma {name} (without the outer braces), with this branch corrected and all other branches left intact.
{prompt_begin_dafny("lemma")}"""


def prompt_statement_repair(program: str, name: str, case_text: str,
                            stmt: str, errors: list, lemma_sigs: str) -> str:
    return f"""You are fixing a specific failing statement in an inductive proof of lemma {name} in a Dafny program.

Program:
{program}

Available lemma signatures in the file (you may call any of these as helper lemmas):
{lemma_sigs}

The proof branch containing the error is:
{case_text}

The specific statement that is failing:
{stmt}

The errors at this statement:
{format_errors(errors)}

Please provide the full fixed body of lemma {name} (without the outer braces), with the failing statement corrected and all other branches left intact.
{prompt_begin_dafny("lemma")}"""


def prompt_lemma_implementer(program: str, name: str, e: list, lemma_sigs: str) -> str:
    return f"""You are implementing a lemma in a Dafny program that is specified but not fully implemented. The current program is
{program}

Available lemma signatures in the file (you may call any of these as helper lemmas):
{lemma_sigs}

The lemma to implement is {name}. {prompt_begin_dafny("lemma")}
The errors in the work-in-progress lemma are:
{format_errors(e)}"""


# ---------------------------------------------------------------------------
# Case-level and statement-level repair
# ---------------------------------------------------------------------------

def try_fix_case(p, lemma, init_p, current_sketch, name, case, case_errors, lemma_sigs):
    """One LLM attempt to fix the entire *case*. Returns updated sketch or ``None``."""
    prompt = prompt_case_repair(p, name, case['text'], case_errors, lemma_sigs)
    r = generate(prompt)
    candidate_sketch = extract_dafny_program(r)
    if candidate_sketch is None:
        return None

    candidate_p = driver.insert_program_todo(lemma, init_p, candidate_sketch)
    candidate_errors = sketcher._list_errors_for_method_core(candidate_p, name)

    if not candidate_errors:
        return candidate_sketch

    # Accept if errors *within this case* decreased or stayed the same
    body_info = find_lemma_body(candidate_p, name)
    if body_info:
        _, _, body_text = body_info
        new_cases = find_top_level_cases(body_text, body_info[0])
        for nc in new_cases:
            if nc['header'] == case['header']:
                new_ce = errors_in_range(candidate_errors, nc['start'], nc['end'])
                if len(new_ce) <= len(case_errors):
                    return candidate_sketch
                break
    return None


def try_fix_statements(p, lemma, init_p, current_sketch, name, case, case_errors, lemma_sigs):
    """
    Iteratively fix individual failing statements inside *case*.

    For each error line, ask the LLM to repair that statement (outputting the
    full lemma body).  Accept any change that doesn't increase total errors.
    Returns updated sketch or ``None``.
    """
    working_sketch = current_sketch

    for _attempt in range(5):
        wp = driver.insert_program_todo(lemma, init_p, working_sketch)
        all_errors = sketcher._list_errors_for_method_core(wp, name)
        if not all_errors:
            return working_sketch

        body_info = find_lemma_body(wp, name)
        if body_info is None:
            return None
        _, _, body_text = body_info
        cases = find_top_level_cases(body_text, body_info[0])
        target = next((c for c in cases if c['header'] == case['header']), None)
        if target is None:
            return None

        ce = errors_in_range(all_errors, target['start'], target['end'])
        if not ce:
            return working_sketch

        err = ce[0]
        err_line = err[0]
        wp_lines = wp.splitlines()
        stmt = wp_lines[err_line - 1].strip() if 0 < err_line <= len(wp_lines) else ''
        line_errors = [e for e in ce if e[0] == err_line]

        prompt = prompt_statement_repair(wp, name, target['text'], stmt, line_errors, lemma_sigs)
        r = generate(prompt)
        candidate = extract_dafny_program(r)
        if candidate is None:
            continue

        cp = driver.insert_program_todo(lemma, init_p, candidate)
        new_errors = sketcher._list_errors_for_method_core(cp, name)

        if not new_errors:
            return candidate
        if len(new_errors) <= len(all_errors):
            working_sketch = candidate

    fp = driver.insert_program_todo(lemma, init_p, working_sketch)
    fe = sketcher._list_errors_for_method_core(fp, name)
    if not fe:
        return working_sketch
    if working_sketch != current_sketch:
        return working_sketch
    return None


# ---------------------------------------------------------------------------
# Top-level case-by-case repair driver
# ---------------------------------------------------------------------------

def case_repair(lemma, init_p, sketch, name, lemma_sigs):
    """
    Try to repair the induction sketch case-by-case.

    Returns the repaired sketch text, or ``None`` on failure.
    """
    current_sketch = sketch

    for _iteration in range(10):
        p = driver.insert_program_todo(lemma, init_p, current_sketch)
        e = sketcher._list_errors_for_method_core(p, name)
        if not e:
            return current_sketch

        body_info = find_lemma_body(p, name)
        if body_info is None:
            print('  Could not find lemma body')
            return None
        body_open, _, body_text = body_info
        cases = find_top_level_cases(body_text, body_open)
        if not cases:
            print('  No cases found in sketch')
            return None

        failing = None
        for case in cases:
            ce = errors_in_range(e, case['start'], case['end'])
            if ce:
                failing = (case, ce)
                break

        if failing is None:
            print('  Errors not inside any case – cannot repair case-by-case')
            return None

        case, case_errors = failing
        print(f"  Case `{case['header']}`: {len(case_errors)} error(s)")

        fixed = try_fix_case(p, lemma, init_p, current_sketch, name,
                             case, case_errors, lemma_sigs)
        if fixed is not None:
            current_sketch = fixed
            print('  -> case-level fix applied')
            continue

        fixed = try_fix_statements(p, lemma, init_p, current_sketch, name,
                                   case, case_errors, lemma_sigs)
        if fixed is not None:
            current_sketch = fixed
            print('  -> statement-level fix applied')
            continue

        print(f"  -> could not fix case `{case['header']}`")
        return None

    p = driver.insert_program_todo(lemma, init_p, current_sketch)
    e = sketcher._list_errors_for_method_core(p, name)
    return current_sketch if not e else None


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def lemma1(lemma, p, stats):
    init_p = p
    name = lemma['name']
    print('lemma', name)

    x = ''
    xp = driver.insert_program_todo(lemma, init_p, x)
    e = sketcher.list_errors_for_method(xp, name)
    if not e:
        print('empty proof works')
        stats[name] = -1
        return

    induction_on_value = None
    ix = None

    lemma_sigs = extract_lemma_signatures(init_p)

    if USE_SKETCHERS:
        prompt = prompt_induction_on(xp, name)
        r = generate(prompt)
        induction_on_value = extract_induction_on(r)
        if induction_on_value and induction_on_value.lower() != 'none':
            print(f'LLM suggested {{:induction_on {induction_on_value}}}')
            xp = insert_induction_on_attribute(xp, lemma, induction_on_value)
        else:
            print('No induction_on suggestion from LLM')

        ix = sketcher.sketch_induction(xp, name)
        p = driver.insert_program_todo(lemma, init_p, ix)
        e = sketcher.list_errors_for_method(p, name)
        if not e:
            print('inductive proof sketch works')
            stats[name] = 0
            if induction_on_value:
                stats['induction_on_' + name] = induction_on_value
            return
    else:
        # If not using sketchers, use LLM to synthesize initial attempt
        print('Not using sketchers!')
        prompt = prompt_lemma_implementer(xp, name, e, lemma_sigs)
        r = generate(prompt)
        x = extract_dafny_program(r)
        if x is None:
            return
        ix = x
        p = driver.insert_program_todo(lemma, init_p, x)
        e = sketcher._list_errors_for_method_core(p, name)
        if not e:
            print('Initial LLM repair works')
            stats[name] = 1
            stats['proof_' + name] = x
            return

    # --- Case-by-case repair --------------------------------------------------
    if ix is not None and ix.strip() and not ix.startswith('Error:'):
        print('Attempting case-by-case repair...')
        result = case_repair(lemma, init_p, ix, name, lemma_sigs)
        if result is not None:
            print('case-by-case repair works')
            stats[name] = 1
            stats['proof_' + name] = result
            return

    # --- Whole-proof fallback -------------------------------------------------
    print('Falling back to whole-proof LLM repair...')
    for i in range(3):
        prompt = prompt_lemma_implementer(p, name, e, lemma_sigs)
        r = generate(prompt)
        x = extract_dafny_program(r)
        if x is None:
            continue
        p = driver.insert_program_todo(lemma, init_p, x)
        e = sketcher._list_errors_for_method_core(p, name)
        if not e:
            print('LLM repair loop works ' + str(i))
            stats[name] = 1
            stats['proof_' + name] = x
            return

    print('all failed :(')
    stats[name] = 2
    stats['failed_proof_' + name] = x


# ---------------------------------------------------------------------------
# Stats reporting
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
    induction_on_helps = [
        k for k in stats
        if k.startswith('induction_on_') and stats.get(k.replace('induction_on_', '')) == 0
    ]
    if induction_on_helps:
        print('induction_on helped (sketch succeeded with LLM suggestion):',
              len(induction_on_helps))


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


if __name__ == '__main__':
    import bench_driver
    bench_driver.run(lemma1, print_stats)
