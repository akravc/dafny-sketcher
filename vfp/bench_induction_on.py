"""
Benchmark script like bench_feedback that adds {:induction_on} support.

On the first iteration (before running the induction sketch), the LLM can optionally
specify the {:induction_on X} attribute. If provided, it is inserted on the lemma
declaration before the induction sketcher runs. This can help when the auto-detected
induction variable is suboptimal.

Example: LLM responds with "{:induction_on s1}" for lemma reverse_append(s1, s2).
"""

import re
from llm import default_generate as generate
import driver
import sketcher
import os
from fine import format_errors
from driver import prompt_begin_dafny, extract_dafny_program

USE_SKETCHERS = os.environ.get('USE_SKETCHERS', 'true').lower() != 'false'

# Regex to extract {:induction_on ...} from LLM response - handles nested parens in e.g. optimize(e)
INDUCTION_ON_PATTERN = re.compile(r'\{:induction_on\s+([^{}]+(?:\s*\([^)]*\))?)\}')


def extract_induction_on(text: str) -> str | None:
    """Extract {:induction_on X} attribute from LLM response. Returns X or None."""
    if not text:
        return None
    text = driver.remove_think_blocks(text)
    match = INDUCTION_ON_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()


def insert_induction_on_attribute(p: str, lemma: dict, induction_on: str) -> str:
    """
    Insert {:induction_on X} on the lemma declaration line.
    induction_on is the value X (e.g. 's1', 'optimize', 'optimize(e)').
    """
    lines = p.splitlines(keepends=True)
    line_idx = lemma['startLine'] - 1
    if line_idx < 0 or line_idx >= len(lines):
        return p
    line = lines[line_idx]
    # Insert {:induction_on X} after "lemma" and before the rest
    if line.strip().startswith('lemma'):
        attr = f"{{:induction_on {induction_on}}}"
        # Find position after "lemma" (skip optional whitespace)
        lemma_end = line.find('lemma') + 5
        rest = line[lemma_end:]
        # Preserve leading whitespace
        indent = line[:lemma_end]
        new_line = indent + " " + attr + " " + rest.lstrip()
        lines[line_idx] = new_line
        return ''.join(lines)
    return p


def prompt_induction_on(program: str, name: str, params: str = "") -> str:
    """Prompt the LLM to suggest an optional {:induction_on} attribute."""
    return f"""You are a Dafny expert. For the following lemma, suggest an induction variable if you have a strong guess.

Program:
{program}

The lemma to consider is {name}.

If you know a good induction target, reply with exactly one line containing the attribute, e.g.:
- {{:induction_on xs}}  (for structural induction on parameter xs)
- {{:induction_on optimize}}  (for rule induction following function optimize)
- {{:induction_on optimize(e)}}  (for rule induction on that call)

If you are unsure, reply with "none" or leave the response empty.

Reply with only the attribute line or "none"."""


def prompt_lemma_implementer(program: str, name: str, e: list[str]) -> str:
    return f'You are implementing a lemma in a Dafny program that is specified but not fully implemented. The current program is\n{program}\n\nThe lemma to implement is {name}. {prompt_begin_dafny("lemma")}\nThe errors in the work-in-progress lemma are:\n{format_errors(e)}'


def lemma1(lemma, p, stats):
    init_p = p
    name = lemma['name']
    print('lemma', name)

    x = ""
    xp = driver.insert_program_todo(lemma, init_p, x)
    e = sketcher.list_errors_for_method(xp, name)
    if not e:
        print("empty proof works")
        stats[name] = -1
        return

    induction_on_value = None
    if USE_SKETCHERS:
        # First iteration: ask LLM for optional {:induction_on} suggestion
        prompt = prompt_induction_on(xp, name)
        r = generate(prompt)
        induction_on_value = extract_induction_on(r)
        if induction_on_value and induction_on_value.lower() != 'none':
            print(f"LLM suggested {{:induction_on {induction_on_value}}}")
            xp = insert_induction_on_attribute(xp, lemma, induction_on_value)
        else:
            print("No induction_on suggestion from LLM")

        ix = sketcher.sketch_induction(xp, name)
        p = driver.insert_program_todo(lemma, init_p, ix)
        e = sketcher.list_errors_for_method(p, name)
        if not e:
            print("inductive proof sketch works")
            stats[name] = 0
            if induction_on_value:
                stats['induction_on_' + name] = induction_on_value
            return
    else:
        print('Not using sketchers!')

    for i in range(3):
        prompt = prompt_lemma_implementer(p, name, e)
        r = generate(prompt)
        x = extract_dafny_program(r)
        if x is None:
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


def print_summary_stats(stats):
    print('total for empty proof works:', len([v for v in stats.values() if isinstance(v, int) and v == -1]))
    print('total for inductive proof sketch works:', len([v for v in stats.values() if isinstance(v, int) and v == 0]))
    print('total for LLM repair loop works:', len([v for v in stats.values() if isinstance(v, int) and v == 1]))
    print('total for unsolved:', len([v for v in stats.values() if isinstance(v, int) and v == 2]))
    induction_on_helps = [k for k in stats if k.startswith('induction_on_') and stats.get(k.replace('induction_on_', '')) == 0]
    if induction_on_helps:
        print('induction_on helped (sketch succeeded with LLM suggestion):', len(induction_on_helps))


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
