"""Calc-step checker for Dafny calc blocks.

Parses calc blocks and verifies each step individually to identify
exactly which transition is failing.
"""

import re
from dataclasses import dataclass
from typing import Optional

import sketcher


@dataclass
class CalcStep:
    """A single step in a calc block."""
    expression: str
    operator: str  # ==, <, <=, >, >=, ==>
    hint: str  # hint block between steps (may be empty)
    line_start: int
    line_end: int


@dataclass
class CalcBlock:
    """A parsed calc block."""
    steps: list[CalcStep]
    default_op: str  # default operator (usually ==)
    full_text: str
    line_start: int
    line_end: int


def parse_calc_block(text: str, start_line: int = 1) -> Optional[CalcBlock]:
    """Parse a calc block from Dafny source text.

    Args:
        text: Text containing a calc block (can be the full method body)
        start_line: Line number offset for reporting

    Returns:
        CalcBlock if found, None otherwise
    """
    # Find the calc block
    m = re.search(r'\bcalc\s*(\S*)\s*\{', text)
    if not m:
        return None

    default_op = m.group(1).strip() if m.group(1).strip() else "=="
    calc_start = m.start()

    # Find matching closing brace
    depth = 0
    calc_end = None
    for i in range(m.end() - 1, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                calc_end = i + 1
                break

    if calc_end is None:
        return None

    calc_text = text[calc_start:calc_end]
    calc_lines = calc_text.splitlines()

    # Parse individual steps
    # A calc block looks like:
    #   calc == {
    #     expr1;
    #     ==  { hint }  // or just == on its own line
    #     expr2;
    #     ...
    #   }
    steps = []
    current_expr_lines = []
    current_hint = ""
    current_op = default_op
    in_hint = False
    hint_depth = 0
    line_num = start_line + text[:calc_start].count('\n')

    # Skip the "calc ... {" line
    body_lines = calc_lines[1:-1] if len(calc_lines) > 2 else []

    for i, raw_line in enumerate(body_lines):
        line = raw_line.strip()
        body_line_num = line_num + 1 + i

        if in_hint:
            if '{' in line:
                hint_depth += line.count('{')
            if '}' in line:
                hint_depth -= line.count('}')
            if hint_depth <= 0:
                in_hint = False
            else:
                current_hint += line + "\n"
            continue

        # Check for operator line (==, <, <=, etc.) possibly with hint block
        op_match = re.match(r'^(==|<=|>=|<|>|==>|<==)\s*(\{.*)?$', line)
        if op_match:
            current_op = op_match.group(1)
            rest = op_match.group(2) or ""
            if '{' in rest:
                hint_depth = rest.count('{') - rest.count('}')
                if hint_depth > 0:
                    in_hint = True
                    current_hint = rest.lstrip('{').rstrip('}').strip() + "\n"
            continue

        # Check for expression line (ends with ;)
        if line and line != '}':
            expr = line.rstrip(';').strip()
            if expr:
                if current_expr_lines:
                    # This is a new expression — the previous one is complete
                    prev_expr = ' '.join(current_expr_lines)
                    steps.append(CalcStep(
                        expression=prev_expr,
                        operator=current_op,
                        hint=current_hint.strip(),
                        line_start=body_line_num - len(current_expr_lines),
                        line_end=body_line_num,
                    ))
                    current_hint = ""
                    current_op = default_op
                current_expr_lines = [expr]
            else:
                current_expr_lines.append(expr)

    # Add final expression
    if current_expr_lines:
        steps.append(CalcStep(
            expression=' '.join(current_expr_lines),
            operator=current_op,
            hint="",
            line_start=line_num + len(body_lines),
            line_end=line_num + len(body_lines),
        ))

    return CalcBlock(
        steps=steps,
        default_op=default_op,
        full_text=calc_text,
        line_start=line_num,
        line_end=line_num + len(calc_lines),
    )


def check_calc_steps(program: str, lemma_name: str) -> str:
    """Check each step of a calc block individually.

    For each pair of consecutive expressions (A op B), creates a small
    Dafny program asserting A op B and verifies it.

    Returns a human-readable report of which steps pass/fail.
    """
    done = sketcher.sketch_done(program)
    lines = program.splitlines()

    lemma = next((x for x in (done or []) if x.get('name') == lemma_name), None)
    if not lemma:
        return f"Lemma '{lemma_name}' not found."

    start = lemma.get('startLine', 1) - 1
    end = lemma.get('endLine', start + 1)
    body_text = '\n'.join(lines[start:end])

    calc = parse_calc_block(body_text, start + 1)
    if not calc:
        return f"No calc block found in '{lemma_name}'."

    if len(calc.steps) < 2:
        return f"Calc block has fewer than 2 steps — nothing to check."

    report = [f"Calc block in '{lemma_name}' ({len(calc.steps)} expressions):"]

    for i in range(len(calc.steps) - 1):
        step_a = calc.steps[i]
        step_b = calc.steps[i + 1]
        op = step_b.operator

        # Build a minimal check program
        # Include everything above the lemma (datatypes, functions, etc.)
        preamble = '\n'.join(lines[:start])
        hint_block = ""
        if step_b.hint:
            hint_block = f"  // hint: {step_b.hint}\n"

        check_program = f"""{preamble}

lemma _calc_check_step_{i}()
  ensures {step_a.expression} {op} {step_b.expression}
{{
{hint_block}}}
"""
        errs = sketcher.list_errors_for_method(check_program, f"_calc_check_step_{i}")
        status = "PASS" if not errs else "FAIL"
        report.append(f"  Step {i+1}: {step_a.expression}")
        report.append(f"    {op}")
        report.append(f"  Step {i+2}: {step_b.expression}")
        if step_b.hint:
            report.append(f"    hint: {step_b.hint}")
        report.append(f"    → {status}")
        if errs:
            from fine import format_errors
            report.append(f"    Errors: {format_errors(errs).strip()}")
        report.append("")

    return '\n'.join(report)
