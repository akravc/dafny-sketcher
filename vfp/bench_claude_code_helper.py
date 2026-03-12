#!/usr/bin/env python3
"""Helper CLI for bench_claude_code.py — called by Claude Code via Bash.

Usage:
    python bench_claude_code_helper.py execute PROGRAM_FILE
    python bench_claude_code_helper.py induction_sketch PROGRAM_FILE METHOD_NAME
    python bench_claude_code_helper.py insert_body LEMMA_NAME ORIGINAL_PROGRAM_FILE BODY_FILE
    python bench_claude_code_helper.py read_persistence
    python bench_claude_code_helper.py write_persistence "your insight here"
"""

import json
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent

# Set up DAFNY env
from dotenv import load_dotenv
load_dotenv(_repo_root / ".env")
if not os.environ.get("DAFNY"):
    _dafny_dll = _repo_root / "dafny" / "Binaries" / "Dafny.dll"
    if _dafny_dll.exists():
        os.environ["DAFNY"] = str(_dafny_dll)

import sketcher
import driver
from fine import format_errors

# Suppress debug output when used as CLI tool
driver.XP_DEBUG = False

PERSISTENCE_PATH = os.environ.get(
    "BENCH_PERSISTENCE_PATH",
    str(_repo_root / "vfp" / "bench_claude_code_persistence_latest.jsonl"),
)


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def cmd_execute(args):
    if len(args) < 1:
        print("Usage: execute PROGRAM_FILE", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    errs = sketcher.list_errors_for_method(program, None)
    if errs:
        print(f"Verification failed:\n{format_errors(errs)}")
    else:
        print("Verification succeeded")


def cmd_induction_sketch(args):
    if len(args) < 2:
        print("Usage: induction_sketch PROGRAM_FILE METHOD_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    method_name = args[1]
    result = sketcher.sketch_induction(program, method_name)
    print(result or "")


def cmd_insert_body(args):
    if len(args) < 3:
        print("Usage: insert_body LEMMA_NAME ORIGINAL_PROGRAM_FILE BODY_FILE", file=sys.stderr)
        sys.exit(1)
    lemma_name = args[0]
    original_program = read_file(args[1])
    body = read_file(args[2])
    done = sketcher.sketch_done(original_program)
    if done is None:
        print("Error: could not resolve program metadata")
        sys.exit(1)
    lemma = next((x for x in done if x['name'] == lemma_name), None)
    if lemma is None:
        print(f"Error: lemma '{lemma_name}' not found in program")
        sys.exit(1)
    program = driver.insert_program_todo(lemma, original_program, body)
    print(program)


def cmd_read_persistence(args):
    memories = []
    try:
        with open(PERSISTENCE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                mem = entry.get("memory", "")
                if mem:
                    memories.append(mem)
    except FileNotFoundError:
        pass
    if memories:
        print("\n".join(memories))
    else:
        print("(empty)")


def cmd_write_persistence(args):
    if len(args) < 1:
        print("Usage: write_persistence \"your insight\"", file=sys.stderr)
        sys.exit(1)
    memory = " ".join(args)
    from datetime import datetime, timezone
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "memory": memory,
    }
    with open(PERSISTENCE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")
    print("Memory written.")


def cmd_verify_method(args):
    """Verify only a specific method/lemma (uses --filter-symbol)."""
    if len(args) < 2:
        print("Usage: verify_method PROGRAM_FILE METHOD_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    method_name = args[1]
    errs = sketcher.list_errors_for_method(program, method_name)
    if errs:
        print(f"Verification failed for {method_name}:\n{format_errors(errs)}")
    else:
        print(f"Verification succeeded for {method_name}")


def cmd_list_declarations(args):
    """List all declarations (lemmas, functions, methods) with signatures."""
    if len(args) < 1:
        print("Usage: list_declarations PROGRAM_FILE", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    done = sketcher.sketch_done(program)
    if not done:
        # Fallback: parse declarations from text
        import re
        for line in program.splitlines():
            stripped = line.strip()
            if re.match(r'^(lemma|function|method|predicate|ghost\s+function|ghost\s+method)\b', stripped):
                print(stripped.rstrip('{').strip())
        return
    for item in done:
        kind = item.get('type', '?')
        name = item.get('name', '?')
        status = item.get('status', '?')
        has_body = status == 'done'
        axiom = '{:axiom}' in program.splitlines()[item['startLine']-1] if item.get('startLine') else False
        flags = []
        if axiom:
            flags.append('axiom')
        if not has_body:
            flags.append('no-body')
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        # Extract signature from source lines
        start = item.get('startLine', 0)
        insert = item.get('insertLine', start)
        sig_lines = program.splitlines()[start-1:insert] if start > 0 else []
        sig = ' '.join(l.strip() for l in sig_lines).rstrip('{').strip()
        print(f"[{kind}] {name}{flag_str}: {sig}")


def cmd_inspect_function(args):
    """Check if a function/lemma is opaque, axiom, or has a body."""
    if len(args) < 2:
        print("Usage: inspect_function PROGRAM_FILE FUNCTION_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    name = args[1]
    done = sketcher.sketch_done(program)
    item = next((x for x in (done or []) if x.get('name') == name), None)
    if item is None:
        print(f"Declaration '{name}' not found")
        return
    lines = program.splitlines()
    start_line = item.get('startLine', 0) - 1
    insert_line = item.get('insertLine', 0) - 1
    end_line = item.get('endLine', 0) - 1

    # Get the full declaration text
    decl_text = '\n'.join(lines[start_line:end_line+1]) if start_line >= 0 else ''

    info = {
        'name': name,
        'type': item.get('type', '?'),
        'status': item.get('status', '?'),
        'has_body': item.get('status') == 'done',
        'is_axiom': '{:axiom}' in decl_text,
        'is_opaque': 'opaque' in decl_text.split(name)[0] if name in decl_text else False,
        'is_ghost': 'ghost' in decl_text.split(name)[0] if name in decl_text else False,
        'start_line': item.get('startLine'),
        'end_line': item.get('endLine'),
    }
    # Check for bodyless ghost functions (uninterpreted)
    if info['type'] == 'function' and not info['has_body']:
        info['note'] = 'This function has no body (uninterpreted). Lemmas about it may be axioms.'
    if info['is_axiom']:
        info['note'] = 'This is an axiom — it cannot be proved, only assumed.'
    if info['is_opaque']:
        info['note'] = f'This is opaque. You need "reveal {name}();" in your proof to use its definition.'
    print(json.dumps(info, indent=2))


def cmd_verify_slow(args):
    """Verify a method with extended timeout (60s instead of 30s)."""
    if len(args) < 2:
        print("Usage: verify_slow PROGRAM_FILE METHOD_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    method_name = args[1]
    import subprocess, tempfile
    file_path = sketcher.write_content_to_temp_file(program)
    if not file_path:
        print("Error writing temp file")
        return
    try:
        dafny_exe = os.environ.get('DAFNY')
        if dafny_exe and dafny_exe.endswith('.dll'):
            cmd = ['dotnet', dafny_exe, 'verify', file_path, '--filter-symbol', method_name]
        else:
            cmd = [dafny_exe or 'dafny', 'verify', file_path, '--filter-symbol', method_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        if "0 errors" in output and "Error:" not in output:
            print(f"Verification succeeded for {method_name} (extended timeout)")
        else:
            errs = sketcher._list_errors_for_method_core(program, method_name)
            if errs:
                print(f"Verification failed for {method_name}:\n{format_errors(errs)}")
            else:
                print(f"Verification succeeded for {method_name} (extended timeout)")
    except subprocess.TimeoutExpired:
        print(f"Verification timed out for {method_name} even with 120s timeout")
    finally:
        try:
            Path(file_path).unlink()
        except Exception:
            pass


def cmd_detect_axiom(args):
    """Check if a lemma depends on uninterpreted (bodyless) functions and is thus an axiom."""
    if len(args) < 2:
        print("Usage: detect_axiom PROGRAM_FILE LEMMA_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    lemma_name = args[1]
    done = sketcher.sketch_done(program)
    if not done:
        print(json.dumps({"lemma": lemma_name, "is_axiom": "unknown", "reason": "could not parse declarations"}))
        return

    # Find the lemma
    lemma = next((x for x in done if x.get('name') == lemma_name), None)
    if lemma is None:
        print(json.dumps({"lemma": lemma_name, "is_axiom": "unknown", "reason": "lemma not found"}))
        return

    # Collect all bodyless functions (uninterpreted)
    # First from sketch_done
    bodyless_fns = []
    for item in done:
        if item.get('type') == 'function' and item.get('status') != 'done':
            bodyless_fns.append(item['name'])

    # Also scan source for function declarations without bodies
    # (sketch_done may not include standalone function signatures)
    import re
    lines = program.splitlines()
    # Collect names already in done (with or without body)
    done_names = {x['name'] for x in done}
    for i, line in enumerate(lines):
        m = re.match(r'^\s*(?:ghost\s+)?function\s+(\w+)', line)
        if m:
            fn_name = m.group(1)
            # Skip if already tracked by sketch_done
            if fn_name in done_names or fn_name in bodyless_fns:
                continue
            # Scan ahead for a body { ... } before hitting the next top-level declaration
            has_body = False
            brace_depth = 0
            for j in range(i, min(i + 20, len(lines))):
                for ch in lines[j]:
                    if ch == '{':
                        brace_depth += 1
                    elif ch == '}':
                        brace_depth -= 1
                if brace_depth > 0:
                    has_body = True
                    break
                # If we hit a blank line or another declaration after the signature, no body
                if j > i and lines[j].strip() == '':
                    break
                if j > i and re.match(r'^\s*(lemma|method|function|predicate|class|module)\b', lines[j]):
                    break
            if not has_body:
                bodyless_fns.append(fn_name)

    # Check if the lemma's signature + ensures clauses reference any bodyless function
    lines = program.splitlines()
    start = lemma.get('startLine', 1) - 1
    end = lemma.get('endLine', lemma.get('insertLine', start + 1))
    lemma_sig = '\n'.join(lines[start:end])

    referenced_bodyless = [fn for fn in bodyless_fns if fn in lemma_sig]

    # Also check {:axiom} attribute
    is_axiom_attr = '{:axiom}' in lemma_sig

    if is_axiom_attr:
        result = {"lemma": lemma_name, "is_axiom": True,
                  "reason": "Has {:axiom} attribute — this is declared as an axiom"}
    elif referenced_bodyless:
        result = {"lemma": lemma_name, "is_axiom": True,
                  "reason": f"Depends on bodyless function(s): {', '.join(referenced_bodyless)}. "
                            f"These have no definition, so the lemma cannot be proved."}
    else:
        result = {"lemma": lemma_name, "is_axiom": False,
                  "reason": "All referenced functions have bodies. This lemma should be provable."}

    print(json.dumps(result, indent=2))


def cmd_edit_program(args):
    """Do a search-and-replace edit on a program file. Avoids insert_body bugs."""
    if len(args) < 3:
        print("Usage: edit_program PROGRAM_FILE SEARCH_FILE REPLACE_FILE", file=sys.stderr)
        print("  Reads SEARCH_FILE and REPLACE_FILE contents, does exact replacement in PROGRAM_FILE.", file=sys.stderr)
        print("  Writes the result to stdout.", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    search = read_file(args[1])
    replace = read_file(args[2])
    if search not in program:
        # Try stripping whitespace for fuzzy match
        search_stripped = search.strip()
        found = False
        for i, line in enumerate(program.splitlines()):
            if search_stripped in line:
                found = True
                break
        if not found:
            print(f"Error: search string not found in program", file=sys.stderr)
            print(program)
            return
    result = program.replace(search, replace, 1)
    print(result)


_HELPERS_DIR = Path(__file__).resolve().parent / "bench_helpers"


def cmd_create_helper(args):
    """Create a verified helper lemma and save to the helper library."""
    if len(args) < 2:
        print("Usage: create_helper HELPER_NAME PROGRAM_FILE", file=sys.stderr)
        print("  PROGRAM_FILE should contain ONLY the helper lemma(s) to verify and save.", file=sys.stderr)
        sys.exit(1)
    helper_name = args[0]
    program = read_file(args[1])

    # Verify the helper lemma first
    errs = sketcher.list_errors_for_method(program, None)
    if errs:
        print(f"Helper verification failed — not saving:\n{format_errors(errs)}")
        return

    # Save to helpers directory
    _HELPERS_DIR.mkdir(parents=True, exist_ok=True)
    helper_path = _HELPERS_DIR / f"{helper_name}.dfy"
    helper_path.write_text(program, encoding="utf-8")
    print(f"Helper '{helper_name}' verified and saved to {helper_path}")


def cmd_list_helpers(args):
    """List all available helper lemma libraries."""
    if not _HELPERS_DIR.exists():
        print("No helpers available yet. Use create_helper to create one.")
        return
    helpers = sorted(_HELPERS_DIR.glob("*.dfy"))
    if not helpers:
        print("No helpers available yet. Use create_helper to create one.")
        return
    for h in helpers:
        content = h.read_text(encoding="utf-8")
        # Extract lemma/function signatures
        sigs = [l.strip() for l in content.splitlines()
                if l.strip().startswith(('lemma ', 'function ', 'predicate '))]
        print(f"[{h.stem}]: {'; '.join(sigs[:3]) if sigs else '(no signatures found)'}")


def cmd_use_helpers(args):
    """Prepend all helper libraries to a program file and output the combined program."""
    if len(args) < 1:
        print("Usage: use_helpers PROGRAM_FILE [HELPER_NAME ...]", file=sys.stderr)
        print("  If no helper names given, includes ALL helpers.", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    helper_names = args[1:] if len(args) > 1 else None

    if not _HELPERS_DIR.exists():
        print(program)
        return

    helpers_content = []
    if helper_names:
        for name in helper_names:
            hp = _HELPERS_DIR / f"{name}.dfy"
            if hp.exists():
                helpers_content.append(f"// --- Helper: {name} ---\n{hp.read_text(encoding='utf-8')}")
            else:
                print(f"Warning: helper '{name}' not found", file=sys.stderr)
    else:
        for hp in sorted(_HELPERS_DIR.glob("*.dfy")):
            helpers_content.append(f"// --- Helper: {hp.stem} ---\n{hp.read_text(encoding='utf-8')}")

    if helpers_content:
        combined = "\n\n".join(helpers_content) + "\n\n// --- Main program ---\n" + program
        print(combined)
    else:
        print(program)


def cmd_verify_isolated(args):
    """Verify a single lemma in isolation by stubbing all other lemma bodies with 'assume false;'."""
    if len(args) < 2:
        print("Usage: verify_isolated PROGRAM_FILE LEMMA_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    target_name = args[1]

    done = sketcher.sketch_done(program)
    if not done:
        # Fallback: just use verify_method
        errs = sketcher.list_errors_for_method(program, target_name)
        if errs:
            print(f"Verification failed for {target_name}:\n{format_errors(errs)}")
        else:
            print(f"Verification succeeded for {target_name}")
        return

    # Stub all other lemma bodies with 'assume false;'
    lines = program.splitlines(keepends=True)
    # Process in reverse order to preserve line numbers
    for item in sorted(done, key=lambda x: x.get('startLine', 0), reverse=True):
        if item.get('name') == target_name:
            continue  # Don't stub the target
        if item.get('type') != 'lemma' or item.get('status') != 'done':
            continue  # Only stub lemmas that have bodies
        start = item.get('insertLine', 0)
        end = item.get('endLine', 0)
        if start <= 0 or end <= 0 or start > end:
            continue
        # Find the opening brace on or after insertLine
        # Replace body content between { and } with assume false;
        body_start = start - 1  # 0-indexed
        body_end = end  # exclusive, 0-indexed
        # Find first { and last }
        joined = ''.join(lines[body_start:body_end])
        brace_open = joined.find('{')
        if brace_open == -1:
            continue
        # Replace the entire body region with a stubbed version
        prefix = joined[:brace_open]
        stub = prefix + "{ assume false; }\n"
        lines[body_start:body_end] = [stub]

    stubbed_program = ''.join(lines)
    errs = sketcher.list_errors_for_method(stubbed_program, target_name)
    if errs:
        print(f"Verification failed for {target_name} (isolated):\n{format_errors(errs)}")
    else:
        print(f"Verification succeeded for {target_name} (isolated)")


def cmd_parse_errors(args):
    """Parse Dafny errors into structured format with categories and suggestions."""
    if len(args) < 1:
        print("Usage: parse_errors PROGRAM_FILE [LEMMA_NAME]", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    method_name = args[1] if len(args) > 1 else None

    from error_parser import parse_dafny_errors, extract_proof_obligations, format_errors_structured, format_proof_obligations

    # Get raw verifier output
    raw_output = sketcher._show_errors_for_method_core(program, method_name)
    if not raw_output:
        print("No errors found.")
        return

    errors = parse_dafny_errors(raw_output)
    if not errors:
        print("Verification succeeded (no errors parsed).")
        return

    print("=== ERRORS ===")
    print(format_errors_structured(errors))

    obligations = extract_proof_obligations(errors, method_name or "")
    if obligations:
        print("\n=== PROOF OBLIGATIONS ===")
        print(format_proof_obligations(obligations))


def cmd_counterexamples(args):
    """Get counterexamples for a lemma to understand why it fails."""
    if len(args) < 2:
        print("Usage: counterexamples PROGRAM_FILE LEMMA_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    method_name = args[1]
    results = sketcher.sketch_counterexamples(program, method_name)
    if isinstance(results, str):
        print(results)
    elif results:
        print("Counterexamples (conditions where the lemma fails):")
        for i, ce in enumerate(results, 1):
            print(f"  {i}. {ce}")
    else:
        print("No counterexamples found (lemma may be correct).")


def cmd_search_lemmas(args):
    """Search for lemmas by name pattern or signature keywords in a program."""
    if len(args) < 2:
        print("Usage: search_lemmas PROGRAM_FILE PATTERN", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    pattern = args[1].lower()

    import re as _re
    results = []
    lines = program.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _re.match(r'^(lemma|function|predicate|ghost\s+function|ghost\s+method|method)\b', stripped):
            # Collect full signature (may span multiple lines)
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
            if pattern in full_sig.lower():
                results.append(f"  Line {i+1}: {full_sig.rstrip('{').strip()}")

    if results:
        print(f"Found {len(results)} matching declaration(s):")
        for r in results:
            print(r)
    else:
        print(f"No declarations matching '{pattern}' found.")


def cmd_find_relevant(args):
    """Find lemmas/functions relevant to proving a specific lemma."""
    if len(args) < 2:
        print("Usage: find_relevant PROGRAM_FILE LEMMA_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    target_name = args[1]
    import re as _re

    done = sketcher.sketch_done(program)
    lines = program.splitlines()

    # Find the target lemma's body
    target = next((x for x in (done or []) if x.get('name') == target_name), None)
    if not target:
        print(f"Lemma '{target_name}' not found.")
        return

    # Get the signature + ensures region
    start = target.get('startLine', 1) - 1
    end = target.get('endLine', start + 1)
    sig_region = '\n'.join(lines[start:end])

    # Extract identifiers used in the signature/ensures
    identifiers = set(_re.findall(r'\b([A-Za-z_]\w*)\b', sig_region))
    # Remove Dafny keywords
    keywords = {'lemma', 'function', 'method', 'predicate', 'requires', 'ensures',
                'decreases', 'modifies', 'reads', 'returns', 'var', 'if', 'else',
                'then', 'match', 'case', 'forall', 'exists', 'true', 'false',
                'int', 'nat', 'bool', 'string', 'seq', 'set', 'map', 'multiset',
                'ghost', 'opaque', 'old', 'fresh', 'allocated', 'null', 'this',
                'assert', 'assume', 'calc', 'reveal', 'while', 'for', 'return',
                target_name}
    identifiers -= keywords

    # Find declarations that use any of these identifiers
    relevant = []
    for item in (done or []):
        name = item.get('name', '')
        if name == target_name or name in keywords:
            continue
        s = item.get('startLine', 1) - 1
        e = item.get('endLine', s + 1)
        item_sig = '\n'.join(lines[s:e])
        # Check if the declaration is referenced by the target
        if name in identifiers:
            kind = item.get('type', '?')
            has_body = item.get('status') == 'done'
            body_str = "has body" if has_body else "NO BODY"
            relevant.append(f"  [{kind}] {name} ({body_str}): {item_sig.splitlines()[0].strip()}")

    if relevant:
        print(f"Declarations relevant to '{target_name}':")
        for r in relevant:
            print(r)
    else:
        print(f"No directly relevant declarations found for '{target_name}'.")


def cmd_check_calc(args):
    """Check each step of a calc block individually to find which step fails."""
    if len(args) < 2:
        print("Usage: check_calc PROGRAM_FILE LEMMA_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    lemma_name = args[1]
    from calc_checker import check_calc_steps
    print(check_calc_steps(program, lemma_name))


def cmd_dependency_order(args):
    """Show the optimal order to solve lemmas based on dependencies."""
    if len(args) < 1:
        print("Usage: dependency_order PROGRAM_FILE [LEMMA_NAME ...]", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    unsolved = args[1:] if len(args) > 1 else None
    from dependency_graph import get_solve_order
    order = get_solve_order(program, unsolved)
    if order:
        print("Recommended solve order:")
        for i, name in enumerate(order, 1):
            print(f"  {i}. {name}")
    else:
        print("No lemmas found to order.")


def cmd_dependency_info(args):
    """Show what a specific lemma depends on."""
    if len(args) < 2:
        print("Usage: dependency_info PROGRAM_FILE LEMMA_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    lemma_name = args[1]
    from dependency_graph import format_dependency_info
    print(format_dependency_info(program, lemma_name))


def cmd_analyze_induction(args):
    """Analyze induction structure: what to induct on, base/recursive cases."""
    if len(args) < 2:
        print("Usage: analyze_induction PROGRAM_FILE LEMMA_NAME", file=sys.stderr)
        sys.exit(1)
    program = read_file(args[0])
    lemma_name = args[1]

    # Get the induction sketch (which includes case analysis)
    sketch = sketcher.sketch_induction(program, lemma_name)
    if not sketch or "Error" in sketch:
        # Fallback: do a shallow sketch
        sketch = sketcher.sketch_induction(program, lemma_name, shallow=True)

    done = sketcher.sketch_done(program)
    lines = program.splitlines()
    lemma = next((x for x in (done or []) if x.get('name') == lemma_name), None)

    analysis = []
    analysis.append(f"=== Induction Analysis for {lemma_name} ===")

    if lemma:
        start = lemma.get('startLine', 1) - 1
        insert = lemma.get('insertLine', start + 1)
        sig = '\n'.join(lines[start:insert])
        analysis.append(f"\nSignature:\n{sig}")

        # Extract parameters
        import re as _re
        params = _re.findall(r'(\w+)\s*:\s*(\w[\w<>,._ ]*)', sig)
        if params:
            analysis.append(f"\nParameters: {', '.join(f'{n}: {t}' for n, t in params)}")

            # Identify recursive/algebraic types that are good induction candidates
            # Look for datatype definitions in the program
            datatypes = set()
            for line in lines:
                dm = _re.match(r'^\s*datatype\s+(\w+)', line)
                if dm:
                    datatypes.add(dm.group(1))

            induction_candidates = [(n, t) for n, t in params
                                    if t in datatypes or t in ('nat', 'Nat')]
            if induction_candidates:
                analysis.append(f"Induction candidates: {', '.join(f'{n} ({t})' for n, t in induction_candidates)}")

    if sketch:
        analysis.append(f"\nInduction sketch from sketcher:\n{sketch}")
    else:
        analysis.append("\nNo induction sketch available. Try manual case analysis.")

    print('\n'.join(analysis))


COMMANDS = {
    "execute": cmd_execute,
    "verify_method": cmd_verify_method,
    "verify_slow": cmd_verify_slow,
    "verify_isolated": cmd_verify_isolated,
    "detect_axiom": cmd_detect_axiom,
    "list_declarations": cmd_list_declarations,
    "inspect_function": cmd_inspect_function,
    "induction_sketch": cmd_induction_sketch,
    "insert_body": cmd_insert_body,
    "edit_program": cmd_edit_program,
    "parse_errors": cmd_parse_errors,
    "counterexamples": cmd_counterexamples,
    "search_lemmas": cmd_search_lemmas,
    "find_relevant": cmd_find_relevant,
    "check_calc": cmd_check_calc,
    "dependency_order": cmd_dependency_order,
    "dependency_info": cmd_dependency_info,
    "analyze_induction": cmd_analyze_induction,
    "create_helper": cmd_create_helper,
    "list_helpers": cmd_list_helpers,
    "use_helpers": cmd_use_helpers,
    "read_persistence": cmd_read_persistence,
    "write_persistence": cmd_write_persistence,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: {sys.argv[0]} <command> [args...]")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)
    COMMANDS[sys.argv[1]](sys.argv[2:])
