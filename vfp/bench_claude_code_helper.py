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


COMMANDS = {
    "execute": cmd_execute,
    "verify_method": cmd_verify_method,
    "verify_slow": cmd_verify_slow,
    "detect_axiom": cmd_detect_axiom,
    "list_declarations": cmd_list_declarations,
    "inspect_function": cmd_inspect_function,
    "induction_sketch": cmd_induction_sketch,
    "insert_body": cmd_insert_body,
    "edit_program": cmd_edit_program,
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
