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


COMMANDS = {
    "execute": cmd_execute,
    "induction_sketch": cmd_induction_sketch,
    "insert_body": cmd_insert_body,
    "read_persistence": cmd_read_persistence,
    "write_persistence": cmd_write_persistence,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: {sys.argv[0]} <command> [args...]")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)
    COMMANDS[sys.argv[1]](sys.argv[2:])
