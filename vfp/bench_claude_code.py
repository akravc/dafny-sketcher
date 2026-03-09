"""Bench workflow using Claude Code SDK as the agent.

Uses claude-code-sdk so it runs on your Claude Max subscription (no API key
needed).  Claude Code gets Bash access and calls into the Dafny tools via a
helper CLI script.  Produces the same JSON report structure as bench_effectful.py.

Features:
    - Runs through all DafnyBench files (bench/*_solution.dfy)
    - Auto-resumes from previous run: skips already-completed lemmas
    - Saves state after each lemma so crashes don't lose progress
    - Persistence memory: insights written by the agent are saved to JSONL
      and automatically loaded on the next run so later lemmas (and future
      runs) can reuse reasoning chains

Usage:
    python bench_claude_code.py
    python bench_claude_code.py --file bench/binary_search_solution.dfy
    python bench_claude_code.py --model claude-sonnet-4-5
    python bench_claude_code.py --resume bench_claude_code_latest.json
    USE_SKETCHERS=false python bench_claude_code.py

Environment variables:
    USE_SKETCHERS: Set to 'false' to disable sketcher tools (default: true)
    CLAUDE_MODEL: Model for Claude Code SDK (default: claude-sonnet-4-5)
    MAX_VERIFICATION_ATTEMPTS: Max execute() failures per lemma (default: 5)
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import bench_driver
import driver
import sketcher
import tests
from dotenv import load_dotenv
from fine import format_errors

# Patch SDK for unknown message types (e.g. rate_limit_event) before importing
from claude_code_sdk._errors import MessageParseError
import claude_code_sdk._internal.message_parser as _mp
import claude_code_sdk._internal.client as _client

_orig_parse = _mp.parse_message

class _SkipMessage:
    pass

def _patched_parse(data):
    try:
        return _orig_parse(data)
    except MessageParseError as e:
        if "Unknown message type" in str(e):
            return _SkipMessage()
        raise

_mp.parse_message = _patched_parse
_client.parse_message = _patched_parse

from claude_code_sdk import (
    ClaudeCodeOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    query as _raw_query,
)

_repo_root = Path(__file__).resolve().parent.parent
_vfp_dir = Path(__file__).resolve().parent

load_dotenv(_repo_root / ".env")
if not os.environ.get("DAFNY"):
    _dafny_dll = _repo_root / "dafny" / "Binaries" / "Dafny.dll"
    if _dafny_dll.exists():
        os.environ["DAFNY"] = str(_dafny_dll)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_SKETCHERS = os.environ.get('USE_SKETCHERS', 'true').lower() != 'false'
CLAUDE_MODEL = os.environ.get('CLAUDE_MODEL', 'claude-sonnet-4-5')
MAX_VERIFICATION_ATTEMPTS = int(os.environ.get('MAX_VERIFICATION_ATTEMPTS', '5'))
FORCE_LLM = False
_run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
DEFAULT_ARTIFACTS_DIR = str(_repo_root / "vfp" / f"bench_claude_code_artifacts_{_run_id}")
DEFAULT_OUT_PATH = str(_repo_root / "vfp" / "bench_claude_code_latest.json")
DEFAULT_SAMPLES_OUT_PATH = str(_repo_root / "vfp" / "bench_claude_code_samples_latest.jsonl")
DEFAULT_PERSISTENCE_PATH = str(_repo_root / "vfp" / "bench_claude_code_persistence_latest.jsonl")
OUT_PATH = DEFAULT_OUT_PATH
SAMPLES_OUT_PATH = DEFAULT_SAMPLES_OUT_PATH
PERSISTENCE_PATH = DEFAULT_PERSISTENCE_PATH
ARTIFACTS_DIR = DEFAULT_ARTIFACTS_DIR

# Per-lemma state
_execute_attempt_count = 0
_current_lemma_name: Optional[str] = None
_current_sample_index = 0
_current_last_code: str = ""

persistence_memory: list[str] = []
_resumed_stats: dict = {}
_current_source_file: str = ""

# ---------------------------------------------------------------------------
# Resume / persistence helpers
# ---------------------------------------------------------------------------

def _load_resume_state(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prev_mem = data.get("persistence_memory", [])
        if prev_mem:
            persistence_memory.extend(prev_mem)
            print(f"[RESUME] Loaded {len(prev_mem)} persistence memories from {path}")
        stats = data.get("stats", {})
        completed = {k: v for k, v in stats.items() if isinstance(v, int)}
        for k, v in stats.items():
            if k.startswith("proof_") or k.startswith("failed_proof_"):
                completed[k] = v
        print(f"[RESUME] Found {len([v for v in completed.values() if isinstance(v, int)])} completed lemmas in {path}")
        return completed
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[RESUME] Error loading {path}: {e}")
        return {}


def _load_persistence_from_jsonl(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                mem = entry.get("memory", "")
                if mem and mem not in persistence_memory:
                    persistence_memory.append(mem)
        if persistence_memory:
            print(f"[PERSISTENCE] Loaded {len(persistence_memory)} memories from {path}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[PERSISTENCE] Error loading {path}: {e}")


def _append_persistence_event(memory: str) -> None:
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "memory": memory,
    }
    try:
        with open(PERSISTENCE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception as e:
        print(f"failed to append persistence event: {e}")


def _save_artifact(lemma_name: str, source_file: str, proof_body: str,
                    full_program: str, status: str) -> None:
    """Save a .dfy artifact for a solved/failed lemma."""
    art_dir = Path(ARTIFACTS_DIR)
    art_dir.mkdir(parents=True, exist_ok=True)
    # Use source file basename to avoid collisions across files
    src_base = Path(source_file).stem if source_file else "unknown"
    tag = "solved" if status == "solved" else "failed"
    fname = f"{src_base}__{lemma_name}__{tag}.dfy"
    artifact_path = art_dir / fname
    header = (
        f"// Lemma: {lemma_name}\n"
        f"// Source: {source_file}\n"
        f"// Status: {status}\n"
        f"// Model: {CLAUDE_MODEL}\n"
        f"// Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        f"//\n"
        f"// Proof body:\n"
    )
    body_comment = "\n".join(f"//   {line}" for line in proof_body.splitlines())
    content = header + body_comment + "\n\n" + full_program
    try:
        artifact_path.write_text(content, encoding="utf-8")
        print(f"[ARTIFACT] Saved {artifact_path}")
    except Exception as e:
        print(f"[ARTIFACT] Failed to save {artifact_path}: {e}")


def _append_sample_event(event_type: str, **payload: Any) -> None:
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "lemma": _current_lemma_name,
        "sample_index": _current_sample_index,
        "llm_model": CLAUDE_MODEL,
        "persistence_memory": list(persistence_memory),
        **payload,
    }
    try:
        with open(SAMPLES_OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception as e:
        print(f"failed to append sample event: {e}")


# ---------------------------------------------------------------------------
# SDK query wrapper
# ---------------------------------------------------------------------------

async def query(**kwargs):
    """Wrapper that skips unknown message types from newer CLI."""
    async for message in _raw_query(**kwargs):
        if isinstance(message, _SkipMessage):
            continue
        yield message


# ---------------------------------------------------------------------------
# Agent loop (claude-code-sdk with Bash tool)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Dafny proof engineer working to prove lemmas.

You have access to a Python helper at {helper_path} that provides Dafny tools.

VERIFICATION:
    python {helper_path} verify_method PROGRAM_FILE NAME   # verify ONLY one lemma (preferred)
    python {helper_path} verify_slow PROGRAM_FILE NAME     # verify with 120s timeout (for complex proofs)
    python {helper_path} execute PROGRAM_FILE              # verify full program

ANALYSIS:
    python {helper_path} detect_axiom PROGRAM_FILE NAME    # check if lemma is unprovable (bodyless functions)
    python {helper_path} list_declarations PROGRAM_FILE    # list all declarations with [axiom]/[no-body] flags
    python {helper_path} inspect_function PROGRAM_FILE NAME # check if opaque/axiom/has body

PROOF CONSTRUCTION:
    python {helper_path} induction_sketch PROGRAM_FILE METHOD_NAME
    python {helper_path} insert_body LEMMA_NAME ORIGINAL_PROGRAM_FILE BODY_FILE
    python {helper_path} edit_program PROGRAM_FILE SEARCH_FILE REPLACE_FILE  # surgical text replacement (avoids insert bugs)

HELPER LIBRARY (reusable verified lemmas):
    python {helper_path} list_helpers                       # see available helper lemmas
    python {helper_path} create_helper NAME HELPER_FILE     # verify & save a helper lemma
    python {helper_path} use_helpers PROGRAM_FILE [NAME...] # prepend helpers to program

PERSISTENCE:
    python {helper_path} read_persistence
    python {helper_path} write_persistence "your insight here"

Where PROGRAM_FILE / BODY_FILE / SEARCH_FILE / REPLACE_FILE are paths to temp files you write.

KEY TIPS:
- FIRST: Run `detect_axiom` to check if the lemma is provable. If it depends on bodyless functions, output `// AXIOM`.
- Use `verify_method` (not `execute`) to check ONLY your lemma — avoids errors from other lemmas.
- Use `verify_slow` for complex proofs that might time out with the default 30s.
- Use `edit_program` instead of `insert_body` if insertion produces malformed code.
- Use `list_helpers` / `create_helper` to build a library of reusable lemmas (e.g., mul_comm, mul_assoc).
- Use `use_helpers` to include helper lemmas in your program before verification.
- If a function is opaque, add `reveal FuncName();` in your proof.

IMPORTANT RULES:
- Write Dafny programs to temp files then pass the file path to the helper.
- When you're done, output your final proof body between // BEGIN DAFNY and // END DAFNY markers.
- If the lemma is an axiom / unprovable, output `// AXIOM` instead.
- Do NOT use any tools other than Bash (via the helper script) and Write/Read for temp files.
"""


def _build_user_prompt(program: str, lemma_name: str, errors: str) -> str:
    return f"""Implement the lemma `{lemma_name}` in this Dafny program:

```dafny
{program}
```

Current verification errors:
```
{errors}
```

Steps:
1. First, use `list_declarations` to see what lemmas/functions are available.
2. Use `inspect_function` on any function your lemma depends on — if it's opaque, you need `reveal FuncName();`. If it's an axiom or has no body, the lemma may be unprovable.
3. Optionally call `induction_sketch` to get a proof sketch.
4. Write your proof body, use `insert_body` to get the full program, then `verify_method` (NOT `execute`) to check ONLY your lemma.
5. If verification fails, refine and retry (max {MAX_VERIFICATION_ATTEMPTS} attempts).
6. When done (success or max attempts), output ONLY the final proof body between `// BEGIN DAFNY` and `// END DAFNY`.
7. If you determine the lemma is an axiom or unprovable (depends on uninterpreted functions), output `// AXIOM` instead.

Start by reading persistence memory for any useful insights from previous lemmas."""


def _make_sdk_options(system_prompt: str, max_turns: int) -> ClaudeCodeOptions:
    helper = str(_vfp_dir / "bench_claude_code_helper.py")
    # Override nesting env vars so the subprocess doesn't refuse to start.
    # The SDK merges os.environ with options.env, so we must explicitly blank them.
    env = {
        "CLAUDECODE": "",
        "CLAUDE_CODE_SSE_PORT": "",
    }
    return ClaudeCodeOptions(
        system_prompt=system_prompt.format(helper_path=helper),
        allowed_tools=["Bash", "Read", "Write"],
        max_turns=max_turns,
        model=CLAUDE_MODEL,
        permission_mode="bypassPermissions",
        cwd=str(_vfp_dir),
        env=env,
    )


async def _collect_agent_text(prompt: str, options: ClaudeCodeOptions) -> str:
    """Run SDK query and return the last text block."""
    last_text = ""
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        last_text = block.text
                        preview = block.text[:200] + "..." if len(block.text) > 200 else block.text
                        print(f"[AGENT] {preview}", flush=True)
            elif isinstance(message, ResultMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        last_text = block.text
    except Exception as e:
        print(f"[AGENT] Error: {type(e).__name__}: {e}", flush=True)
    return last_text


_AXIOM_MARKER = "// AXIOM"


async def _run_agent(program: str, lemma_name: str, errors: str) -> tuple[Optional[str], bool]:
    """Run Claude Code SDK agent for a single lemma.

    Returns (proof_body_or_None, is_axiom).
    """
    prompt = _build_user_prompt(program, lemma_name, errors)
    options = _make_sdk_options(SYSTEM_PROMPT, MAX_VERIFICATION_ATTEMPTS * 4 + 5)
    last_text = await _collect_agent_text(prompt, options)
    if last_text:
        if _AXIOM_MARKER in last_text:
            return None, True
        return driver.extract_dafny_program(last_text), False
    return None, False


# ---------------------------------------------------------------------------
# Raw-file agent: handle files where the sketcher can't extract lemmas
# ---------------------------------------------------------------------------

RAW_FILE_SYSTEM_PROMPT = """You are an expert Dafny proof engineer.

You have access to a Python helper at {helper_path} that provides Dafny tools:

VERIFICATION:
    python {helper_path} verify_method PROGRAM_FILE NAME   # verify ONLY one lemma (preferred)
    python {helper_path} verify_slow PROGRAM_FILE NAME     # verify with 120s timeout
    python {helper_path} execute PROGRAM_FILE              # verify full program

ANALYSIS:
    python {helper_path} detect_axiom PROGRAM_FILE NAME    # check if lemma is unprovable
    python {helper_path} list_declarations PROGRAM_FILE    # list all declarations with flags
    python {helper_path} inspect_function PROGRAM_FILE NAME

PROOF CONSTRUCTION:
    python {helper_path} induction_sketch PROGRAM_FILE METHOD_NAME
    python {helper_path} insert_body LEMMA_NAME ORIGINAL_PROGRAM_FILE BODY_FILE
    python {helper_path} edit_program PROGRAM_FILE SEARCH_FILE REPLACE_FILE

HELPER LIBRARY:
    python {helper_path} list_helpers / create_helper / use_helpers

PERSISTENCE:
    python {helper_path} read_persistence / write_persistence "insight"

IMPORTANT RULES:
- Write Dafny programs to temp files then pass the file path to the helper.
- You can read files with the Read tool and write with the Write tool.
- Do NOT use any tools other than Bash (via the helper script) and Write/Read for temp files.
"""

RAW_FILE_PROMPT = """Here is a Dafny file that needs lemma proofs completed:

```dafny
{program}
```

Source file: {source_file}

The Dafny Sketcher could not parse this file to extract lemmas automatically.
Your task:
1. Read persistence memory for useful insights.
2. Identify all lemmas that need proofs (empty bodies, {{:axiom}} attribute, or verification errors).
3. For each lemma, write the proof body.
4. Write the complete fixed program to a temp file and use `execute` to verify.
5. Iterate until verification succeeds or you've tried {max_attempts} times.

When done, output a summary in this exact format:

// RESULTS
// LEMMA: <lemma_name> STATUS: <solved|failed>
// LEMMA: <lemma_name> STATUS: <solved|failed>
// END RESULTS

Then output the final complete program between:
// BEGIN DAFNY
<full program>
// END DAFNY
"""


def _parse_raw_results(text: str) -> dict[str, str]:
    """Parse the // RESULTS block from agent output."""
    results = {}
    in_results = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "// RESULTS":
            in_results = True
            continue
        if stripped == "// END RESULTS":
            break
        if in_results and stripped.startswith("// LEMMA:"):
            # Parse: // LEMMA: name STATUS: solved|failed
            parts = stripped[len("// LEMMA:"):].strip()
            if "STATUS:" in parts:
                name_part, status_part = parts.rsplit("STATUS:", 1)
                name = name_part.strip()
                status = status_part.strip().lower()
                results[name] = status
    return results


async def _run_raw_file_agent(program: str, source_file: str) -> tuple[dict[str, str], Optional[str]]:
    """Run agent on a raw file. Returns (lemma_results, final_program)."""
    prompt = RAW_FILE_PROMPT.format(
        program=program,
        source_file=source_file,
        max_attempts=MAX_VERIFICATION_ATTEMPTS,
    )
    options = _make_sdk_options(RAW_FILE_SYSTEM_PROMPT, MAX_VERIFICATION_ATTEMPTS * 6 + 10)
    last_text = await _collect_agent_text(prompt, options)
    if not last_text:
        return {}, None
    results = _parse_raw_results(last_text)
    final_program = driver.extract_dafny_program(last_text)
    return results, final_program


def _handle_raw_file(source_file: str, program: str, stats: dict) -> None:
    """Handle a file where the sketcher couldn't extract lemmas."""
    global _current_lemma_name, _current_sample_index, _current_last_code

    file_key = f"__raw__{Path(source_file).stem}"

    # Resume: skip if already completed
    if file_key in _resumed_stats and file_key not in stats:
        prev = _resumed_stats[file_key]
        stats[file_key] = prev
        # Restore per-lemma results too
        for k, v in _resumed_stats.items():
            if k.startswith(f"proof_{file_key}__") or k.startswith(f"failed_proof_{file_key}__"):
                stats[k] = v
        print(f"[RESUME] Skipping raw file {source_file} (previous result: {prev})")
        return

    _current_lemma_name = file_key
    _current_sample_index = 0
    _current_last_code = ""
    print(f"[RAW] Processing file with agent: {source_file}", flush=True)

    # First: try just verifying as-is — maybe it already passes
    errs = sketcher.list_errors_for_method(program, None)
    if not errs:
        print(f"[RAW] File already verifies: {source_file}")
        stats[file_key] = -1
        save_run_state(stats)
        return

    _append_sample_event("raw_file_start", source_file=source_file,
                         errors=format_errors(errs))

    # Run the agent
    lemma_results, final_program = asyncio.run(
        _run_raw_file_agent(program, source_file)
    )

    if not lemma_results and final_program is None:
        print(f"[RAW] Agent returned nothing for {source_file}")
        stats[file_key] = 2
        save_run_state(stats)
        return

    # Verify the final program if we got one
    if final_program:
        e_final = sketcher._list_errors_for_method_core(final_program, None)
        if not e_final:
            print(f"[RAW] Agent fully solved {source_file}")
            stats[file_key] = 1
            stats[f"proof_{file_key}"] = final_program
            _save_artifact(file_key, source_file, final_program, final_program, "solved")
            _append_sample_event("raw_file_solved", source_file=source_file,
                                 program=final_program)
        else:
            print(f"[RAW] Agent partially solved {source_file} ({len(lemma_results)} lemmas reported)")
            stats[file_key] = 2
            stats[f"failed_proof_{file_key}"] = final_program
            _save_artifact(file_key, source_file, final_program, final_program, "failed")
            _append_sample_event("raw_file_partial", source_file=source_file,
                                 program=final_program, errors=format_errors(e_final))
    else:
        stats[file_key] = 2

    # Record per-lemma results from agent's summary
    for lemma_name, status in lemma_results.items():
        lkey = f"{file_key}__{lemma_name}"
        stats[lkey] = 1 if status == "solved" else 2

    save_run_state(stats)


# ---------------------------------------------------------------------------
# Core benchmark logic (mirrors bench_effectful.py)
# ---------------------------------------------------------------------------

def lemma1(lemma, p, stats):
    global _current_lemma_name, _current_sample_index, _current_last_code
    init_p = p
    name = lemma['name']

    # Resume: skip if already completed (but re-attempt unsolved=2)
    if name in _resumed_stats and name not in stats:
        prev = _resumed_stats[name]
        if prev != 2:  # Only skip solved/axiom, re-attempt unsolved
            stats[name] = prev
            for prefix in ("proof_", "failed_proof_"):
                key = prefix + name
                if key in _resumed_stats:
                    stats[key] = _resumed_stats[key]
            print(f"[RESUME] Skipping {name} (previous result: {prev})")
            return
        else:
            print(f"[RETRY-UNSOLVED] Re-attempting {name} (was unsolved)")

    _current_lemma_name = name
    _current_sample_index = 0
    _current_last_code = ""
    print('lemma', name)

    # Step -1: auto-detect axioms (bodyless functions)
    try:
        done = sketcher.sketch_done(init_p)
        if done:
            # Collect bodyless functions from sketch_done
            import re as _re
            bodyless_fns = [x['name'] for x in done
                           if x.get('type') == 'function' and x.get('status') != 'done']
            # Also scan source for standalone function declarations without bodies
            _done_names = {x['name'] for x in done}
            _lines = init_p.splitlines()
            for _i, _line in enumerate(_lines):
                _m = _re.match(r'^\s*(?:ghost\s+)?function\s+(\w+)', _line)
                if _m and _m.group(1) not in bodyless_fns and _m.group(1) not in _done_names:
                    _has_body = False
                    _depth = 0
                    for _j in range(_i, min(_i + 20, len(_lines))):
                        for _ch in _lines[_j]:
                            if _ch == '{': _depth += 1
                            elif _ch == '}': _depth -= 1
                        if _depth > 0:
                            _has_body = True
                            break
                        if _j > _i and _lines[_j].strip() == '':
                            break
                        if _j > _i and _re.match(r'^\s*(lemma|method|function|predicate|class|module)\b', _lines[_j]):
                            break
                    if not _has_body:
                        bodyless_fns.append(_m.group(1))
            # Check lemma signature + ensures for references to bodyless functions
            lines = init_p.splitlines()
            start = lemma.get('startLine', 1) - 1
            end = lemma.get('endLine', lemma.get('insertLine', start + 1))
            lemma_sig = '\n'.join(lines[start:end])
            referenced = [fn for fn in bodyless_fns if fn in lemma_sig]
            if referenced:
                print(f"[AXIOM] {name} depends on bodyless function(s): {', '.join(referenced)} — skipping")
                stats[name] = -2
                save_run_state(stats)
                return
    except Exception as e:
        print(f"[AXIOM] Detection failed for {name}: {e}")

    # Step 0: try empty proof
    xp = driver.insert_program_todo(lemma, init_p, "")
    e = sketcher.list_errors_for_method(xp, name)
    if not e and not FORCE_LLM:
        print("empty proof works")
        stats[name] = -1
        save_run_state(stats)
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
            save_run_state(stats)
            return
        if not e_ind and FORCE_LLM:
            print("inductive proof sketch works (continuing due to --force-llm)")
        p, e = p_ind, e_ind
    else:
        p = xp

    # Step 2: Claude Code SDK agent loop with retries
    global _execute_attempt_count
    _execute_attempt_count = 0

    max_outer_retries = MAX_VERIFICATION_ATTEMPTS
    best_x = None
    best_errors = None

    for attempt in range(max_outer_retries):
        errors_str = format_errors(e) if e else "No errors yet"
        if attempt > 0:
            errors_str = f"[Retry {attempt+1}/{max_outer_retries}] Previous attempt failed.\n{errors_str}"
            print(f"[RETRY] Attempt {attempt+1}/{max_outer_retries} for {name}")

        x, is_axiom = asyncio.run(_run_agent(p, name, errors_str))
        _current_last_code = x or ""
        _append_sample_event("llm_returned", code=_current_last_code, attempt=attempt+1)

        # Agent determined this is an axiom / unprovable
        if is_axiom:
            print(f"Agent identified {name} as axiom/unprovable")
            stats[name] = -2  # special code for axiom
            save_run_state(stats)
            return

        if x is None:
            print(f"LLM did not return valid Dafny (attempt {attempt+1})")
            continue

        # Verify the final result
        p_final = driver.insert_program_todo(lemma, init_p, x)
        e_final = sketcher._list_errors_for_method_core(p_final, name)
        if not e_final:
            print("LLM repair succeeded")
            stats[name] = 1
            stats['proof_' + name] = x
            _append_sample_event("lemma_solved", code=x, program=p_final)
            _save_artifact(name, _current_source_file, x, p_final, "solved")
            save_run_state(stats)
            return

        # Track best attempt (fewest errors)
        if best_errors is None or len(e_final) < len(best_errors):
            best_x = x
            best_errors = e_final

        # Feed errors back for next attempt
        p, e = p_final, e_final

    # All retries exhausted
    print("LLM repair failed - still has errors after all retries")
    x = best_x or _current_last_code
    if x:
        p_final = driver.insert_program_todo(lemma, init_p, x)
        stats[name] = 2
        stats['failed_proof_' + name] = x
        _append_sample_event("lemma_unsolved", code=x, program=p_final, errors=format_errors(best_errors or []))
        _save_artifact(name, _current_source_file, x, p_final, "failed")
    else:
        stats[name] = 2
    save_run_state(stats)


# ---------------------------------------------------------------------------
# Stats reporting (same structure as bench_effectful.py)
# ---------------------------------------------------------------------------

def print_summary_stats(stats):
    print('total for empty proof works:',
          len([v for v in stats.values() if isinstance(v, int) and v == -1]))
    print('total for axiom/unprovable:',
          len([v for v in stats.values() if isinstance(v, int) and v == -2]))
    print('total for inductive proof sketch works:',
          len([v for v in stats.values() if isinstance(v, int) and v == 0]))
    print('total for LLM repair loop works:',
          len([v for v in stats.values() if isinstance(v, int) and v == 1]))
    print('total for unsolved:',
          len([v for v in stats.values() if isinstance(v, int) and v == 2]))


def _summary_counts(stats):
    return {
        "empty_proof_works": len([v for v in stats.values() if isinstance(v, int) and v == -1]),
        "axiom_unprovable": len([v for v in stats.values() if isinstance(v, int) and v == -2]),
        "inductive_proof_sketch_works": len([v for v in stats.values() if isinstance(v, int) and v == 0]),
        "llm_repair_loop_works": len([v for v in stats.values() if isinstance(v, int) and v == 1]),
        "unsolved": len([v for v in stats.values() if isinstance(v, int) and v == 2]),
    }


def save_run_state(stats):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "llm_model": CLAUDE_MODEL,
            "use_sketchers": USE_SKETCHERS,
            "max_verification_attempts": MAX_VERIFICATION_ATTEMPTS,
            "force_llm": FORCE_LLM,
            "persistence_out": PERSISTENCE_PATH,
            "samples_out": SAMPLES_OUT_PATH,
            "artifacts_dir": ARTIFACTS_DIR,
        },
        "summary": _summary_counts(stats),
        "persistence_memory": list(persistence_memory),
        "stats": stats,
    }
    try:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
    except Exception as e:
        print(f"failed to save run state: {e}")
    try:
        snapshot_path = str(Path(OUT_PATH).with_suffix("")) + "_persistence.json"
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "persistence_memory": list(persistence_memory),
                },
                f, indent=2, ensure_ascii=True,
            )
    except Exception as e:
        print(f"failed to save persistence snapshot: {e}")


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
    save_run_state(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default=CLAUDE_MODEL,
                        help='Claude model to use (overrides CLAUDE_MODEL env)')
    parser.add_argument('--force-llm', action='store_true',
                        help='Always run LLM stage even if empty/sketch proof succeeds')
    parser.add_argument('--out', type=str, default=DEFAULT_OUT_PATH,
                        help='Path to JSON file where final run state is saved')
    parser.add_argument('--samples-out', type=str, default=DEFAULT_SAMPLES_OUT_PATH,
                        help='Path to JSONL file where per-iteration samples are appended')
    parser.add_argument('--persistence-out', type=str, default=DEFAULT_PERSISTENCE_PATH,
                        help='Path to JSONL persistence memory file')
    parser.add_argument('--persistence-in', type=str, default=None,
                        help='Path to persistence JSONL to load from a previous run')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a previous run JSON to resume from')
    parser.add_argument('--artifacts-dir', type=str, default=DEFAULT_ARTIFACTS_DIR,
                        help='Directory to save per-lemma .dfy artifacts')
    args, remaining = parser.parse_known_args()
    CLAUDE_MODEL = args.model
    OUT_PATH = args.out
    SAMPLES_OUT_PATH = args.samples_out
    PERSISTENCE_PATH = args.persistence_out
    ARTIFACTS_DIR = args.artifacts_dir
    FORCE_LLM = args.force_llm

    # Auto-resume
    resume_path = args.resume
    if resume_path is None and Path(OUT_PATH).exists():
        resume_path = OUT_PATH
        print(f"[RESUME] Auto-resuming from existing {OUT_PATH}")
    if resume_path:
        _resumed_stats.update(_load_resume_state(resume_path))

    # Load persistence memory
    persistence_in = args.persistence_in
    if persistence_in is None and Path(PERSISTENCE_PATH).exists():
        persistence_in = PERSISTENCE_PATH
    if persistence_in:
        _load_persistence_from_jsonl(persistence_in)

    # Wrap bench_driver.main1 to:
    # 1. Track current source file
    # 2. Fall back to raw-file agent when no lemmas are extracted
    _orig_main1 = bench_driver.main1
    def _tracking_main1(lemma1_fn, f, stats, lemma_names=None):
        global _current_source_file
        _current_source_file = f
        p = tests.read_file(f)
        done = sketcher.sketch_done(p)
        lemmas = [x for x in (done or []) if isinstance(x, dict) and x.get('type') == 'lemma']
        if not lemmas:
            try:
                todo_lemmas = sketcher.sketch_todo_lemmas(p)
            except Exception:
                todo_lemmas = []
            if isinstance(todo_lemmas, list):
                lemmas = [x for x in todo_lemmas if isinstance(x, dict) and x.get('type') == 'lemma']
        if not lemmas:
            # Sketcher can't extract lemmas — give the whole file to the agent
            print(f"[RAW] No lemmas extracted by sketcher for {f}, falling back to agent")
            _handle_raw_file(f, p, stats)
            return
        return _orig_main1(lemma1_fn, f, stats, lemma_names=lemma_names)
    bench_driver.main1 = _tracking_main1

    sys.argv = [sys.argv[0]] + remaining
    bench_driver.run(lemma1, print_stats)
