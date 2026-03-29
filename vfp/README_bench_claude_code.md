# Claude Code Agent Bench (`bench_claude_code.py`)

Automated Dafny proof benchmark using Claude Code SDK. Runs on a Claude Max
subscription (no API key needed). Produces the same JSON report structure as
`bench_effectful.py`.

## Prerequisites

- Python 3.10+ with the project venv:
  ```
  /Users/nguyendat/Marc/dafny-sketcher/.venv/bin/python
  ```
- `claude-code-sdk` installed (`pip install claude-code-sdk`)
- Claude Code CLI installed and authenticated (Claude Max)
- Dafny binary configured via `.env` or `DAFNY` env var

## Quick Start

```bash
cd /Users/nguyendat/Marc/dafny-sketcher/vfp

# Single file
python bench_claude_code.py --file bench/binary_search_solution.dfy

# Full bench suite (default: bench/*_solution.dfy)
python bench_claude_code.py

# DafnyBench suite
python bench_claude_code.py --glob-pattern "DafnyBench/*.dfy" \
  --out bench_claude_code_dafnybench.json \
  --samples-out bench_claude_code_dafnybench_samples.jsonl \
  --persistence-out bench_claude_code_dafnybench_persistence.jsonl \
  --artifacts-dir bench_claude_code_dafnybench_artifacts
```

Use the project venv if running outside an activated environment:
```bash
/Users/nguyendat/Marc/dafny-sketcher/.venv/bin/python bench_claude_code.py ...
```

## Long-running (tmux)

```bash
tmux new-session -d -s dafnybench \
  '/Users/nguyendat/Marc/dafny-sketcher/.venv/bin/python -u bench_claude_code.py \
    --glob-pattern "DafnyBench/*.dfy" \
    --out bench_claude_code_dafnybench.json \
    --samples-out bench_claude_code_dafnybench_samples.jsonl \
    --persistence-out bench_claude_code_dafnybench_persistence.jsonl \
    --artifacts-dir bench_claude_code_dafnybench_artifacts \
    2>&1 | tee bench_claude_code_dafnybench_run.log'

# Monitor
tmux attach -t dafnybench
# or
tail -f bench_claude_code_dafnybench_run.log
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `claude-sonnet-4-5` | Claude model to use |
| `--file` | — | Run on a single .dfy file |
| `--glob-pattern` | `bench/*_solution.dfy` | Glob pattern for input files |
| `--out` | `bench_claude_code_latest.json` | Output JSON report path |
| `--samples-out` | `bench_claude_code_samples_latest.jsonl` | Per-iteration JSONL log |
| `--persistence-out` | `bench_claude_code_persistence_latest.jsonl` | Persistence memory JSONL |
| `--persistence-in` | — | Load persistence from a previous run's JSONL |
| `--resume` | — | Resume from a previous run's JSON (auto-detects if `--out` exists) |
| `--artifacts-dir` | `bench_claude_code_artifacts_<timestamp>` | Per-lemma .dfy artifact directory |
| `--force-llm` | off | Always run agent even if empty/sketch proof succeeds |
| `--skip-file` | — | Skip specific files (passed to bench_driver) |
| `--on-track` | — | Only process specified files (passed to bench_driver) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SKETCHERS` | `true` | Set to `false` to disable induction sketch step |
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | Model (overridden by `--model`) |
| `MAX_VERIFICATION_ATTEMPTS` | `5` | Max `execute()` failures per lemma |
| `DAFNY` | auto-detected | Path to Dafny.dll |

## How It Works

### Three-stage pipeline (per lemma)

1. **Empty proof** (stat `-1`): Try verifying with an empty body.
2. **Induction sketch** (stat `0`): Use the Dafny Sketcher to generate a non-LLM induction proof.
3. **Agent loop** (stat `1` solved, `2` unsolved): Claude Code agent iterates with `execute`, `induction_sketch`, and `insert_body` tools.

### Raw-file fallback

When the Dafny Sketcher can't extract lemmas (e.g., timeout under batch load), the entire file is given to the agent. The agent has full access to all sketcher tools and works on the complete file.

### Auto-resume

- On startup, if `--out` file already exists, it loads previous results and skips completed lemmas.
- State is saved after every lemma, so crashes don't lose progress.

### Persistence memory

- The agent can read/write persistent insights via `read_persistence` / `write_persistence`.
- Insights are saved to JSONL and loaded on the next run, enabling cross-lemma and cross-run learning.

## Output Files

- **JSON report** (`--out`): Full results with config, summary counts, persistence memory, and per-lemma stats.
- **Samples JSONL** (`--samples-out`): Per-iteration events (agent calls, results, errors).
- **Persistence JSONL** (`--persistence-out`): Agent-written insights for reuse.
- **Artifacts dir** (`--artifacts-dir`): Individual `.dfy` files for each solved/failed lemma.
- **Persistence snapshot** (`<out>_persistence.json`): Current persistence memory state.

## Helper Script

`bench_claude_code_helper.py` is the CLI bridge between Claude Code and the Dafny Sketcher:

```bash
python bench_claude_code_helper.py execute PROGRAM_FILE
python bench_claude_code_helper.py induction_sketch PROGRAM_FILE METHOD_NAME
python bench_claude_code_helper.py insert_body LEMMA_NAME ORIGINAL_FILE BODY_FILE
python bench_claude_code_helper.py read_persistence
python bench_claude_code_helper.py write_persistence "your insight here"
```
