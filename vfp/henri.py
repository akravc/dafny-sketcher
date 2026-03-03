"""DafnySketcher tool hook for Henri.

Usage:
    henri --hook /path/to/dafny-sketcher/vfp/henri.py

Adds dafny_sketcher tool for proof sketching assistance.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
import importlib.util, sys as _sys
_spec = importlib.util.spec_from_file_location(
    "dafny_utils", Path(__file__).with_name("dafny_utils.py"))
utils = importlib.util.module_from_spec(_spec)
_sys.modules[_spec.name] = utils
_spec.loader.exec_module(utils)

from henri.tools.base import Tool

# Path to the CLI DLL
CLI_DLL = os.environ.get(
    'DAFNY_SKETCHER_CLI_DLL_PATH',
    str(Path(__file__).parent.parent / 'cli' / 'bin' / 'Release' / 'net8.0' / 'DafnySketcherCli.dll')
)


class DafnySketcherTool(Tool):
    """Run dafny-sketcher CLI for proof assistance."""

    name = "dafny_sketcher"
    description = """Run dafny-sketcher to assist with Dafny proofs. Available sketch types:
  - errors: Show verification errors in the file
  - errors_warnings: Show errors and warnings
  - todo: List unimplemented functions/lemmas (JSON)
  - done: List implemented units (JSON)
  - todo_lemmas: List lemmas with errors (JSON)
  - induction_search: Generate induction proof sketch for a lemma (requires lemma name to be passed to --method)
  - counterexamples: Find counterexamples for a lemma, if you suspect the lemma does not hold (requires lemma name to be passed to --method)
  - proof_lines: List proof lines in file or lemma (JSON)"""

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the .dfy file",
            },
            "sketch": {
                "type": "string",
                "description": "Sketch type: errors, errors_warnings, todo, done, todo_lemmas, induction_search, counterexamples, proof_lines",
            },
            "method": {
                "type": "string",
                "description": "Method name (required for induction_search, counterexamples)",
            },
        },
        "required": ["path", "sketch"],
    }
    requires_permission = True

    def execute(self, path: str, sketch: str, method: str = None) -> str:
        try:
            # Build command
            cmd = ["dotnet", CLI_DLL, "--file", path, "--sketch", sketch]
            if method:
                cmd.extend(["--method", method])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"
            return output or "(no output)"
        except FileNotFoundError:
            return f"[error: dotnet or CLI not found. Ensure dotnet is installed and CLI is built at {CLI_DLL}]"
        except subprocess.TimeoutExpired:
            return "[error: dafny-sketcher timed out after 120 seconds]"
        except Exception as e:
            return f"[error: {e}]"


class InsertLemmaBodyTool(Tool):
    """Insert or replace the body of a lemma in a Dafny file."""

    name = "insert_lemma_body"
    description = """Insert or replace the body of a lemma in a Dafny file.
    The given body string is treated as the lemma body (without outer braces)
    and will replace the entire existing body, or be inserted if the lemma has no body yet."""

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the .dfy file",
            },
            "lemma_name": {
                "type": "string",
                "description": "Name of the lemma whose body should be inserted/replaced",
            },
            "lemma_body": {
                "type": "string",
                "description": "Lemma body to insert (without outer braces)",
            },
        },
        "required": ["path", "lemma_name", "lemma_body"],
    }
    requires_permission = True

    def execute(self, path: str, lemma_name: str, lemma_body: str) -> str:
        try:
            # Get todo_lemmas to find the lemma and its position
            cmd = ["dotnet", CLI_DLL, "--file", path, "--sketch", "todo_lemmas"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                return f"[error: dafny-sketcher failed: {result.stderr}]"

            # Parse JSON output
            try:
                todos = json.loads(result.stdout)
            except json.JSONDecodeError:
                return f"[error: failed to parse JSON output: {result.stdout}]"

            if not isinstance(todos, list):
                return f"[error: unexpected todo_lemmas format ({type(todos).__name__}): {result.stdout[:200]}]"

            # Find the lemma with matching name
            target = None
            for todo in todos:
                if todo.get("name") == lemma_name and todo.get("type") == "lemma":
                    target = todo
                    break

            if target is None:
                return f"[error: lemma '{lemma_name}' not found in todo_lemmas]"

            # Read the current file contents
            try:
                with open(path, "r") as f:
                    program = f.read()
            except OSError as e:
                return f"[error: could not read file: {e}]"

            # Use the same insertion logic as the VFP driver
            new_program = utils.insert_program_todo_helper(target, program, lemma_body)

            # Write back the modified program
            try:
                with open(path, "w") as f:
                    f.write(new_program)
            except OSError as e:
                return f"[error: could not write file: {e}]"

            return f"[inserted lemma body for '{lemma_name}' in {path}]"

        except FileNotFoundError:
            return f"[error: dotnet or CLI not found. Ensure dotnet is installed and CLI is built at {CLI_DLL}]"
        except subprocess.TimeoutExpired:
            return "[error: dafny-sketcher timed out after 120 seconds]"
        except Exception as e:
            return f"[error: {e}]"


class FailingCaseSnippetTool(Tool):
    """Extract the code snippet for a failing case at an error line (e.g. inductive/structural breakdown)."""

    name = "failing_case_snippet"
    description = """Extract the code for a failing case when there is an inductive or structural breakdown.
    Given a line number where the error shows up: returns the code from that line until the matching closing brace.
    If the error line has no open curly brace, returns just that line."""

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the .dfy file",
            },
            "line": {
                "type": "integer",
                "description": "Line number where the error shows up (1-based)",
            },
        },
        "required": ["path", "line"],
    }
    requires_permission = True

    def execute(self, path: str, line) -> str:
        try:
            line_no = int(line)
        except (TypeError, ValueError):
            return f"[error: line must be an integer, got {type(line).__name__}]"
        return utils.extract_snippet_from_line(path, line_no)


# Tools to add
TOOLS = [DafnySketcherTool(), InsertLemmaBodyTool(), FailingCaseSnippetTool()]

# Make dafny_sketcher path-based (per-path "always")
PATH_BASED = {"dafny_sketcher", "insert_lemma_body", "failing_case_snippet"}

# Auto-allow dafny_sketcher within cwd
AUTO_ALLOW_CWD = {"dafny_sketcher", "insert_lemma_body", "failing_case_snippet"}
