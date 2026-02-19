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
  - counterexamples: Find counterexamples for a lemma (requires lemma name to be passed to --method)
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


class LemmaInsertLineTool(Tool):
    """Get the line number where to insert a lemma body (the opening brace line)."""

    name = "lemma_insert_line"
    description = """Returns the line number of the opening brace for a lemma body.
    This is the line where you would insert the proof body for a lemma."""

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the .dfy file",
            },
            "lemma_name": {
                "type": "string",
                "description": "Name of the lemma to find the insert line for",
            },
        },
        "required": ["path", "lemma_name"],
    }
    requires_permission = True

    def execute(self, path: str, lemma_name: str) -> str:
        try:
            # Get todo_lemmas to find the lemma
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
            
            # Find the lemma with matching name
            for todo in todos:
                if todo.get('name') == lemma_name and todo.get('type') == 'lemma':
                    insert_line = todo.get('insertLine')
                    if insert_line is not None:
                        return str(insert_line)
                    else:
                        return f"[error: lemma '{lemma_name}' found but missing insertLine field]"
            
            return f"[error: lemma '{lemma_name}' not found in todo_lemmas]"
            
        except FileNotFoundError:
            return f"[error: dotnet or CLI not found. Ensure dotnet is installed and CLI is built at {CLI_DLL}]"
        except subprocess.TimeoutExpired:
            return "[error: dafny-sketcher timed out after 120 seconds]"
        except Exception as e:
            return f"[error: {e}]"


# Tools to add
TOOLS = [DafnySketcherTool(), LemmaInsertLineTool()]

# Make dafny_sketcher path-based (per-path "always")
PATH_BASED = {"dafny_sketcher", "lemma_insert_line"}

# Auto-allow dafny_sketcher within cwd
AUTO_ALLOW_CWD = {"dafny_sketcher", "lemma_insert_line"}
