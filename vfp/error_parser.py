"""Dafny error parser and proof obligation extractor.

Parses Dafny verifier output into structured error objects with categories,
and extracts proof obligations from error messages to guide proof construction.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DafnyError:
    """A structured Dafny verification error."""
    line: int
    col: int
    message: str
    snippet: str = ""
    category: str = "unknown"  # postcondition, precondition, assertion, decreases, type, resolution, timeout
    related_expression: str = ""
    suggestion: str = ""


# Error message patterns → (category, suggestion_template)
_ERROR_PATTERNS = [
    (r"a postcondition could not be proved on this return path",
     "postcondition",
     "Add assertions that build up to the postcondition. Try case analysis or induction."),
    (r"this postcondition could not be proved on a return path",
     "postcondition",
     "This specific ensures clause failed. Focus your proof on establishing this property."),
    (r"assertion might not hold",
     "assertion",
     "The assertion you added is not provable at this point. Add intermediate steps or weaken it."),
    (r"cannot prove termination",
     "decreases",
     "Add a `decreases` clause, or the recursive structure may need adjustment."),
    (r"decreases expression .* might not decrease",
     "decreases",
     "The decreases measure doesn't shrink on recursive calls. Check the recursive argument."),
    (r"a precondition for this call could not be proved",
     "precondition",
     "A function/lemma call requires a precondition you haven't established. Check `requires` clauses."),
    (r"function precondition could not be proved",
     "precondition",
     "A function's `requires` clause isn't satisfied. Establish it before the call."),
    (r"index out of range",
     "precondition",
     "Array/sequence index may be out of bounds. Add a bounds assertion."),
    (r"resolution/type errors? detected",
     "resolution",
     "There are name resolution or type errors. Check that all names and types are correct."),
    (r"Error: unresolved identifier",
     "resolution",
     "An identifier is not in scope. Check spelling and imports."),
    (r"Error: member .* does not exist",
     "resolution",
     "A member access failed. Check the type and available members."),
    (r"timed out",
     "timeout",
     "Verification timed out. Simplify the proof, add intermediate assertions, or use `verify_slow`."),
    (r"calc step .* might not hold",
     "calc",
     "A step in your calc block is not provable. Break it into smaller steps or add a hint."),
    (r"cannot establish the existence of LHS values",
     "exists",
     "An existential quantifier needs a witness. Provide one with `var x :| ...` or explicit construction."),
    (r"forall .* might not hold",
     "forall",
     "A universal quantifier assertion failed. You may need `forall x :: ... { body }` with a proof body."),
]


def categorize_error(message: str) -> tuple[str, str]:
    """Categorize an error message and return (category, suggestion)."""
    for pattern, category, suggestion in _ERROR_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            return category, suggestion
    return "unknown", ""


def parse_dafny_errors(verifier_output: str) -> list[DafnyError]:
    """Parse raw Dafny verifier output into structured DafnyError objects.

    Args:
        verifier_output: Raw output from `dafny verify`

    Returns:
        List of DafnyError objects with categories and suggestions
    """
    errors = []
    lines = verifier_output.splitlines()

    for i, line in enumerate(lines):
        # Match: /path/to/file.dfy(line,col): Error: message
        m = re.match(r'.*\.dfy\((\d+),(\d+)\):\s*Error:\s*(.*)', line)
        if not m:
            # Also match: /path/to/file.dfy(line,col): Warning: message
            m = re.match(r'.*\.dfy\((\d+),(\d+)\):\s*Warning:\s*(.*)', line)
        if not m:
            continue

        line_num = int(m.group(1))
        col_num = int(m.group(2))
        message = m.group(3).strip()

        # Extract code snippet from following lines
        snippet = ""
        for j in range(i + 1, min(i + 5, len(lines))):
            if f"{line_num} |" in lines[j]:
                snippet_start = lines[j].find(f"{line_num} |") + len(f"{line_num} |")
                snippet = lines[j][snippet_start:].strip()
                break

        # Extract related expression from "Related location" lines
        related = ""
        for j in range(i + 1, min(i + 8, len(lines))):
            if "Related location:" in lines[j] or "related location:" in lines[j]:
                rm = re.search(r'\.dfy\(\d+,\d+\):\s*Related location:\s*(.*)', lines[j])
                if rm:
                    related = rm.group(1).strip()
                break

        category, suggestion = categorize_error(message)

        errors.append(DafnyError(
            line=line_num,
            col=col_num,
            message=message,
            snippet=snippet,
            category=category,
            related_expression=related,
            suggestion=suggestion,
        ))

    # Check for global errors (resolution, infra failures)
    if not errors:
        low = verifier_output.lower()
        if "resolution/type errors detected" in low or "error:" in low:
            errors.append(DafnyError(
                line=0, col=0,
                message=verifier_output.strip()[:500],
                category="resolution",
                suggestion="Fix resolution/type errors before attempting proofs.",
            ))

    return errors


@dataclass
class ProofObligation:
    """A proof obligation extracted from errors — what needs to be proved."""
    lemma_name: str
    obligation_type: str  # e.g., "postcondition", "assertion", "precondition"
    description: str
    line: int = 0
    suggestion: str = ""


def extract_proof_obligations(errors: list[DafnyError], lemma_name: str = "") -> list[ProofObligation]:
    """Extract proof obligations from parsed errors.

    Groups and deduplicates errors into actionable proof obligations.
    """
    obligations = []
    seen = set()

    for err in errors:
        # Create a dedup key
        key = (err.category, err.line, err.message[:80])
        if key in seen:
            continue
        seen.add(key)

        if err.category in ("postcondition", "assertion", "precondition", "calc", "exists", "forall"):
            obligations.append(ProofObligation(
                lemma_name=lemma_name,
                obligation_type=err.category,
                description=err.message,
                line=err.line,
                suggestion=err.suggestion,
            ))
        elif err.category == "decreases":
            obligations.append(ProofObligation(
                lemma_name=lemma_name,
                obligation_type="termination",
                description=err.message,
                line=err.line,
                suggestion=err.suggestion,
            ))
        elif err.category == "timeout":
            obligations.append(ProofObligation(
                lemma_name=lemma_name,
                obligation_type="complexity",
                description="Proof search timed out — proof may be too complex",
                line=err.line,
                suggestion=err.suggestion,
            ))

    return obligations


def format_proof_obligations(obligations: list[ProofObligation]) -> str:
    """Format proof obligations as a human-readable string for the agent."""
    if not obligations:
        return "No proof obligations found."

    parts = []
    for i, ob in enumerate(obligations, 1):
        s = f"{i}. [{ob.obligation_type.upper()}] {ob.description}"
        if ob.line:
            s += f" (line {ob.line})"
        if ob.suggestion:
            s += f"\n   Hint: {ob.suggestion}"
        parts.append(s)

    return "\n".join(parts)


def format_errors_structured(errors: list[DafnyError]) -> str:
    """Format parsed errors with categories and suggestions."""
    if not errors:
        return "No errors."

    parts = []
    for err in errors:
        s = f"Line {err.line},{err.col}: [{err.category}] {err.message}"
        if err.snippet:
            s += f"\n  Code: {err.snippet}"
        if err.suggestion:
            s += f"\n  Suggestion: {err.suggestion}"
        if err.related_expression:
            s += f"\n  Related: {err.related_expression}"
        parts.append(s)

    return "\n".join(parts)
