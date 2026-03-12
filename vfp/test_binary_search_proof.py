#!/usr/bin/env python3
"""Test different proof bodies for binarySearchCorrect."""

import sys
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

import sketcher
import driver
from fine import format_errors

# Read the base program
with open("temp_binary_search.dfy", "r") as f:
    base_program = f.read()

# Test different proof bodies
proof_attempts = [
    # Attempt 1: Just call the helper lemma
    """binarySearchHelperCorrect(s, key, 0, |s|);""",

    # Attempt 2: Call helper and add assertion about 'in'
    """binarySearchHelperCorrect(s, key, 0, |s|);
    var idx := binarySearch(s, key);
    if idx == |s| {
        assert forall i :: 0 <= i < |s| ==> s[i] != key;
    }""",

    # Attempt 3: More explicit
    """binarySearchHelperCorrect(s, key, 0, |s|);
    var idx := binarySearch(s, key);
    if idx == |s| {
        assert forall i :: 0 <= i < |s| ==> s[i] != key;
        assert key !in s;
    }""",
]

lemma_name = "binarySearchCorrect"

# Find the lemma metadata
done = sketcher.sketch_done(base_program)
if done is None:
    print("Error: could not parse program")
    sys.exit(1)

lemma = next((x for x in done if x['name'] == lemma_name), None)
if lemma is None:
    print(f"Error: lemma '{lemma_name}' not found")
    sys.exit(1)

print(f"Testing {len(proof_attempts)} proof attempts for {lemma_name}")
print("=" * 80)

for i, proof_body in enumerate(proof_attempts, 1):
    print(f"\nAttempt {i}:")
    print(f"Proof body:\n{proof_body}")
    print("-" * 80)

    # Insert the proof body
    program = driver.insert_program_todo(lemma, base_program, proof_body)

    # Check for errors
    errors = sketcher.list_errors_for_method(program, lemma_name)

    if not errors:
        print("✓ VERIFICATION SUCCEEDED!")
        print("\nFinal program:")
        print(program)
        print("\n" + "=" * 80)
        print("SUCCESS! Proof body:")
        print("// BEGIN DAFNY")
        print(proof_body)
        print("// END DAFNY")
        sys.exit(0)
    else:
        print("✗ Verification failed:")
        print(format_errors(errors))

print("\n" + "=" * 80)
print("All attempts failed.")
