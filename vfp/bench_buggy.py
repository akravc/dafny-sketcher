"""
Benchmark script for buggy Dafny programs using counterexample-guided repair.

Runs the driver workflow on all *_buggy.dfy files in the bench/ directory,
limiting the LLM repair loop to at most 5 iterations per program.
Records which programs are solved vs unsolved.
"""

import glob
import os
import driver
import sketcher
import tests


def run_buggy_benchmark(max_iterations: int = 5):
    """Run the driver workflow on all buggy files in bench/."""
    # Find all *_buggy.dfy files
    buggy_files = sorted(glob.glob("bench/*_buggy.dfy"))
    
    print(f"Found {len(buggy_files)} buggy files:")
    for f in buggy_files:
        print(f"  - {f}")
    print()
    
    solved = []
    unsolved = []
    results = {}
    
    for filepath in buggy_files:
        filename = os.path.basename(filepath)
        print('=' * 60)
        print(f'Processing: {filename}')
        print('=' * 60)
        
        # Read the file content
        p = tests.read_file(filepath)
        
        # Extract lemma names from the buggy file
        done = sketcher.sketch_done(p)
        lemmas = [x for x in done if x['type'] == 'lemma'] if done else []
        lemma_names = [l['name'] for l in lemmas]
        
        # Create a fresh cache for each file
        cache = driver.Cache()
        
        # Run the driver with max_iterations limit
        try:
            result_p = driver.drive_program(p, max_iterations=max_iterations, cache=cache)
            print("\nRESULT PROGRAM")
            print(result_p)
            
            # Check if there are remaining todos (lemmas not yet implemented)
            remaining_todo = sketcher.sketch_next_todo(result_p)
            
            # Check if ALL lemmas verify successfully (no errors for any)
            errors = []
            for lemma_name in lemma_names:
                errors.extend(sketcher.list_errors_for_method(result_p, lemma_name))
            
            if remaining_todo is not None:
                print(f"\n✗ UNSOLVED: {filename} (remaining todo: {remaining_todo['name']})")
                print(f"  Remaining todos: {remaining_todo}")
                unsolved.append(filename)
                results[filename] = {
                    'status': 'unsolved',
                    'remaining_todo': remaining_todo['name'],
                    'final_program': result_p
                }
            elif not errors:
                print(f"\n✓ SOLVED: {filename}")
                solved.append(filename)
                results[filename] = {
                    'status': 'solved',
                    'solution': result_p
                }
            else:
                print(f"\n✗ UNSOLVED: {filename} ({len(errors)} verification errors)")
                print(f"  Errors: {errors}")
                unsolved.append(filename)
                results[filename] = {
                    'status': 'unsolved',
                    'verification_errors': len(errors),
                    'final_program': result_p
                }
        except Exception as e:
            print(f"\n✗ ERROR: {filename}: {e}")
            unsolved.append(filename)
            results[filename] = {
                'status': 'error',
                'error': str(e)
            }
        
        print()
    
    # Print summary
    print_summary(solved, unsolved, results)
    
    return results


def print_summary(solved, unsolved, results):
    """Print a summary of the benchmark results."""
    print('=' * 60)
    print('BENCHMARK SUMMARY')
    print('=' * 60)
    print(f"Total files: {len(solved) + len(unsolved)}")
    print(f"Solved: {len(solved)}")
    print(f"Unsolved: {len(unsolved)}")
    print()
    
    if solved:
        print("Solved programs:")
        for f in solved:
            print(f"  ✓ {f}")
    print()
    
    if unsolved:
        print("Unsolved programs:")
        for f in unsolved:
            status = results[f]['status']
            if status == 'unsolved':
                if 'remaining_todo' in results[f]:
                    todo = results[f]['remaining_todo']
                    print(f"  ✗ {f} (remaining todo: {todo})")
                else:
                    verr = results[f].get('verification_errors', 0)
                    print(f"  ✗ {f} ({verr} verification errors)")
            else:
                error = results[f].get('error', 'unknown error')
                print(f"  ✗ {f} (error: {error})")
    print()
    print('=' * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run buggy Dafny benchmark')
    parser.add_argument('--max-iterations', type=int, default=5,
                        help='Maximum LLM repair iterations per program (default: 5)')
    args = parser.parse_args()
    
    # Run on all buggy files
    run_buggy_benchmark(max_iterations=args.max_iterations)
