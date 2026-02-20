import glob
import os
import tests
import sketcher

def main(lemma1, print_stats, solution_files=None, lemma_names=None, glob_pattern=None, skip_files=None):
    stats = {}
    if glob_pattern is None:
        glob_pattern = "bench/*_solution.dfy"
    if not solution_files:
        solution_files = sorted(glob.glob(glob_pattern))
        if glob_pattern == "bench/*_solution.dfy":
            solution_files = [f for f in solution_files if os.path.basename(f)[0].islower()]
    if skip_files:
        solution_files = [f for f in solution_files if f not in skip_files]
    print(len(solution_files))
    print(solution_files)
    for f in solution_files:
        main1(lemma1, f, stats, lemma_names=lemma_names)
    print_stats(stats)

def main1(lemma1, f, stats, lemma_names=None):
    print('---------- Looking at file: ', f, '--------------')
    p = tests.read_file(f)
    if False:
        e = sketcher.show_errors(p)
        if e is not None:
            print('ERRORS')
            print(e)
    done = sketcher.sketch_done(p)
    lemmas = [x for x in (done or []) if isinstance(x, dict) and x.get('type') == 'lemma']
    # Fallback for files with TODO/empty bodies where sketch_done is empty.
    if not lemmas:
        try:
            todo_lemmas = sketcher.sketch_todo_lemmas(p)
        except Exception as e:
            print(f"todo_lemmas failed: {e}")
            todo_lemmas = []
        # sketch_todo_lemmas can return the sentinel string "errors" when
        # verifier errors exist but cannot be mapped to enclosing methods.
        if todo_lemmas == "errors":
            try:
                todos = sketcher.sketch_todo(p)
            except Exception as e:
                print(f"todo fallback failed: {e}")
                todos = []
            lemmas = [x for x in (todos or []) if isinstance(x, dict) and x.get('type') == 'lemma']
        else:
            lemmas = [x for x in (todo_lemmas or []) if isinstance(x, dict) and x.get('type') == 'lemma']
    if not lemmas:
        print("No lemmas discovered (done/todo_lemmas/todo all empty).")
        return
    for lemma in lemmas:
        if not lemma_names or lemma['name'] in lemma_names:
            try:
                lemma1(lemma, p, stats)
            except Exception as e:
                print(f"Error processing lemma {lemma['name']}: {e}")

def run(lemma1, print_stats, only_lemmas=None, on_track=False):
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the bench suite')
    parser.add_argument('--file', type=str, action='append', help='Path to Dafny file to process (can be used multiple times)')
    parser.add_argument('--lemma', type=str, action='append', help='Name of lemma to process (can be used multiple times)')
    parser.add_argument('--skip-file', type=str, action='append', help='Path to Dafny file to skip (can be used multiple times)')
    parser.add_argument('--on-track', action='store_true', help='Only run on track lemmas (for default glob pattern)')
    parser.add_argument('--for-poetry', action='store_true', help='Only run on the lemmas suitable for poetry')
    parser.add_argument('--glob-pattern', type=str, help='Glob pattern for solution files (default: "bench/*_solution.dfy")')
    
    args = parser.parse_args()

    if only_lemmas is None and args.glob_pattern is None and args.on_track:
        assertions_and_forall_needed = ["dedupCorrect", "maxIsCorrect"]
        helper_calls_needed = ["flattenCorrect", "HeapHeightBound", "reverseAppend", "ReverseAppend", "reverse_involutes"]
        both_assertions_and_helper_calls_needed = ["DequeueCorrect", "sumDistributive", "reverseReverse", "ReverseReverse", "reverse_append"]
        only_lemmas = assertions_and_forall_needed + helper_calls_needed + both_assertions_and_helper_calls_needed
    if only_lemmas is None and args.glob_pattern is None and args.for_poetry:
        only_lemmas = ['binarySearchCorrect', 'flattenCorrect', 'SingleNodeAcyclic', 'HeapHeightBound', 'ReverseLength', 'ReverseReverse', 'GcdDividesBoth', 'reverseAppend', 'reverseReverse', 'ReverseAppend', 'QueueSizeProperty', 'RevAcc_Helper', 'RevAcc_Correct'] 
    main(lemma1, print_stats, args.file, args.lemma or only_lemmas, args.glob_pattern, args.skip_file)

