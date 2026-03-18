"""Dependency graph and topological solver for Dafny lemmas.

Builds a dependency graph from a Dafny program's declarations and computes
the optimal order to solve lemmas — solving dependencies first so later
lemmas can rely on already-proved results.
"""

import re
from collections import defaultdict
from typing import Optional

import sketcher


def build_dependency_graph(program: str) -> dict[str, set[str]]:
    """Build a dependency graph: name → set of names it depends on.

    Analyzes which declarations reference which other declarations
    by scanning signature, requires/ensures, and body text.
    """
    done = sketcher.sketch_done(program)
    if not done:
        return {}

    lines = program.splitlines()

    # Collect all declaration names
    all_names = {item['name'] for item in done if 'name' in item}

    # Build adjacency: for each decl, find which other decls it references
    graph = {}
    for item in done:
        name = item.get('name', '')
        if not name:
            continue

        start = item.get('startLine', 1) - 1
        end = item.get('endLine', start + 1)
        decl_text = '\n'.join(lines[start:end])

        # Find all identifiers in the declaration text
        identifiers = set(re.findall(r'\b([A-Za-z_]\w*)\b', decl_text))

        # Dependencies = other declarations referenced (excluding self)
        deps = identifiers & all_names - {name}
        graph[name] = deps

    return graph


def topological_order(graph: dict[str, set[str]]) -> list[str]:
    """Compute a topological ordering of the dependency graph.

    Uses Kahn's algorithm. If there are cycles, nodes in cycles
    are appended at the end in arbitrary order.
    """
    # Compute in-degree
    in_degree = defaultdict(int)
    for node in graph:
        if node not in in_degree:
            in_degree[node] = 0
        for dep in graph[node]:
            if dep in graph:  # Only count deps that are in the graph
                in_degree[dep]  # ensure exists
                in_degree[node] += 0  # node itself

    # Actually compute in-degrees properly
    in_degree = {node: 0 for node in graph}
    for node, deps in graph.items():
        for dep in deps:
            if dep in graph:
                in_degree[node] += 0  # node depends on dep, so dep should come first
                # Actually: if node depends on dep, dep has an edge to node
                pass

    # Reverse: who depends on me?
    reverse_graph = defaultdict(set)
    for node, deps in graph.items():
        for dep in deps:
            if dep in graph:
                reverse_graph[dep].add(node)

    # In-degree: how many things does this node depend on (that are in graph)?
    in_degree = {}
    for node in graph:
        in_degree[node] = len(graph[node] & set(graph.keys()))

    # Kahn's algorithm
    queue = [n for n, d in in_degree.items() if d == 0]
    queue.sort()  # deterministic ordering
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for dependent in sorted(reverse_graph.get(node, [])):
            if dependent in in_degree:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    # Add any remaining nodes (cycles)
    remaining = [n for n in graph if n not in result]
    remaining.sort()
    result.extend(remaining)

    return result


def get_solve_order(program: str, unsolved_names: Optional[list[str]] = None) -> list[str]:
    """Get the optimal order to solve lemmas.

    Args:
        program: Dafny source code
        unsolved_names: If given, only include these names in the result.
                       Dependencies outside this list are assumed already solved.

    Returns:
        List of lemma names in the order they should be solved.
    """
    graph = build_dependency_graph(program)

    if unsolved_names is not None:
        # Filter graph to only include unsolved names,
        # but keep dependency edges between them
        unsolved_set = set(unsolved_names)
        filtered = {}
        for name in unsolved_names:
            if name in graph:
                filtered[name] = graph[name] & unsolved_set
            else:
                filtered[name] = set()
        graph = filtered

    # Filter to only lemmas
    done = sketcher.sketch_done(program)
    lemma_names = {item['name'] for item in done if item.get('type') == 'lemma'}
    lemma_graph = {k: v for k, v in graph.items() if k in lemma_names}

    return topological_order(lemma_graph)


def format_dependency_info(program: str, lemma_name: str) -> str:
    """Format dependency information for a specific lemma."""
    graph = build_dependency_graph(program)
    deps = graph.get(lemma_name, set())

    if not deps:
        return f"{lemma_name} has no dependencies on other declarations."

    done = sketcher.sketch_done(program)
    done_map = {item['name']: item for item in done}

    parts = [f"Dependencies of {lemma_name}:"]
    for dep_name in sorted(deps):
        item = done_map.get(dep_name)
        if item:
            kind = item.get('type', '?')
            has_body = item.get('status') == 'done'
            status = "has body" if has_body else "NO BODY (uninterpreted)"
            parts.append(f"  [{kind}] {dep_name} ({status})")
        else:
            parts.append(f"  [?] {dep_name}")

    return "\n".join(parts)
