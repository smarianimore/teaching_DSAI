"""
Maintaining Arc Consistency (MAC), slides 27-30, applied to the Australia
map-coloring CSP (slides 9-11).

Contents
    ac3()        the arc consistency algorithm AC-3 (slide 29)
    mac_search() Standard BackTracking that enforces arc consistency before
                 every assignment (slide 30)
    main()       applies both to the map-coloring problem, which is defined in
                 generate_and_test.py (map_coloring_csp), and compares MAC with
                 standard backtracking (backtracking.py)

Binary constraints only: arc consistency is defined on binary CSPs (slide 28).
"""
import time
from collections import deque

if __package__:
    from .map_coloring import map_coloring_csp
else:
    from map_coloring import map_coloring_csp


class Stats:
    def __init__(self):
        self.assignments = 0       # variable=value assignments attempted
        self.backtracks = 0        # variables left with no value that works
        self.values_pruned = 0     # values removed from domains by AC-3


# --------------------------------------------------------------------------
# Constraint graph: directed arcs (X, Y) -> predicates p(x_value, y_value)
# --------------------------------------------------------------------------
def build_arcs(csp):
    arcs = {}
    for c in csp.constraints:
        if len(c.scope) != 2:
            raise ValueError("AC-3 needs binary constraints; got scope %s" % (c.scope,))
        a, b = c.scope
        arcs.setdefault((a, b), []).append(c.predicate)
        arcs.setdefault((b, a), []).append(lambda y, x, p=c.predicate: p(x, y))  # we need to swap arguments on the other direction in case the constraint is not symmetric (e.g. < or >)
    neighbors = {v: [] for v in csp.variables}
    for (a, b) in arcs:
        neighbors[a].append(b)
    return arcs, neighbors


# --------------------------------------------------------------------------
# AC-3 (slide 29)
# --------------------------------------------------------------------------
def rm_inconsistent_values(arcs, domains, xi, xj, stats=None):
    """Remove every x in dom(Xi) that has no supporting y in dom(Xj).
    Returns True iff some value was removed."""
    removed = False
    for x in list(domains[xi]):
        if not any(all(p(x, y) for p in arcs[(xi, xj)]) for y in domains[xj]):  # if there is no y that abides to all constraints with x, remove x
            domains[xi].remove(x)
            removed = True
            if stats:
                stats.values_pruned += 1
    return removed


def ac3(arcs, neighbors, domains, queue=None, stats=None):
    """Reduce `domains` in place until every arc is consistent.
    `queue` defaults to all arcs. Returns False as soon as a domain becomes
    empty (the CSP has no solution under these domains), True otherwise."""
    queue = deque(arcs if queue is None else queue)
    while queue:
        xi, xj = queue.popleft()
        if rm_inconsistent_values(arcs, domains, xi, xj, stats):
            if not domains[xi]:
                return False
            for xk in neighbors[xi]:        # Xi changed: re-check arcs pointing at it
                if xk != xj:
                    queue.append((xk, xi))
    return True


# --------------------------------------------------------------------------
# MAC (slide 30): backtracking + arc consistency maintained at every node
# --------------------------------------------------------------------------
def select_unassigned_variable(csp, assignment):
    return next(v for v in csp.variables if v not in assignment)   # static order


def _mac(csp, arcs, neighbors, assignment, domains, stats):
    if len(assignment) == len(csp.variables):
        yield dict(assignment)
        return  # in this case we finished, so we also want to return besides yielding
    var = select_unassigned_variable(csp, assignment)
    found_any = False
    for value in domains[var]:
        stats.assignments += 1
        trial = {v: list(d) for v, d in domains.items()}
        trial[var] = [value]                                   # the assignment...
        queue = [(xk, var) for xk in neighbors[var]]           # ...is a new constraint to propagate
        if ac3(arcs, neighbors, trial, queue, stats):
            assignment[var] = value
            for solution in _mac(csp, arcs, neighbors, assignment, trial, stats):
                found_any = True
                yield solution
            del assignment[var]
    if not found_any:
        stats.backtracks += 1


def mac_search(csp, find_all=False):
    """Return (solutions, stats)."""
    arcs, neighbors = build_arcs(csp)
    stats = Stats()
    domains = {v: list(csp.domains[v]) for v in csp.variables}
    if not ac3(arcs, neighbors, domains, stats=stats):         # propagate before the search starts
        return [], stats
    solutions = []
    for s in _mac(csp, arcs, neighbors, {}, domains, stats):
        solutions.append(s)
        if not find_all:
            break
    return solutions, stats


# --------------------------------------------------------------------------
# Apply to map coloring
# --------------------------------------------------------------------------
def main():
    csp = map_coloring_csp()
    arcs, neighbors = build_arcs(csp)

    # 1. AC-3 on the initial problem: nothing can be pruned yet, because with
    #    3 colors every color of X is supported by some color of each neighbor.
    domains = {v: list(csp.domains[v]) for v in csp.variables}
    st = Stats()
    ok = ac3(arcs, neighbors, domains, stats=st)
    print(f"AC-3 on the initial problem: consistent={ok}, values pruned={st.values_pruned}\n")

    # 2. The example on slide 26: WA=R and Q=G. Forward checking leaves
    #    NT={B}, SA={B} and does not notice the clash. AC-3 does.
    domains = {v: list(csp.domains[v]) for v in csp.variables}
    domains["WA"], domains["Q"] = ["R"], ["G"]
    st = Stats()
    ok = ac3(arcs, neighbors, domains, stats=st)
    print("Slide 26 example (WA=R, Q=G), AC-3 propagation:")
    print(f"   consistent={ok}, values pruned={st.values_pruned}")
    print(f"   domains: NT={domains['NT']}  SA={domains['SA']}  "
          "-> NT is forced to B, which wipes out SA: failure detected\n")

    # 3. MAC search
    start = time.time()
    sols, mac_stats = mac_search(csp, find_all=False)
    checkpoint = time.time()
    first_t = checkpoint - start
    print("First solution:")
    print("  ", sols[0])
    print(f"   MAC: {mac_stats.assignments} assignments, {mac_stats.backtracks} backtracks (in {first_t} seconds)")

    sols, mac_stats = mac_search(csp, find_all=True)
    end = time.time() - checkpoint
    print("All solutions:")
    print(f"   MAC: {len(sols)} solutions, {mac_stats.assignments} assignments, "
          f"{mac_stats.backtracks} backtracks, {mac_stats.values_pruned} values pruned (in {end} seconds)")


if __name__ == "__main__":
    main()
