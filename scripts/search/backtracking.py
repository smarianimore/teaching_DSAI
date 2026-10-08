"""
Standard BackTracking (SBT) for Constraint Satisfaction Problems (slides 15-17).

Generic: works with any CSP object that has
    .variables   list of variable names
    .domains     dict  variable -> list of values
    .constraints list of objects with .scope (tuple of variables)
                 and .predicate(*values) -> bool

Differences from G&T: the assignment is tested for consistency *as it is built*,
so a partial assignment that already violates a constraint is abandoned
immediately, together with every complete assignment below it in the search tree.
"""
import time

from scripts.search.map_coloring import map_coloring_csp


class Stats:
    """Counters for comparing algorithms."""
    def __init__(self):
        self.assignments = 0   # variable=value assignments attempted
        self.backtracks = 0    # times a variable ran out of consistent values


# --- the three pluggable choices named on slides 17-18 -----------------------
def select_unassigned_variable(csp, assignment):
    """Static order: first variable not yet assigned."""
    for var in csp.variables:
        if var not in assignment:
            return var


def order_domain_values(csp, var, assignment):
    """Static order: values as listed in the domain."""
    return csp.domains[var]


def is_consistent_with(csp, var, value, assignment):
    """Would var=value violate any constraint whose variables are all assigned?
    Constraints that still involve free variables cannot be violated yet."""
    assignment[var] = value
    try:
        for c in csp.constraints:
            if var in c.scope and all(v in assignment for v in c.scope):
                if not c.predicate(*(assignment[v] for v in c.scope)):
                    return False
        return True
    finally:
        del assignment[var]


# --- slide 17 ----------------------------------------------------------------
def recursive_backtracking(assignment, csp, stats):
    """Yield every solution reachable from this partial assignment."""
    if len(assignment) == len(csp.variables):              # complete
        yield dict(assignment)
        return
    var = select_unassigned_variable(csp, assignment)
    found_any = False
    for value in order_domain_values(csp, var, assignment):
        if is_consistent_with(csp, var, value, assignment):
            stats.assignments += 1
            assignment[var] = value                        # add {var = value}
            for solution in recursive_backtracking(assignment, csp, stats):
                found_any = True
                yield solution
            del assignment[var]                            # remove {var = value}
    if not found_any:
        stats.backtracks += 1


def backtracking_search(csp, find_all=False):
    """Return (solutions, stats). With find_all=False, stop at the first solution
    (the function on slide 17 returns "a solution, or failure")."""
    stats = Stats()
    solutions = []
    for solution in recursive_backtracking({}, csp, stats):
        solutions.append(solution)
        if not find_all:
            break
    return solutions, stats


def main():
    csp = map_coloring_csp()
    print("Variables:  ", csp.variables)
    print("Domain:     ", csp.domains["WA"])
    print("Constraints:", len(csp.constraints), "(neighboring zones must differ)\n")

    # --- first solution -------------------------------------------------
    start = time.time()
    solutions, stats = backtracking_search(csp, find_all=False)
    checkpoint = time.time()
    first_t = checkpoint - start
    print("Standard backtracking, first solution:")
    print("  ", solutions[0])
    print(f"   {stats.assignments} assignments made, {stats.backtracks} backtracks (in {first_t} seconds)")

    # --- all solutions --------------------------------------------------
    solutions, stats = backtracking_search(csp, find_all=True)
    end = time.time() - checkpoint
    print("All solutions:")
    print(f"   backtracking:      {len(solutions)} solutions, "
          f"{stats.assignments} assignments made, {stats.backtracks} backtracks (in {end} seconds)")


if __name__ == "__main__":
    main()
