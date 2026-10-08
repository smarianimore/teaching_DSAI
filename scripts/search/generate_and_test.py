"""
Generate & Test (G&T) for Constraint Satisfaction Problems.

Slide 14: "All partial assignments are generated until all complete
assignments are found. Each complete assignment is tested for consistency."

Applied to the Australia map-coloring CSP (slides 9-11).
"""
import time
import map_coloring


# --------------------------------------------------------------------------
# Generate: extend partial assignments one variable at a time. Nothing is
# checked on the way down; only complete assignments are yielded.
# --------------------------------------------------------------------------
def generate(csp, assignment=None):
    assignment = {} if assignment is None else assignment
    if len(assignment) == len(csp.variables):          # complete
        yield dict(assignment)  # compared with 'return', 'yield' returns the value without exiting the function, so that the next call to generate() will continue from here
        return
    var = csp.variables[len(assignment)]               # next free variable
    for value in csp.domains[var]:
        assignment[var] = value
        yield from generate(csp, assignment)  # recursively extend the assignment
        del assignment[var]  # backtrack: remove the last variable assignment so that the next value can be tried


# --------------------------------------------------------------------------
# Test: a complete assignment is consistent iff every constraint is satisfied.
# --------------------------------------------------------------------------
def is_consistent(csp, assignment):
    return all(c.predicate(*(assignment[v] for v in c.scope))  # operator '*' unpacks the list of values it is applied to: e.g., if c.scope = (x, y) and assignment = {x: 1, y: 2}, then *(assignment[v] for v in c.scope) = (1, 2) as two separate arguments
               for c in csp.constraints)


def generate_and_test(csp, find_all=True):
    """Return (solutions, number_of_assignments_tested)."""
    solutions, tested = [], 0
    for assignment in generate(csp):
        tested += 1
        if is_consistent(csp, assignment):
            solutions.append(assignment)
            if not find_all:
                break
    return solutions, tested


if __name__ == "__main__":
    csp = map_coloring.map_coloring_csp()
    total = 1
    for v in csp.variables:
        total *= len(csp.domains[v])
    print(f"Complete assignments in the search space: |D|^|V| = 3^7 = {total}")

    start = time.time()
    first, tested = generate_and_test(csp, find_all=False)
    first_t = time.time() - start
    print(f"\nFirst solution (after testing {tested} assignments in {first_t} seconds):")
    print("  ", first[0])

    solutions, tested = generate_and_test(csp, find_all=True)
    finish = time.time() - start
    print(f"\nAll solutions: {len(solutions)} found, {tested} assignments tested (in {finish} seconds):")
    for s in solutions:
        print("  ", s)
