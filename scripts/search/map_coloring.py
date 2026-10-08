"""
Australia map coloring (slides 9-11) solved with Standard BackTracking,
and compared with Generate & Test.

The problem definition lives in generate_and_test.py (map_coloring_csp);
the algorithm lives in backtracking.py. This script only wires them together.
"""

from collections import namedtuple  # like a tuple, but each element can be referred to by name (like in a dictionary)

# A CSP is a triple <V, D, C> (slide 4).
CSP = namedtuple("CSP", ["variables", "domains", "constraints"])

# A constraint is a scope (the variables it involves) plus a predicate that
# says whether a full set of values for that scope is allowed.
Constraint = namedtuple("Constraint", ["scope", "predicate"])


def different(x, y):
    """Binary constraint: x != y."""
    return Constraint((x, y), lambda a, b: a != b)


def map_coloring_csp():
    variables = ["WA", "NT", "Q", "NSW", "V", "SA", "T"]
    domains = {v: ["R", "G", "B"] for v in variables}  # one domain, shared
    neighbors = [
        ("WA", "NT"), ("WA", "SA"),
        ("NT", "SA"), ("NT", "Q"),
        ("SA", "Q"), ("SA", "NSW"), ("SA", "V"),
        ("Q", "NSW"), ("NSW", "V"),
    ]  # Tasmania (T) has no neighbors
    return CSP(variables, domains, [different(x, y) for x, y in neighbors])


