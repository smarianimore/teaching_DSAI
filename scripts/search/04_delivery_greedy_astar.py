"""Slides 38, 40, 41: deliver a parcel from Arad to Bucharest.
Run with 'greedy', 'astar', or no argument to compare both.
"""
import sys

# Undirected roads, in km: the Romania map in the slides.
ROADS = [
    ("Arad", "Sibiu", 140), ("Arad", "Timisoara", 118),
    ("Arad", "Zerind", 75), ("Zerind", "Oradea", 71),
    ("Sibiu", "Fagaras", 99), ("Sibiu", "Oradea", 151),
    ("Sibiu", "Rimnicu Vilcea", 80), ("Timisoara", "Lugoj", 111),
    ("Lugoj", "Mehadia", 70), ("Mehadia", "Drobeta", 75),
    ("Drobeta", "Craiova", 120), ("Rimnicu Vilcea", "Craiova", 146),
    ("Rimnicu Vilcea", "Pitesti", 97), ("Craiova", "Pitesti", 138),
    ("Fagaras", "Bucharest", 211), ("Pitesti", "Bucharest", 101),
    ("Bucharest", "Giurgiu", 90), ("Bucharest", "Urziceni", 85),
    ("Urziceni", "Hirsova", 98), ("Hirsova", "Eforie", 86),
    ("Urziceni", "Vaslui", 142), ("Vaslui", "Iasi", 92),
    ("Iasi", "Neamt", 87),
]
# h: straight-line distance to Bucharest, copied from slide 32.
H = {
    "Arad": 366, "Bucharest": 0, "Craiova": 160, "Drobeta": 242,
    "Eforie": 161, "Fagaras": 176, "Giurgiu": 77, "Hirsova": 151,
    "Iasi": 226, "Lugoj": 244, "Mehadia": 241, "Neamt": 234,
    "Oradea": 380, "Pitesti": 100, "Rimnicu Vilcea": 193,
    "Sibiu": 253, "Timisoara": 329, "Urziceni": 80, "Vaslui": 199,
    "Zerind": 374,
}
GRAPH = {}
for a, b, distance in ROADS:
    GRAPH.setdefault(a, []).append((b, distance))
    GRAPH.setdefault(b, []).append((a, distance))


def search(strategy):
    def score(node):
        path, cost = node
        h = H[path[-1]]
        if strategy == "greedy":
            return h
        return cost + h  # A*: distance so far + estimated distance remaining.

    # Each node holds a complete path and g (distance already travelled).
    # Tree search deliberately retains repeated cities, as in the slide trees.
    fringe = [(["Arad"], 0)]
    while fringe:
        fringe.sort(key=score)  # Lowest score first; insertion order breaks ties.
        print("\nFRINGE (best first): city [g, h, score]")
        for path, cost in fringe:
            print(" ", path[-1], [cost, H[path[-1]], score((path, cost))])
        path, cost = fringe.pop(0)
        city = path[-1]
        print("SELECT:", " -> ".join(path))
        if city == "Bucharest":
            # Stop when the goal is selected, not when it is first generated.
            print("DELIVERED:", cost, "km")
            return path, cost
        for neighbor, distance in GRAPH[city]:
            fringe.append((path + [neighbor], cost + distance))
    return None


def main():
    strategies = sys.argv[1:] or ["greedy", "astar"]
    for strategy in strategies:
        if strategy not in ["greedy", "astar"]:
            raise SystemExit("Use: python 04_delivery_greedy_astar.py [greedy|astar]")
        print("\n===", strategy.upper(), "===")
        search(strategy)


if __name__ == "__main__":
    main()
