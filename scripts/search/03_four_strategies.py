"""Slide 20: four search strategies on the same production decision tree."""

# A slow manual route is listed before a short automated route.
# Each edge costs one operation. Either dispatch state is a goal.
NEXT = {
    "Raw": ["Manual cut", "Auto cell"],
    "Manual cut": ["Polish"],
    "Polish": ["Manual cut", "Paint"],
    "Paint": ["Dispatch manual"],
    "Auto cell": ["Dispatch auto"],
    "Dispatch manual": [],
    "Dispatch auto": [],
}
GOALS = ["Dispatch manual", "Dispatch auto"]


def show(path):
    print("  " * (len(path) - 1) + path[-1], "(depth", str(len(path) - 1) + ")")


def bfs(start):
    fringe = [[start]]
    while fringe:
        print("Fringe:", [p[-1] for p in fringe])
        path = fringe.pop(0)  # FIFO: oldest path first.
        show(path)
        if path[-1] in GOALS:
            return path
        for child in NEXT[path[-1]]:
            fringe.append(path + [child])
    return None


def dfs(start):
    fringe = [[start]]
    while fringe:
        print("Fringe:", [p[-1] for p in fringe])
        path = fringe.pop()  # LIFO: newest path first.
        show(path)
        if path[-1] in GOALS:
            return path
        # Reverse insertion so the first listed child is explored first.
        for child in reversed(NEXT[path[-1]]):
            fringe.append(path + [child])
    return None


def dls(path, limit):
    show(path)
    state = path[-1]
    if state in GOALS:  # Test the goal before checking the depth limit.
        return path
    if len(path) - 1 == limit:
        return "cutoff" if NEXT[state] else None
    had_cutoff = False
    for child in NEXT[state]:
        result = dls(path + [child], limit)
        if result == "cutoff":
            had_cutoff = True
        elif result is not None:
            return result
    return "cutoff" if had_cutoff else None


def ids(start):
    limit = 0
    while True:
        print("\nDepth limit =", limit)
        result = dls([start], limit)
        if result != "cutoff":
            return result  # Solution, or exhausted finite tree (None).
        limit += 1


def report(result):
    if result == "cutoff":
        print("RESULT: cutoff; a deeper solution may exist")
    elif result is None:
        print("RESULT: no solution")
    else:
        print("RESULT:", " -> ".join(result), "| cost:", len(result) - 1)


def main():
    print("Production choices (each edge costs 1):")
    for state, children in NEXT.items():
        if children:
            print(" ", state, "->", ", ".join(children))
    print("\nBFS: select the shallowest waiting path")
    report(bfs("Raw"))
    #print("\nDFS: follow the first branch deeply")
    #report(dfs("Raw"))
    print("\nDLS: DFS with limit 1")
    report(dls(["Raw"], 1))
    print("\nDLS: DFS with limit 2")
    report(dls(["Raw"], 2))
    print("\nIDS: repeat DLS with increasing limits")
    report(ids("Raw"))


if __name__ == "__main__":
    main()
