"""Slides 9 and 12: breadth-first tree search for a factory cleaning robot."""

# State = (robot location, dirt in A, dirt in B); 1 = dirty, 0 = clean.
STATES = [
    ("A", 0, 0), ("A", 0, 1), ("A", 1, 0), ("A", 1, 1),
    ("B", 0, 0), ("B", 0, 1), ("B", 1, 0), ("B", 1, 1),
]


def label(state):
    return "s" + str(STATES.index(state) + 1)


def picture(state):
    location, dirt_a, dirt_b = state
    a = "dirty" if dirt_a else "clean"
    b = "dirty" if dirt_b else "clean"
    return "[A: " + a + "] [B: " + b + "] robot=" + location


def successors(state):
    location, dirt_a, dirt_b = state
    # Slide 9 shows example transitions, not every legal action.
    # Include movement even when the current room is dirty.
    # Omit actions that do nothing (e.g. vacuuming a clean room).
    if location == "A":
        result = [("Go right", ("B", dirt_a, dirt_b))]
        if dirt_a:
            result.insert(0, ("Aspire", ("A", 0, dirt_b)))
    else:
        result = [("Go left", ("A", dirt_a, dirt_b))]
        if dirt_b:
            result.insert(0, ("Aspire", ("B", dirt_a, 0)))
    return result


def tree_search(start):
    # A node records its parent, action, depth and accumulated cost (slide 15).
    nodes = [{"state": start, "parent": None, "action": "START", "depth": 0}]
    fringe = [0]  # Node IDs waiting to be selected.
    while fringe:
        node_id = fringe.pop(0)
        node = nodes[node_id]
        state = node["state"]
        print("SELECT n" + str(node_id), label(state),
              "depth/cost=" + str(node["depth"]))
        if state[1:] == (0, 0):  # Either robot location is acceptable.
            path = []
            while node_id is not None:
                path.insert(0, nodes[node_id])
                node_id = nodes[node_id]["parent"]
            return path
        for action, next_state in successors(state):
            child_id = len(nodes)
            nodes.append({"state": next_state, "parent": node_id,
                          "action": action, "depth": node["depth"] + 1})
            fringe.append(child_id)
            print("  n" + str(node_id), "--" + action + "-->",
                  "n" + str(child_id), label(next_state), picture(next_state))
    return None


def main():
    print("Factory cleaning: start in A, both zones dirty. Each action costs 1.")
    path = tree_search(("A", 1, 1))
    print("\nREPLAY OF THE SOLUTION")
    for node in path:
        print(str(node["depth"]) + ":", node["action"], "|", picture(node["state"]))
    print("Total cost:", path[-1]["depth"])


if __name__ == "__main__":
    main()
