"""Slide 5: observe, formulate a problem, plan, execute one action."""

# Each pair is (action, next state). All actions cost one step.
PROCESS = {
    "raw": [("cut", "cut part"), ("discard", "scrap")],
    "cut part": [("paint", "painted part")],
    "painted part": [("pack", "packed")],
    "scrap": [],
    "packed": [],
}


def update_state(old_state, percept):
    # Our sensor reports the complete state, so old_state is unnecessary here.
    return percept


def formulate_goal(state):
    return "packed"


def formulate_problem(state, goal):
    return {"start": state, "goal": goal, "actions": PROCESS}


def search(problem):
    # Breadth-first search: each entry holds a state and its action sequence.
    fringe = [(problem["start"], [])]
    while fringe:
        print("  FRINGE:", fringe)
        state, plan = fringe.pop(0)
        print("    SEARCH:", state, "| plan:", plan)
        if state == problem["goal"]:
            return plan
        for action, next_state in problem["actions"][state]:  # states "scrap" and "packed" have no actions, so this loop is skipped for them.
            fringe.append((next_state, plan + [action]))
    return None  # Different from []: [] means the goal is already satisfied.


def agent(percept, memory):
    memory["state"] = update_state(memory["state"], percept)
    if not memory["plan"]:
        goal = formulate_goal(memory["state"])
        problem = formulate_problem(memory["state"], goal)
        memory["plan"] = search(problem)
    if memory["plan"] is None:
        raise ValueError("No production plan exists")
    if not memory["plan"]:
        return "STOP"
    return memory["plan"].pop(0)  # Execute only the first remaining action.


def main():
    world = "raw"
    memory = {"state": None, "plan": []}  # Persists across calls to agent.
    while True:
        print("\nSENSOR:", world)
        action = agent(world, memory)
        if action == "STOP":
            print("GOAL: packed part ready for dispatch")
            break
        # The environment changes only here, after planning has finished.
        for possible_action, next_state in PROCESS[world]:
            if possible_action == action:
                print("EXECUTE:", world, "--", action, "-->", next_state)
                world = next_state
                break


if __name__ == "__main__":
    main()
