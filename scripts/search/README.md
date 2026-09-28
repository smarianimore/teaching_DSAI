# Search algorithms in production and logistics

Beginner Python examples accompanying **AI2a - Search.pptx**. Slide numbers refer to the numbered slides in the supplied deck. These are small, deterministic teaching models; no packages or GUI are required.

## Run the examples

Install Python 3 if needed, unzip this folder, and open a terminal in it. Run:

```sh
python 01_production_agent.py
python 02_vacuum_tree.py
python 03_four_strategies.py
python 04_delivery_greedy_astar.py
```

On systems where the command is `python3`, use that instead. Each script is independent. There are no inputs to enter and no dependencies to install. 

| Slides | Script | Scenario and what to watch |
|---|---|---|
| 5–6 | `01_production_agent.py` | A production agent observes a part, makes a plan, then executes one action at a time. |
| 9, 12, 15 | `02_vacuum_tree.py` | A cleaning robot searches over the slide's eight states. Node IDs expose repeated states in the tree. |
| 20–28 | `03_four_strategies.py` | BFS, DFS, DLS and IDS choose between manual and automated production routes. |
| 32 | `04_delivery_greedy_astar.py greedy` | A delivery vehicle chooses the waiting node with smallest estimated remaining distance. |
| 35–36 | `04_delivery_greedy_astar.py astar` | The same delivery includes distance already travelled in its score. |

## Python you need

- A list, such as `["cut", "paint"]`, stores items in order.
- A tuple, such as `("A", 1, 1)`, groups values describing a state.
- A dictionary maps keys to values: `H["Arad"]` is `366`.
- `for` visits items; `while` repeats while its condition is true.
- `def` defines a function; `return` gives its result to the caller.
- `path + [child]` creates a new path without changing the old one.
- `pop(0)` removes the first item; `pop()` removes the last item.
- `path[-1]` is the last state in a path. `state[1:]` selects everything after the first element.
- `None` means no result. An empty list `[]` can mean a valid plan requiring no actions.
- The final `if __name__ == "__main__":` block runs the demo when you run that file.

## 1. A problem-solving production agent

Slide 5 is the simple problem-solving agent loop. This example instantiates that loop; it does not implement the historical GPS means–ends analysis system.

The part can follow `raw -> cut part -> painted part -> packed`. Discarding the raw part leads to a dead end. The goal is to reach `packed`; each action costs 1.

| Slide function | Concrete function |
|---|---|
| Update-state | The sensor reports the complete current part state. |
| Formulate-goal | Set the target to `packed`. |
| Formulate-problem | Package the start, goal and allowed production actions. |
| Search | Explore action sequences with BFS and return a plan. |
| First / Rest | Remove and return the next action using `pop(0)`. |

`memory` preserves the agent's state and remaining plan between observations. Planning only explores descriptions: it does not operate machinery. The environment applies an action after the agent returns it.

The console distinguishes SEARCH from EXECUTE. The plan is `cut, paint, pack`. This toy assumes a fully observable, deterministic process that behaves as predicted; unexpected failures would require checking and replanning.

**Try:** remove the `paint` transition. Search returns `None` and the agent reports that no production plan exists.

## 2. Vacuum cleaner: states versus tree nodes

Interpret rooms A and B as two factory work zones. A state is `(location, dirt_A, dirt_B)`, with `1` for dirty and `0` for clean. The eight state labels exactly follow slide 9. Both `s1` and `s5` satisfy the goal, because the final robot location does not matter.

The table on slide 9 gives one example action per state. To make a branching search problem, the code also permits movement while the current zone is dirty. It omits actions that leave the state unchanged. All eight listed transitions remain legal.

A **state** describes the world. A **tree node** describes one way of reaching a state. For example, moving right and then left returns to the same state but creates a different tree node. The console shows the parent ID, action, child ID and resulting state for each generated edge. The node's depth is also its path cost here because every action costs 1.

This is genuine tree search: there is no global visited-state filter. BFS stops after finding the shallowest goal, even though movement can produce arbitrarily long paths. On a cyclic problem without a reachable goal, this version would not terminate.

The final replay is:

```text
0: START  | [A: dirty] [B: dirty] robot=A
1: Aspire | [A: clean] [B: dirty] robot=A
2: Go right | [A: clean] [B: dirty] robot=B
3: Aspire | [A: clean] [B: clean] robot=B
Total cost: 3
```

**Try:** start at `("B", 1, 0)` and predict the solution before running it.

## 3. Four uninformed strategies

The manual production route takes four operations. The automated route takes two. Each operation costs 1; both dispatch states are goals. This is a finite, acyclic tree, and the manual branch is listed first.

| Strategy | Selection rule | Result in this example |
|---|---|---|
| BFS | Oldest waiting path first (FIFO) | Automated route, cost 2 |
| DFS | Newest waiting path first (LIFO) | Manual route, cost 4 |
| DLS, limit 1 | DFS, stopping at depth 1 | Cutoff |
| DLS, limit 2 | DFS, stopping at depth 2 | Automated route, cost 2 |
| IDS | Repeat DLS at limits 0, 1, 2, ... | Automated route, cost 2 |

Indentation shows the depth of each selected path. It is an exploration trace, not a drawing of the entire tree: BFS naturally jumps between indentation levels.

DLS distinguishes **cutoff** (the limit prevented further exploration) from **failure** (`None`, the tree was exhausted). IDS increases the limit only after cutoff. A goal at exactly the limit is accepted.

BFS and IDS find a solution with the fewest actions. They also find minimum cost when every action has the same positive cost, as here. With unequal action costs, the fewest actions need not give the cheapest route. DFS terminates in this finite example; unrestricted DFS can follow an infinite branch in a cyclic or infinite search tree.

**Try:** reverse the two children of `Raw`. DFS now finds the automated route first. This shows why its first answer depends on action order.

## 4. Greedy best-first and A*: the slides' delivery route

A parcel starts in Arad and must reach Bucharest. The script transcribes the road graph and straight-line estimates from the slides. Distances and costs are in kilometres. A node stores `(path, cost)`; cost is `g`.

- `g(n)`: distance already travelled along this particular path.
- `h(n)`: estimated distance remaining, from the slide's heuristic table.
- Greedy's score: `h(n)`.
- A*'s score: `g(n) + h(n)`.

The script uses one search loop with a different scoring function. Sorting the fringe puts the lowest score first. Selection considers **all waiting nodes**, not just the current city's neighbors. This is why A* can return to Fagaras after expanding Rimnicu Vilcea.

| Selected city | Greedy score h | A* score g+h |
|---|---:|---:|
| Arad | 366 | 0 + 366 = 366 |
| Sibiu | 253 | 140 + 253 = 393 |
| Greedy next: Fagaras | 176 | — |
| A* next: Rimnicu Vilcea | — | 220 + 193 = 413 |

The full selection sequences are:

```text
Greedy: Arad -> Sibiu -> Fagaras -> Bucharest
A*:     Arad -> Sibiu -> Rimnicu Vilcea -> Fagaras -> Pitesti -> Bucharest
```

These are search selection sequences, not necessarily travel routes. A*'s final travel route does not include Fagaras.

| Method | Final delivery route | Distance |
|---|---|---:|
| Greedy | Arad → Sibiu → Fagaras → Bucharest | 450 km |
| A* | Arad → Sibiu → Rimnicu Vilcea → Pitesti → Bucharest | 418 km |

After expanding Rimnicu Vilcea, Fagaras has score 415 and Pitesti has 417. A* therefore selects Fagaras first. That generates a Bucharest node costing 450, but A* does **not** stop at generation: Pitesti still has the better score. Expanding Pitesti generates another Bucharest node costing 418, which is selected next. This reproduces the point of slide 36.

Repeated cities and both Bucharest nodes are deliberately retained to match the slide trees. This tree-search implementation is for the fixed, reachable example; greedy tree search can loop on other cyclic maps. A production route planner should handle repeated states and retain cheaper paths, reopening states when needed.

Straight-line distance is a lower bound on road distance, so the heuristic is admissible. With positive road costs, finite branching and this admissible heuristic (zero at the goal), A* tree search finds an optimal route. These assumptions matter; an arbitrary heuristic does not give the same guarantee.

**Try:** set all values in `H` to zero. A* becomes uniform-cost search, selecting the path with smallest distance so far. Or change a heuristic to an overestimate and watch the selection order change; optimality is then no longer guaranteed.

## Deliberate simplifications

These scripts use ordinary lists and print statements so the search decisions remain visible. Sorting a list and removing its first element are inefficient for large problems. Paths are copied for readability, so these programs are not demonstrations of the tightest memory bounds in the slides. There are no priority queues, classes, decorators or third-party packages. All results in the saved runs were generated by executing the included code.
