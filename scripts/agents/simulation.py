"""Headless simulation: agents random-walk on a torus grid and play the
Prisoner's Dilemma whenever they share a cell.  No GUI code lives here, so
it can be tested or run without a display.
"""
import random
from collections import defaultdict

from strategies import PAYOFFS, STRATEGIES


class Agent:
    def __init__(self, ident, strategy, x, y):
        self.ident = ident
        self.strategy = strategy
        self.x, self.y = x, y
        self.total_payoff = 0
        self.games = 0
        # Per-opponent memory: opponent id -> (my moves, their moves).
        # defaultdict creates the pair of empty lists on first access.
        self.memory = defaultdict(lambda: ([], []))


class Simulation:
    def __init__(self, width=30, height=30, counts=None, seed=None, memory=None):
        """counts: dict strategy name -> number of agents, e.g. {"hawk": 10}.

        memory: how many past rounds against each opponent an agent remembers
        (None = unlimited, 0 = no memory at all).
        """
        if memory is not None and memory < 0:
            raise ValueError("memory must be >= 0 or None")
        self.memory = memory
        if counts is None:
            counts = {name: 5 for name in STRATEGIES}
        unknown = set(counts) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"unknown strategies: {sorted(unknown)}")
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        random.seed(seed)  # strategies use the global `random` module
        self.agents = []
        for name, n in counts.items():
            strategy = STRATEGIES[name]()  # stateless, so one instance per group is fine
            for _ in range(n):
                self.agents.append(Agent(len(self.agents), strategy,
                                         self.rng.randrange(width),
                                         self.rng.randrange(height)))
        self.step_count = 0
        self.names = [n for n, c in counts.items() if c > 0]
        # History for the dashboard: per strategy, one value per step.
        self.history = {n: [] for n in self.names}

    def _play(self, a, b):
        """One round of the dilemma between agents a and b (simultaneous moves)."""
        mine_a, theirs_a = a.memory[b.ident]
        mine_b, theirs_b = b.memory[a.ident]
        move_a = a.strategy.move(mine_a, theirs_a)
        move_b = b.strategy.move(mine_b, theirs_b)
        pay_a = PAYOFFS[(move_a, move_b)]
        pay_b = PAYOFFS[(move_b, move_a)]
        mine_a.append(move_a); theirs_a.append(move_b)
        mine_b.append(move_b); theirs_b.append(move_a)
        if self.memory is not None:
            # Forget the oldest rounds: keep only the last `memory` ones.
            # `del lst[:n - memory]` drops the first n - memory items (this
            # form also works for memory=0, unlike lst[:-memory]).
            for lst in (mine_a, theirs_a, mine_b, theirs_b):
                del lst[:max(0, len(lst) - self.memory)]
        for agent, pay in ((a, pay_a), (b, pay_b)):
            agent.total_payoff += pay
            agent.games += 1

    def step(self):
        # 1. Everybody moves one cell (or stays); the grid wraps around (torus).
        for ag in self.agents:
            ag.x = (ag.x + self.rng.choice((-1, 0, 1))) % self.width
            ag.y = (ag.y + self.rng.choice((-1, 0, 1))) % self.height
        # 2. Group agents by cell; cells with 2+ agents trigger encounters.
        cells = defaultdict(list)
        for ag in self.agents:
            cells[(ag.x, ag.y)].append(ag)
        for group in cells.values():
            self.rng.shuffle(group)
            # Pair up agents two by two; an odd one out sits the round out.
            for i in range(0, len(group) - 1, 2):
                self._play(group[i], group[i + 1])
        self.step_count += 1
        # 3. Record the cumulative average payoff per game for each strategy.
        for name in self.names:
            members = [a for a in self.agents if a.strategy.name == name]
            games = sum(a.games for a in members)
            total = sum(a.total_payoff for a in members)
            self.history[name].append(total / games if games else None)
