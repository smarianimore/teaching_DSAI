"""Strategies for the (iterated) Prisoner's Dilemma.

A strategy is a small class with ONE method, ``move``, which looks at what
happened so far against a specific opponent and answers with ``COOPERATE``
or ``DEFECT``.  Strategies are stateless: all memory is passed in as
arguments, so a single instance can safely be shared by many agents.
"""
import random

COOPERATE = "C"
DEFECT = "D"

# Classic payoff matrix, indexed by (my_move, opponent_move) -> my payoff.
# It must satisfy T > R > P > S and 2R > T + S for a true Prisoner's Dilemma.
PAYOFFS = {
    (COOPERATE, COOPERATE): 3,  # R: reward for mutual cooperation
    (COOPERATE, DEFECT): 0,     # S: sucker's payoff
    (DEFECT, COOPERATE): 5,     # T: temptation to defect
    (DEFECT, DEFECT): 1,        # P: punishment for mutual defection
}


class Strategy:
    name = "base"

    def move(self, my_history, opp_history):
        """my_history / opp_history: lists of past moves vs THIS opponent."""
        raise NotImplementedError


class Random(Strategy):
    name = "random"

    def move(self, my_history, opp_history):
        return random.choice([COOPERATE, DEFECT])


class Hawk(Strategy):
    """Always defect."""
    name = "hawk"

    def move(self, my_history, opp_history):
        return DEFECT


class Dove(Strategy):
    """Always cooperate."""
    name = "dove"

    def move(self, my_history, opp_history):
        return COOPERATE


class TitForTat(Strategy):
    """Cooperate first, then copy the opponent's last move (Axelrod's winner)."""
    name = "tit-for-tat"

    def move(self, my_history, opp_history):
        return opp_history[-1] if opp_history else COOPERATE


class SuspiciousTitForTat(Strategy):
    """Like tit-for-tat, but starts by defecting."""
    name = "suspicious-tft"

    def move(self, my_history, opp_history):
        return opp_history[-1] if opp_history else DEFECT


class TitForTwoTats(Strategy):
    """Defect only after two consecutive opponent defections (more forgiving)."""
    name = "tit-for-two-tats"

    def move(self, my_history, opp_history):
        if opp_history[-2:] == [DEFECT, DEFECT]:
            return DEFECT
        return COOPERATE


class GenerousTitForTat(Strategy):
    """Tit-for-tat that forgives a defection 10% of the time."""
    name = "generous-tft"

    def move(self, my_history, opp_history):
        if opp_history and opp_history[-1] == DEFECT:
            return COOPERATE if random.random() < 0.1 else DEFECT
        return COOPERATE


class GrimTrigger(Strategy):
    """Cooperate until the opponent defects once, then defect forever."""
    name = "grim-trigger"

    def move(self, my_history, opp_history):
        return DEFECT if DEFECT in opp_history else COOPERATE


class Pavlov(Strategy):
    """Win-stay, lose-shift: repeat last move if it paid well (R or T), else switch."""
    name = "pavlov"

    def move(self, my_history, opp_history):
        if not my_history:
            return COOPERATE
        if PAYOFFS[(my_history[-1], opp_history[-1])] >= 3:
            return my_history[-1]
        return DEFECT if my_history[-1] == COOPERATE else COOPERATE


class Joss(Strategy):
    """Sneaky tit-for-tat: 10% of the time defects even when it should cooperate."""
    name = "joss"

    def move(self, my_history, opp_history):
        if opp_history and opp_history[-1] == DEFECT:
            return DEFECT
        return DEFECT if random.random() < 0.1 else COOPERATE


# Registry: name -> class.  The simulation and the command line use it, so
# adding a new strategy only requires writing a class and listing it here.
STRATEGIES = {cls.name: cls for cls in (
    Random, Hawk, Dove, TitForTat, SuspiciousTitForTat, TitForTwoTats,
    GenerousTitForTat, GrimTrigger, Pavlov, Joss,
)}
