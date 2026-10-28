# Axelrod tournament on a 2D grid

Agents random-walk on a wrapping (torus) grid. When two agents land on the
same cell they play one round of the Prisoner's Dilemma; each agent remembers
the history with every opponent it has met, so strategies like tit-for-tat
work across repeated encounters. If 3+ agents share a cell they are paired
randomly (the odd one sits out).

Payoffs: both cooperate 3/3, both defect 1/1, defect vs cooperate 5/0.

## Requirements

Python 3.8+ with `tkinter` (bundled with most Python installs; on Debian/Ubuntu:
`sudo apt install python3-tk`). No other packages needed.

## Run

```bash
cd scripts/agents
python main.py                                   # 5 agents per strategy, 30x30 grid
python main.py --width 20 --height 20 \
    --agents hawk=15,tit-for-tat=15,random=10,pavlov=5 --seed 1
python main.py --headless 5000 --seed 1          # no GUI, prints average payoffs
```

Options: `--width`, `--height`, `--agents name=n,...`, `--seed`, `--cell`
(pixels), `--memory N`, `--delay` (ms/frame), `--steps-per-frame`, `--headless STEPS`.

The window shows the arena (one colour per strategy) and a dashboard with the
cumulative average payoff per game of each strategy over time (legend sorted
by current value). Buttons: Pause/Resume and Step.

## Memory

Each agent remembers its past rounds *per opponent*. `--memory N` limits this
to the last `N` rounds against each opponent (default: unlimited; `0` = no
memory, so every encounter looks like a first meeting). It only affects
strategies that read the history (marked *memory* below):

```bash
python main.py --memory 1      # only the last round is remembered
python main.py --memory 0      # nobody remembers anything
```

Effects: with `--memory 0` retaliating strategies are exploited by hawks;
`grim-trigger` forgets grudges once the defection leaves its window.

## Strategies

| Name | Uses memory | Behaviour |
|------|-------------|-----------|
| `random` | no | Cooperates or defects with 50% probability. |
| `hawk` | no | Always defects. |
| `dove` | no | Always cooperates (easy prey for hawks). |
| `tit-for-tat` | yes | Cooperates first, then copies the opponent's last move. Winner of Axelrod's original tournament. |
| `suspicious-tft` | yes | Like tit-for-tat but defects on the first meeting. |
| `tit-for-two-tats` | yes | Defects only after two consecutive opponent defections; forgiving. |
| `generous-tft` | yes | Tit-for-tat, but forgives a defection with 10% probability, breaking retaliation spirals. |
| `grim-trigger` | yes | Cooperates until the opponent defects once, then defects forever (within its memory). |
| `pavlov` | yes | Win-stay, lose-shift: repeats its last move if it earned 3 or 5, otherwise switches. |
| `joss` | yes | Tit-for-tat that sneakily defects 10% of the time when it would cooperate. |

To add one, write a class in `strategies.py` with a `name` and a
`move(my_history, opp_history)` method, and add it to `STRATEGIES`. The
histories passed in are already truncated to the configured memory.

## Files

- `strategies.py` - payoff matrix and strategies
- `simulation.py` - grid, agents, encounters (no GUI)
- `gui.py` - tkinter arena + dashboard
- `main.py` - command-line entry point
