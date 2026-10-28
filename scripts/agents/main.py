"""Entry point: parse command-line options, build the simulation, start the GUI."""
import argparse

from strategies import STRATEGIES
from simulation import Simulation


def parse_counts(text):
    """Turn 'hawk=10,tit-for-tat=5' into {'hawk': 10, 'tit-for-tat': 5}."""
    counts = {}
    for item in text.split(","):
        name, _, n = item.partition("=")
        if name.strip() not in STRATEGIES or not n.strip().isdigit():
            raise argparse.ArgumentTypeError(
                f"bad item '{item}'; use name=number with name in {sorted(STRATEGIES)}")
        counts[name.strip()] = int(n)
    return counts


def main():
    p = argparse.ArgumentParser(description="Axelrod tournament on a 2D grid")
    p.add_argument("--width", type=int, default=30)
    p.add_argument("--height", type=int, default=30)
    p.add_argument("--agents", type=parse_counts,
                   help="strategy counts, e.g. hawk=10,tit-for-tat=10,random=5 "
                        "(default: 5 of every strategy)")
    p.add_argument("--seed", type=int, default=None, help="random seed (reproducible runs)")
    p.add_argument("--memory", type=int, default=None,
                   help="rounds remembered per opponent by memory-based strategies "
                        "(default: unlimited; 0 = no memory)")
    p.add_argument("--cell", type=int, default=16, help="cell size in pixels")
    p.add_argument("--delay", type=int, default=60, help="ms between frames")
    p.add_argument("--steps-per-frame", type=int, default=1)
    p.add_argument("--headless", type=int, metavar="STEPS",
                   help="no GUI: run STEPS steps and print average payoffs")
    args = p.parse_args()

    sim = Simulation(args.width, args.height, args.agents, args.seed, args.memory)
    if args.headless:
        for _ in range(args.headless):
            sim.step()
        for name in sorted(sim.names, key=lambda n: -(sim.history[n][-1] or 0)):
            print(f"{name:20s} {sim.history[name][-1]}")
        return
    from gui import App  # imported here so --headless works without tkinter
    App(sim, args.cell, args.delay, args.steps_per_frame).run()


if __name__ == "__main__":
    main()
