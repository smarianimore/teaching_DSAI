"""Minimal tic-tac-toe with pygame and interchangeable players.

Players: human, random, minimax. Python 3.9+ required.
The game logic works without pygame.

Interactive mode (needs pygame: python -m pip install pygame)
    python tic_tac_toe.py --x human --o minimax
    python tic_tac_toe.py --x random --o minimax --delay 500
  X starts. Click to play; R restarts; Escape quits.
  --delay sets the pause (ms) before each AI move.

Play mode (headless AI vs AI; needs pandas and matplotlib)
    python tic_tac_toe.py --games 500 --x random --o minimax --start random
  Runs N games with no window and writes one CSV row per game
  (game, x_strategy, o_strategy, first_mark, first_strategy,
  winner_mark, winner_strategy, moves), then plots and saves:
    1. wins per AI strategy (and draws);
    2. outcomes (first player won / second player won / draw) per starter;
    3. average number of moves per winning strategy.
  --start random|alternate|x|o  who moves first (default: random)
  --csv FILE / --plot FILE      output paths (default: tic_tac_toe_results.csv/.png)
  Human players are not allowed in play mode.
"""

import argparse
import csv
import random
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)  # every new board state is a new object
class Board:
    """Immutable state; cells are indexed left-to-right, top-to-bottom."""

    cells: tuple = ("",) * 9

    def available_moves(self):
        return [i for i, cell in enumerate(self.cells) if not cell]

    def winner(self):
        lines = (
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        )
        for a, b, c in lines:
            if self.cells[a] and self.cells[a] == self.cells[b] == self.cells[c]:
                return self.cells[a]
        return None

    def is_terminal(self):
        return self.winner() is not None or not self.available_moves()

    def with_move(self, index, mark):
        if mark not in ("X", "O"):
            raise ValueError("Mark must be X or O")
        if self.is_terminal() or index not in self.available_moves():
            raise ValueError("Illegal move")
        return Board(self.cells[:index] + (mark,) + self.cells[index + 1:])


class Game:
    """Owns the current board, enforces legal moves, and alternates turns."""

    def __init__(self, first="X"):
        self.board = Board()
        self.turn = first
        self.moves = 0

    def move(self, index):
        """Return False for an illegal move without changing the game."""
        try:
            self.board = self.board.with_move(index, self.turn)
        except ValueError:
            return False
        self.moves += 1
        if not self.board.is_terminal():
            self.turn = "O" if self.turn == "X" else "X"
        return True


class HumanPlayer:
    """Converts a clicked cell into a move; None means wait for input."""

    def choose_move(self, board, mark, clicked_cell=None):
        if not board.is_terminal() and clicked_cell in board.available_moves():
            return clicked_cell
        return None


class RandomPlayer:
    """Chooses uniformly among legal moves."""

    def choose_move(self, board, mark):
        if board.is_terminal():
            return None
        return random.choice(board.available_moves())


class MinimaxPlayer:
    """Searches the complete game tree: X maximizes, O minimizes."""

    @staticmethod
    @lru_cache(maxsize=None)  # technicality, ignore this (cache scores for already visited states)
    def _score(board, turn):
        winner = board.winner()
        if winner:
            return 1 if winner == "X" else -1
        if not board.available_moves():
            return 0
        next_turn = "O" if turn == "X" else "X"
        scores = (
            MinimaxPlayer._score(board.with_move(i, turn), next_turn)  # recursive evaluation: expand nodes to explore the tree until the leaves
            for i in board.available_moves()
        )
        return max(scores) if turn == "X" else min(scores)

    def choose_move(self, board, mark):
        if board.is_terminal():
            return None
        next_turn = "O" if mark == "X" else "X"
        best = max if mark == "X" else min  # store a function into a variable :)
        return best(
            board.available_moves(),
            key=lambda i: self._score(board.with_move(i, mark), next_turn),  # max/min criteria
        )


class PygameView:
    """Draws the game and maps window coordinates to board indices."""

    CELL = 140
    SIZE = CELL * 3

    def __init__(self):
        import pygame

        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.SIZE, self.SIZE + 90))
        pygame.display.set_caption("Tic-tac-toe")
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 23)

    def cell_at(self, position):
        x, y = position
        if 0 <= x < self.SIZE and 0 <= y < self.SIZE:
            return (y // self.CELL) * 3 + x // self.CELL
        return None

    def draw(self, game, player_name):
        pg = self.pg
        self.screen.fill((245, 245, 245))
        for i in (1, 2):
            p = i * self.CELL
            pg.draw.line(self.screen, (60, 60, 60), (p, 0), (p, self.SIZE), 3)
            pg.draw.line(self.screen, (60, 60, 60), (0, p), (self.SIZE, p), 3)
        for i, mark in enumerate(game.board.cells):
            x = (i % 3) * self.CELL + self.CELL // 2
            y = (i // 3) * self.CELL + self.CELL // 2
            r = 42
            if mark == "X":
                pg.draw.line(self.screen, (45, 100, 190), (x-r, y-r), (x+r, y+r), 7)
                pg.draw.line(self.screen, (45, 100, 190), (x-r, y+r), (x+r, y-r), 7)
            elif mark == "O":
                pg.draw.circle(self.screen, (205, 80, 65), (x, y), r, 7)
        winner = game.board.winner()
        if winner:
            status = f"{winner} wins!"
        elif game.board.is_terminal():
            status = "Draw!"
        else:
            status = f"{game.turn}'s turn ({player_name})"
        self.screen.blit(self.font.render(status, True, (30, 30, 30)), (16, self.SIZE + 15))
        hint = self.small_font.render("Click to play | R: restart | Esc: quit", True, (80, 80, 80))
        self.screen.blit(hint, (16, self.SIZE + 53))
        pg.display.flip()


CSV_FIELDS = ["game", "x_strategy", "o_strategy", "first_mark", "first_strategy",
              "winner_mark", "winner_strategy", "moves"]


def play_game(players, names, first):
    """Play one headless game; return a CSV row (without the game id)."""
    game = Game(first)
    while not game.board.is_terminal():
        game.move(players[game.turn].choose_move(game.board, game.turn))
    mark = game.board.winner()
    return {
        "x_strategy": names["X"], "o_strategy": names["O"],
        "first_mark": first, "first_strategy": names[first],
        "winner_mark": mark or "draw",
        "winner_strategy": names[mark] if mark else "draw",
        "moves": game.moves,
    }


def run_games(n, x_name, o_name, types, start, csv_path):
    names = {"X": x_name, "O": o_name}
    players = {mark: types[name]() for mark, name in names.items()}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, CSV_FIELDS)
        writer.writeheader()
        for g in range(n):
            first = {"x": "X", "o": "O", "alternate": "XO"[g % 2]}.get(start) or random.choice("XO")
            writer.writerow({"game": g + 1, **play_game(players, names, first)})


def plot_results(csv_path, png_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    # one row per (game, seat) so strategies are comparable even if both seats share one
    seats = pd.concat([
        df.assign(strategy=df["x_strategy"], seat="X"),
        df.assign(strategy=df["o_strategy"], seat="O"),
    ])
    seats["won"] = seats["winner_mark"] == seats["seat"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    label = lambda m: f"{m} ({df[m.lower() + '_strategy'].iloc[0]})"

    wins = seats.groupby(["seat", "strategy"])["won"].sum()
    wins.index = [f"{st} as {se}" for se, st in wins.index]
    wins["draw"] = (df["winner_mark"] == "draw").sum()
    wins.plot.bar(ax=axes[0], color="tab:blue", rot=20)
    axes[0].set_title("Wins per AI player strategy")
    axes[0].set_ylabel("wins")

    df["outcome"] = df.apply(
        lambda r: "draw" if r["winner_mark"] == "draw"
        else "first player won" if r["winner_mark"] == r["first_mark"] else "second player won", axis=1)
    first = df.groupby(["first_mark", "outcome"]).size().unstack(fill_value=0)
    first = first.reindex(columns=["first player won", "second player won", "draw"], fill_value=0)
    first.index = [f"{label(m)} starts" for m in first.index]
    first.plot.bar(ax=axes[1], rot=20)
    axes[1].set_title("Outcomes per who starts first")
    axes[1].set_ylabel("games")

    avg = df.groupby("winner_strategy")["moves"].mean()
    avg.plot.bar(ax=axes[2], color="tab:green", rot=20)
    axes[2].set_title("Average moves per winning strategy")
    axes[2].set_ylabel("moves until end state")
    fig.suptitle(f"{len(df)} games: X={df['x_strategy'][0]} vs O={df['o_strategy'][0]}")
    fig.tight_layout()
    fig.savefig(png_path)
    print(f"Saved {png_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    types = {"human": HumanPlayer, "random": RandomPlayer, "minimax": MinimaxPlayer}
    parser.add_argument("--x", choices=types, default="human")
    parser.add_argument("--o", choices=types, default="minimax")
    parser.add_argument("--delay", type=int, default=1000, help="AI turn delay in milliseconds")
    parser.add_argument("--games", type=int, default=0,
                        help="play mode: run N headless games (no human) and plot metrics")
    parser.add_argument("--start", choices=["random", "alternate", "x", "o"], default="random",
                        help="play mode: who moves first")
    parser.add_argument("--csv", default="tic_tac_toe_results.csv")
    parser.add_argument("--plot", default="tic_tac_toe_results.png")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    if args.games < 0:
        parser.error("--games must be nonnegative")
    if args.games:
        if "human" in (args.x, args.o):
            parser.error("play mode needs AI players (random, minimax)")
        run_games(args.games, args.x, args.o, types, args.start, args.csv)
        print(f"Saved {args.csv}")
        plot_results(args.csv, args.plot)
        return

    try:
        view = PygameView()
    except ImportError:
        parser.exit(1, "Install pygame first: python -m pip install pygame\n")
    pg = view.pg
    clock = pg.time.Clock()
    names = {"X": args.x, "O": args.o}
    players = {mark: types[name]() for mark, name in names.items()}
    game = Game()
    next_ai = pg.time.get_ticks() + args.delay
    running = True
    try:
        while running:
            clicked_cell = None
            restarted = False
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key == pg.K_r:
                        game = Game()
                        next_ai = pg.time.get_ticks() + args.delay
                        restarted = True
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    clicked_cell = view.cell_at(event.pos)
            if not running:
                break
            if not restarted and not game.board.is_terminal():
                player = players[game.turn]
                move = None
                if isinstance(player, HumanPlayer):
                    move = player.choose_move(game.board, game.turn, clicked_cell)
                elif pg.time.get_ticks() >= next_ai:
                    move = player.choose_move(game.board, game.turn)
                if move is not None and game.move(move):
                    next_ai = pg.time.get_ticks() + args.delay
            view.draw(game, names[game.turn])
            clock.tick(60)
    finally:
        pg.quit()


if __name__ == "__main__":
    main()
