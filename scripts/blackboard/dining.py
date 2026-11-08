"""Dining philosophers using process-shared LINDA tuples and a Tk GUI."""

import argparse
import math
import multiprocessing as mp
import queue
import re
import threading
import tkinter as tk

if __package__:
    from .linda import TupleSpaceManager
else:
    from linda import TupleSpaceManager


def philosopher(index, count, scenario, meals, delay, space, events, stop, barrier):
    """Run one philosopher in its own process."""
    left, right = index, (index + 1) % count

    def report(kind, value):
        events.put((index, kind, value))

    def take(chopstick):
        report("state", f"waiting for C{chopstick}")
        while not stop.is_set():
            try:
                space.in_(("chopstick", str(chopstick)), timeout=0.2)
                report("take", chopstick)
                return True
            except TimeoutError:
                pass
        return False

    def put(chopstick):
        space.out(("chopstick", str(chopstick)))
        report("put", chopstick)

    def rendezvous():
        try:
            barrier.wait()
            return not stop.is_set()
        except threading.BrokenBarrierError:
            return False

    try:
        if scenario == "deadlock":
            if take(left) and rendezvous():
                take(right)
        elif scenario == "livelock":
            while not stop.is_set():
                if not take(left) or not rendezvous():
                    break
                report("state", f"cannot take C{right}; yielding")
                # All left chopsticks are held at this point, so all attempts fail.
                other = space.try_in(("chopstick", str(right)))
                if other is not None:
                    raise RuntimeError("livelock synchronization failed")
                if stop.wait(delay):
                    break
                if not rendezvous():
                    break
                put(left)
                report("state", "retrying")
                if not rendezvous() or stop.wait(delay):
                    break
        else:
            for meal in range(meals):
                report("state", "thinking")
                if stop.wait(delay):
                    break
                report("state", "waiting for room")
                room = None
                while room is None and not stop.is_set():
                    try:
                        room = space.in_(("room", re.compile(r"\d+")), timeout=0.2)
                    except TimeoutError:
                        pass
                if room is None or not take(left) or not take(right):
                    break
                report("state", f"eating ({meal + 1}/{meals})")
                if stop.wait(delay * 2):
                    break
                put(right)
                put(left)
                space.out(room)
                report("meal", meal + 1)
            else:
                report("state", "finished")
    except Exception as exc:
        report("error", f"{type(exc).__name__}: {exc}")
        stop.set()
        barrier.abort()
        raise


class DiningWindow:
    def __init__(self, root, count, scenario, meals, delay):
        self.root, self.count, self.scenario = root, count, scenario
        self.states = ["starting"] * count
        self.held = [set() for _ in range(count)]
        self.meals = [0] * count
        self.manager = TupleSpaceManager()
        self.manager.start()
        self.space = self.manager.TupleSpace()
        for i in range(count):
            self.space.out(("chopstick", str(i)))
        if scenario == "success":
            for i in range(count - 1):
                self.space.out(("room", str(i)))
        self.events = mp.Queue()
        self.stop = mp.Event()
        self.barrier = mp.Barrier(count)
        self.processes = [
            mp.Process(target=philosopher, args=(
                i, count, scenario, meals, delay, self.space, self.events,
                self.stop, self.barrier,
            ))
            for i in range(count)
        ]
        root.title(f"LINDA dining philosophers — {scenario}")
        root.protocol("WM_DELETE_WINDOW", self.close)
        heading = tk.Label(root, text=f"{scenario.upper()}  |  {count} philosophers",
                           font=("TkDefaultFont", 16, "bold"))
        heading.pack(pady=(10, 2))
        self.canvas = tk.Canvas(root, width=720, height=610, bg="#f4f6fa",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(root, text="", anchor="w", justify="left")
        self.status.pack(fill="x", padx=12, pady=8)
        tk.Button(root, text="Close", command=self.close).pack(pady=(0, 10))
        self.draw()
        for process in self.processes:
            process.start()
        self.root.after(40, self.update)

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 720), max(canvas.winfo_height(), 610)
        cx, cy = width / 2, height / 2
        radius = min(width, height) * 0.34
        canvas.create_oval(cx - radius * .57, cy - radius * .57,
                           cx + radius * .57, cy + radius * .57,
                           fill="#dbe5ed", outline="#9eb0c0", width=2)
        canvas.create_text(cx, cy, text="LINDA\nTUPLE SPACE", fill="#40576c",
                           font=("TkDefaultFont", 12, "bold"), justify="center")
        for i in range(self.count):
            angle = 2 * math.pi * i / self.count - math.pi / 2
            x, y = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
            state = self.states[i]
            color = ("#35a86b" if state.startswith("eating") else
                     "#efa83b" if state.startswith(("waiting", "cannot")) else
                     "#7289c4" if state == "finished" else "#8b9aac")
            canvas.create_oval(x-24, y-24, x+24, y+24, fill=color, outline="")
            canvas.create_text(x, y, text=f"P{i}", fill="white",
                               font=("TkDefaultFont", 11, "bold"))
            canvas.create_text(x, y+38, text=f"{state}\nmeals: {self.meals[i]}",
                               fill="#263849", justify="center", width=140)
            # C_i is between P_i and P_(i-1), and is P_i's left chopstick.
            stick_angle = angle - math.pi / self.count
            sx, sy = cx + radius * .72 * math.cos(stick_angle), cy + radius * .72 * math.sin(stick_angle)
            owners = [p for p, sticks in enumerate(self.held) if i in sticks]
            label = f"C{i}: P{owners[0]}" if owners else f"C{i}: free"
            canvas.create_text(sx, sy, text=label, fill="#7b3f16" if owners else "#37785b",
                               font=("TkDefaultFont", 9, "bold"))
        self.status.config(text="Green: eating   |   Amber: waiting/yielding   |   "
                                "Blue: finished   |   Chopsticks show their owner")

    def update(self):
        changed = False
        while True:
            try:
                index, kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "state":
                self.states[index] = value
            elif kind == "take":
                for sticks in self.held:
                    sticks.discard(value)
                self.held[index].add(value)
            elif kind == "put":
                self.held[index].discard(value)
            elif kind == "meal":
                self.meals[index] = value
            elif kind == "error":
                self.states[index] = f"ERROR: {value}"
        if changed:
            self.draw()
        self.root.after(40, self.update)

    def close(self):
        self.stop.set()
        self.barrier.abort()
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()
        self.manager.shutdown()
        self.events.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("deadlock", "livelock", "success"),
                        default="success")
    parser.add_argument("--philosophers", type=int, default=5)
    parser.add_argument("--meals", type=int, default=3,
                        help="meals per philosopher in the success scenario")
    parser.add_argument("--delay", type=float, default=0.7,
                        help="animation delay in seconds")
    args = parser.parse_args()
    if not 2 <= args.philosophers <= 12:
        parser.error("--philosophers must be between 2 and 12")
    if args.meals < 1:
        parser.error("--meals must be positive")
    if not math.isfinite(args.delay) or args.delay <= 0:
        parser.error("--delay must be a positive finite number")
    root = tk.Tk()
    DiningWindow(root, args.philosophers, args.scenario, args.meals, args.delay)
    root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()
