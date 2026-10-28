"""Tkinter GUI: arena on the left, payoff dashboard on the right.

Only the standard library is used (tkinter), so nothing needs installing.
Everything is drawn on plain Canvas widgets.
"""
import tkinter as tk

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#008080", "#9a6324", "#808000", "#000075", "#f032e6"]


class App:
    def __init__(self, sim, cell=16, delay_ms=60, steps_per_frame=1):
        self.sim = sim
        self.cell = cell
        self.delay_ms = delay_ms
        self.steps_per_frame = steps_per_frame
        self.running = True
        self.colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sim.names)}

        self.root = tk.Tk()
        self.root.title("Axelrod tournament on a grid")
        size_w, size_h = sim.width * cell, sim.height * cell
        self.arena = tk.Canvas(self.root, width=size_w, height=size_h, bg="white")
        self.arena.grid(row=0, column=0, padx=5, pady=5)
        # The plot needs room for the legend below it, hence the extra height.
        self.plot = tk.Canvas(self.root, width=520, height=max(size_h, 420), bg="white")
        self.plot.grid(row=0, column=1, padx=5, pady=5)
        self.status = tk.Label(self.root, text="")
        self.status.grid(row=1, column=0, columnspan=2)
        buttons = tk.Frame(self.root)
        buttons.grid(row=2, column=0, columnspan=2, pady=4)
        self.toggle = tk.Button(buttons, text="Pause", command=self.toggle_run)
        self.toggle.pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Step", command=self.single_step).pack(side=tk.LEFT, padx=4)

    def toggle_run(self):
        self.running = not self.running
        self.toggle.config(text="Pause" if self.running else "Resume")

    def single_step(self):
        self.running = False
        self.toggle.config(text="Resume")
        self.sim.step()
        self.draw()

    def draw(self):
        self.draw_arena()
        self.draw_dashboard()
        self.status.config(text=f"step {self.sim.step_count} | agents {len(self.sim.agents)}")

    def draw_arena(self):
        c, cv = self.cell, self.arena
        cv.delete("all")  # redraw from scratch each frame: simple and fast enough
        for ag in self.sim.agents:
            x, y = ag.x * c, ag.y * c
            cv.create_oval(x + 2, y + 2, x + c - 2, y + c - 2,
                           fill=self.colors[ag.strategy.name], outline="")

    def draw_dashboard(self):
        cv = self.plot
        cv.delete("all")
        w, h = int(cv["width"]), int(cv["height"])
        left, right, top, bottom = 45, 10, 15, 150  # bottom margin holds the legend
        pw, ph = w - left - right, h - top - bottom
        ymax = 5.0  # highest possible payoff per game
        cv.create_rectangle(left, top, left + pw, top + ph, outline="black")
        for v in range(0, 6):  # y axis ticks
            y = top + ph - v / ymax * ph
            cv.create_line(left - 3, y, left, y)
            cv.create_text(left - 8, y, text=str(v), anchor="e")
        cv.create_text(left + pw / 2, top + ph + 15,
                       text="time (steps)  -  y: average payoff per game", anchor="n")
        n = self.sim.step_count
        if n >= 2:
            for name in self.sim.names:
                # Downsample to at most ~pw points so long runs stay fast.
                stride = max(1, n // pw)
                pts = []
                for i in range(0, n, stride):
                    v = self.sim.history[name][i]
                    if v is not None:
                        pts += [left + i / (n - 1) * pw, top + ph - v / ymax * ph]
                if len(pts) >= 4:
                    cv.create_line(*pts, fill=self.colors[name], width=2)
        # Legend with the latest value for each strategy, best first.
        latest = {nm: (self.sim.history[nm][-1] if self.sim.history[nm] else None)
                  for nm in self.sim.names}
        ranked = sorted(latest, key=lambda k: -(latest[k] or 0))
        for i, name in enumerate(ranked):  # i + 1 is the rank shown in the legend
            x = left + (i % 2) * (pw // 2)
            y = top + ph + 45 + (i // 2) * 18
            cv.create_rectangle(x, y, x + 10, y + 10, fill=self.colors[name], outline="")
            val = "n/a" if latest[name] is None else f"{latest[name]:.2f}"
            cv.create_text(x + 15, y + 5, text=f"{i + 1}. {name}: {val}", anchor="w")

    def loop(self):
        # tkinter is event-driven: instead of `while True`, we schedule this
        # function again with `after`, so the window stays responsive.
        if self.running:
            for _ in range(self.steps_per_frame):
                self.sim.step()
            self.draw()
        self.root.after(self.delay_ms, self.loop)

    def run(self):
        self.draw()
        self.loop()
        self.root.mainloop()
