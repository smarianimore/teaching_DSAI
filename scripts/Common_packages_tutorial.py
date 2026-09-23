#!/usr/bin/env python3
"""
TUTORIAL: numpy, pandas, matplotlib, scipy, pytorch, torchrl
=============================================================

Each library has its own self-contained function (imports live *inside* the
functions, so you can run one section even if the other libraries are not
installed).

Installation:
    pip install numpy pandas matplotlib scipy torch torchrl tensordict gymnasium

Usage:
    python scientific_python_tutorial.py                 # all sections
    python scientific_python_tutorial.py numpy pandas    # only some of them
    Available sections: numpy, pandas, matplotlib, scipy, pytorch, torchrl

Plots are saved in the "tutorial_figures/" folder.
Set SHOW_PLOTS = True to also open interactive windows.
"""

import sys
from pathlib import Path

from torchrl import render

SHOW_PLOTS = False              # True -> plt.show() (needs a display)
OUT_DIR = Path("tutorial_figures")
OUT_DIR.mkdir(exist_ok=True)


def header(text):
    """Print a banner to separate the sections in the output."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# =====================================================================
# 1. NUMPY - n-dimensional arrays and vectorized computation
# =====================================================================
def tutorial_numpy():
    import numpy as np

    header("1. NUMPY")

    # --- Creating arrays -----------------------------------------------
    a = np.array([1, 2, 3, 4, 5])                 # from a Python list
    m = np.array([[1, 2, 3],
                  [4, 5, 6]])                     # 2x3 matrix
    print("a:", a, "| dtype:", a.dtype)
    print("m.shape:", m.shape, "| m.ndim:", m.ndim, "| m.size:", m.size)

    # Handy constructors
    print("arange  :", np.arange(0, 10, 2))       # like range(), but an array
    print("linspace:", np.linspace(0, 1, 5))      # 5 evenly spaced points in [0,1]
    print("zeros   :\n", np.zeros((2, 3)))
    print("eye     :\n", np.eye(3))               # identity matrix

    # Random numbers: always use a seeded Generator for reproducibility
    rng = np.random.default_rng(seed=42)
    x = rng.normal(loc=0.0, scale=1.0, size=1000)  # 1000 samples from N(0,1)
    print("mean ~ 0:", x.mean().round(3), "| std ~ 1:", x.std().round(3))

    # --- Indexing and slicing --------------------------------------------
    print("m[0, 1]  =", m[0, 1])                  # row 0, column 1
    print("m[:, 1]  =", m[:, 1])                  # the whole column 1
    print("m[1, :2] =", m[1, :2])                 # row 1, first 2 columns
    print("a[::-1]  =", a[::-1])                  # reversed array

    # Boolean indexing (masks): very powerful for filtering
    mask = a > 2
    print("mask:", mask, "-> a[mask] =", a[mask])

    # --- Vectorized operations (no for loops!) -----------------------------
    print("a * 2   =", a * 2)
    print("a ** 2  =", a ** 2)
    print("sqrt(a) =", np.sqrt(a).round(3))

    # --- Reductions along an axis ------------------------------------------
    # axis=0 -> "collapse the rows"    (one result per column)
    # axis=1 -> "collapse the columns" (one result per row)
    print("total sum   :", m.sum())
    print("column sums :", m.sum(axis=0))
    print("row means   :", m.mean(axis=1))

    # --- Broadcasting: different shapes are automatically "stretched" -------
    row = np.array([10, 20, 30])                  # shape (3,)
    print("m + row:\n", m + row)                  # row is added to every row of m
    column = np.array([[1], [2], [3]])            # shape (3,1)
    print("column + row (3x3 table):\n", column + row)

    # --- reshape / concatenation ----------------------------------------------
    b = np.arange(12).reshape(3, 4)               # from 1D to 3x4
    print("b:\n", b, "\nb.T (transpose):\n", b.T)
    print("vstack:", np.vstack([b, b]).shape, "| hstack:", np.hstack([b, b]).shape)

    # --- Linear algebra -----------------------------------------------------------
    A = np.array([[3.0, 1.0],
                  [1.0, 2.0]])
    y = np.array([9.0, 8.0])
    sol = np.linalg.solve(A, y)                   # solves A @ sol = y
    print("solution of A x = y:", sol, "| check:", np.allclose(A @ sol, y))
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("eigenvalues:", eigenvalues.round(3))
    print("matrix product A @ A:\n", A @ A)       # @ = matrix product (NOT *)


# =====================================================================
# 2. PANDAS - tabular data (DataFrame) and time series
# =====================================================================
def tutorial_pandas():
    import numpy as np
    import pandas as pd

    header("2. PANDAS")
    rng = np.random.default_rng(0)

    # --- Creating a DataFrame ----------------------------------------------------
    df = pd.DataFrame({
        "city": rng.choice(["Milan", "Rome", "Turin"], size=12),
        "product": rng.choice(["A", "B"], size=12),
        "sales": rng.integers(10, 100, size=12),
        "price": rng.uniform(5, 20, size=12).round(2),
    })
    print(df.head(), "\n")                        # first rows
    df.info()                                     # types and non-null counts
    print(df.describe(), "\n")                    # stats on the numeric columns

    # --- New columns (vectorized operations, just like in numpy) ------------------
    df["revenue"] = df["sales"] * df["price"]

    # --- Selection -------------------------------------------------------------------
    print("Single column (Series):\n", df["city"].head(3), "\n")
    print("loc  (by label): rows 0-2, columns city and revenue\n",
          df.loc[0:2, ["city", "revenue"]], "\n")
    print("iloc (by position): first 2 rows, first 2 columns\n",
          df.iloc[:2, :2], "\n")

    # Boolean filter
    high = df[(df["revenue"] > 500) & (df["city"] == "Milan")]
    print("Milan with revenue > 500:\n", high, "\n")

    # --- Sorting ----------------------------------------------------------------------
    print("Top 3 revenues:\n", df.sort_values("revenue", ascending=False).head(3), "\n")

    # --- groupby: split -> apply -> combine ---------------------------------------------
    by_city = df.groupby("city")["revenue"].agg(["sum", "mean", "count"])
    print("Aggregation by city:\n", by_city.round(1), "\n")

    # Pivot table (city x product)
    pivot = df.pivot_table(index="city", columns="product",
                           values="revenue", aggfunc="sum", fill_value=0)
    print("Pivot table:\n", pivot.round(1), "\n")

    # --- Merge (like an SQL JOIN) ----------------------------------------------------------
    regions = pd.DataFrame({"city": ["Milan", "Rome", "Turin"],
                            "region": ["Lombardy", "Lazio", "Piedmont"]})
    merged = df.merge(regions, on="city", how="left")
    print("After the merge:\n", merged[["city", "region", "revenue"]].head(3), "\n")

    # --- Missing values -----------------------------------------------------------------------
    d2 = df.copy()
    d2.loc[[2, 5], "price"] = np.nan              # insert two NaNs
    print("NaNs per column:\n", d2.isna().sum(), "\n")
    d2["price"] = d2["price"].fillna(d2["price"].mean())  # fill with the mean
    # alternative: d2 = d2.dropna()  -> drops the rows containing NaN

    # --- apply: run a Python function on every element/row -------------------------------------
    df["band"] = df["revenue"].apply(lambda r: "high" if r > 800 else "low")
    print(df["band"].value_counts(), "\n")

    # --- Time series ------------------------------------------------------------------------------
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    series = pd.Series(rng.normal(size=60).cumsum(), index=dates, name="value")
    print("Weekly mean (resample):\n", series.resample("W").mean().head(3), "\n")
    rolling_mean = series.rolling(window=7).mean()  # 7-day moving average
    print("Moving average (last 3):\n", rolling_mean.tail(3))

    # I/O: reading/writing files is a one-liner
    df.to_csv(OUT_DIR / "sales.csv", index=False)
    reloaded = pd.read_csv(OUT_DIR / "sales.csv")
    print("\nReloaded from CSV, shape:", reloaded.shape)


# =====================================================================
# 3. MATPLOTLIB - plots (object-oriented API: fig, ax)
# =====================================================================
def tutorial_matplotlib():
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")      # "windowless" backend: only saves to file
    import matplotlib.pyplot as plt
    import numpy as np

    header("3. MATPLOTLIB")
    rng = np.random.default_rng(1)

    # A Figure contains one or more Axes (the "panels" you draw on).
    # Tip: always use plt.subplots() and work on the ax objects.
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # --- (1) Line plot ---------------------------------------------------------------
    x = np.linspace(0, 2 * np.pi, 200)
    axs[0, 0].plot(x, np.sin(x), label="sin(x)", color="tab:blue")
    axs[0, 0].plot(x, np.cos(x), label="cos(x)", color="tab:orange", linestyle="--")
    axs[0, 0].set_title("Lines")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)

    # --- (2) Scatter plot with colors --------------------------------------------------
    px = rng.normal(size=150)
    py = px * 0.8 + rng.normal(scale=0.5, size=150)
    sc = axs[0, 1].scatter(px, py, c=py, cmap="viridis", s=25)
    fig.colorbar(sc, ax=axs[0, 1], label="value of y")
    axs[0, 1].set_title("Scatter")

    # --- (3) Histogram --------------------------------------------------------------------
    axs[1, 0].hist(rng.normal(size=1000), bins=30, color="tab:green", alpha=0.7)
    axs[1, 0].set_title("Histogram")

    # --- (4) Bars with error bars ---------------------------------------------------------
    categories = ["A", "B", "C"]
    values = [3.2, 4.5, 2.8]
    errors = [0.3, 0.5, 0.2]
    axs[1, 1].bar(categories, values, yerr=errors, capsize=5, color="tab:red")
    axs[1, 1].set_title("Bars + errors")

    fig.suptitle("A tour of matplotlib")
    fig.tight_layout()                            # avoids overlapping elements
    path = OUT_DIR / "matplotlib_demo.png"
    fig.savefig(path, dpi=120)
    print("Plot saved to:", path)

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)                                # free the memory


# =====================================================================
# 4. SCIPY - scientific algorithms (stats, optimize, integrate, interpolate)
# =====================================================================
def tutorial_scipy():
    import numpy as np
    from scipy import stats, optimize, integrate, interpolate

    header("4. SCIPY")

    # --- scipy.stats: distributions and statistical tests ---------------------------
    print("-- stats --")
    print("P(Z <= 1.96)       =", stats.norm.cdf(1.96).round(4))   # cumulative distribution
    print("97.5% quantile     =", stats.norm.ppf(0.975).round(4))  # inverse of the cdf

    g1 = stats.norm.rvs(loc=0.0, scale=1.0, size=100, random_state=1)
    g2 = stats.norm.rvs(loc=0.6, scale=1.0, size=100, random_state=2)
    t = stats.ttest_ind(g1, g2)                   # two-sample independent t-test
    print(f"t-test: statistic={t.statistic:.3f}, p-value={t.pvalue:.4f}")

    r, p = stats.pearsonr(g1, g1 + 0.5 * g2)      # Pearson correlation
    print(f"Pearson r={r:.3f} (p={p:.2g})")

    # --- scipy.optimize: minimization and curve fitting --------------------------------
    print("\n-- optimize --")

    def rosenbrock(v):
        """Test function with its minimum at (1, 1)."""
        return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2

    res = optimize.minimize(rosenbrock, x0=[-1.0, 2.0], method="BFGS")
    print("minimum found at:", res.x.round(3), "| success:", res.success)

    # curve_fit: estimates a function's parameters from noisy data
    def model(x, a, b):
        return a * np.exp(-b * x)

    xd = np.linspace(0, 4, 40)
    rng = np.random.default_rng(0)
    yd = model(xd, 2.5, 1.3) + rng.normal(scale=0.05, size=xd.size)
    popt, pcov = optimize.curve_fit(model, xd, yd)
    errors = np.sqrt(np.diag(pcov))               # uncertainty on the parameters
    print("estimated parameters (a, b):", popt.round(3), "+-", errors.round(3),
          "| true: (2.5, 1.3)")

    # --- scipy.integrate: integrals and differential equations ----------------------------
    print("\n-- integrate --")
    value, error = integrate.quad(np.sin, 0, np.pi)   # definite integral
    print(f"integral of sin(x) over [0, pi] = {value:.6f} (estimated error {error:.1e})")

    # ODE: dy/dt = -k*y, with exact solution y(t) = y0 * exp(-k t)
    k = 0.8
    sol = integrate.solve_ivp(lambda t, y: -k * y, t_span=(0, 5), y0=[1.0],
                              t_eval=np.linspace(0, 5, 6))
    exact = np.exp(-k * sol.t)
    print("Numerical ODE :", sol.y[0].round(4))
    print("Exact solution:", exact.round(4))

    # --- scipy.interpolate ------------------------------------------------------------------
    print("\n-- interpolate --")
    xs = np.linspace(0, 10, 8)
    ys = np.sin(xs)
    spline = interpolate.CubicSpline(xs, ys)      # cubic spline
    print("true sin(2.5):", np.sin(2.5).round(4),
          "| interpolated:", round(float(spline(2.5)), 4))


# =====================================================================
# 5. PYTORCH - tensors, autograd, neural networks
# =====================================================================
def tutorial_pytorch():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    header("5. PYTORCH")
    torch.manual_seed(0)                          # reproducibility

    # --- Tensors: like numpy arrays, but with GPU support and autograd -------------------
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("shape:", t.shape, "| dtype:", t.dtype, "| device:", t.device)
    print("t @ t:\n", t @ t)
    print("zeros:", torch.zeros(2, 3).shape, "| rand:", torch.rand(2).shape)

    # Interoperability with numpy (they share memory on CPU!)
    arr = np.arange(4, dtype=np.float32)
    t_from_np = torch.from_numpy(arr)
    print("from numpy:", t_from_np, "-> back to numpy:", t_from_np.numpy())

    # Device: move tensors/models to the GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device in use:", device)

    # --- Autograd: automatic differentiation ------------------------------------------------
    x = torch.tensor(2.0, requires_grad=True)     # "track" the operations on x
    y = x ** 3 + 2 * x                            # y = x^3 + 2x
    y.backward()                                  # computes dy/dx
    print(f"dy/dx at x=2: {x.grad.item()} (expected 3*4 + 2 = 14)")

    # --- A real training loop: regression y = 3x + 2 + noise ----------------------------------
    X = torch.linspace(-1, 1, 200).unsqueeze(1)   # shape (200, 1)
    Y = 3 * X + 2 + 0.1 * torch.randn_like(X)

    # Dataset + DataLoader: handle batching and shuffling
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)

    class Regressor(nn.Module):
        """Small network: 1 -> 16 -> 1 with ReLU activation."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):                     # defines the forward pass
            return self.net(x)

    model = Regressor().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(60):
        model.train()                             # training mode
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)                      # 1) forward pass
            loss = loss_fn(pred, yb)              # 2) compute the loss
            optimizer.zero_grad()                 # 3) reset the old gradients
            loss.backward()                       # 4) backpropagation
            optimizer.step()                      # 5) update the weights
            total_loss += loss.item() * len(xb)
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | loss = {total_loss / len(X):.4f}")

    # --- Evaluation without tracking gradients -------------------------------------------------
    model.eval()
    with torch.no_grad():
        probe = torch.tensor([[0.0], [1.0]], device=device)
        out = model(probe)
        print("f(0) expected ~2:", round(out[0].item(), 2),
              "| f(1) expected ~5:", round(out[1].item(), 2))

    # --- Saving and reloading the weights ---------------------------------------------------------
    path = OUT_DIR / "model.pt"
    torch.save(model.state_dict(), path)
    fresh = Regressor().to(device)
    fresh.load_state_dict(torch.load(path, map_location=device))
    print("Weights reloaded from:", path)


# =====================================================================
# 6. TORCHRL - reinforcement learning on top of PyTorch
# =====================================================================
# Key concepts:
#  - TensorDict : a "dictionary of tensors" with a shared batch_size; it is the
#                 format environment, policy, buffer and loss use to exchange data.
#  - Env        : the environment (here Gymnasium, wrapped in GymEnv).
#  - Policy     : a module that reads "observation" and writes "action".
#  - Collector  : gathers experience by running the policy in the environment.
#  - ReplayBuffer, Loss: transition memory and the loss function (DQN).
def tutorial_torchrl():
    import torch
    from tensordict import TensorDict
    from tensordict.nn import TensorDictSequential

    header("6. TORCHRL")
    torch.manual_seed(0)

    # --- TensorDict ------------------------------------------------------------------------
    td = TensorDict(
        {
            "observation": torch.randn(4, 3),     # 4 items in the batch, 3 features
            "reward": torch.zeros(4, 1),
            "nested": {"x": torch.ones(4, 2)},    # entries can be nested
        },
        batch_size=[4],                           # dimensions shared by all tensors
    )
    print("batch_size:", td.batch_size)
    print("td[0] (first item of the batch):", td[0]["observation"].shape)
    print("td[:2] (slice of the batch):", td[:2].batch_size)
    print("nested access:", td["nested", "x"].shape)

    # --- Environment -------------------------------------------------------------------------
    try:
        from torchrl.envs.libs.gym import GymEnv
        env = GymEnv("CartPole-v1")
    except Exception as e:                        # e.g. gymnasium not installed
        print("Could not create the environment (install 'gymnasium'):", e)
        return

    print("\nobservation spec:", env.observation_spec)
    print("action spec     :", env.action_spec)

    state = env.reset()                           # returns a TensorDict
    print("keys after reset:", list(state.keys()))
    state = env.rand_step(state)                  # one step with a random action
    print("reward at the first step:", state["next", "reward"].item())

    # rollout: runs a whole trajectory (here with random actions)
    rollout = env.rollout(max_steps=10)
    print("rollout batch_size:", rollout.batch_size,
          "| observations:", rollout["observation"].shape)

    # --- Policy ---------------------------------------------------------------------------------
    from torchrl.modules import MLP, QValueActor, EGreedyModule

    n_obs = env.observation_spec["observation"].shape[-1]      # 4 for CartPole
    n_act = env.action_spec.shape[-1]                          # 2 actions (one-hot)
    value_net = MLP(in_features=n_obs, out_features=n_act, num_cells=[64, 64])

    # QValueActor: computes the Q-values and picks the action with the highest Q (greedy)
    policy = QValueActor(value_net, in_keys=["observation"], spec=env.action_spec)

    # Add epsilon-greedy exploration (random choice with probability epsilon)
    exploration = EGreedyModule(spec=env.action_spec, eps_init=1.0,
                                eps_end=0.05, annealing_num_steps=500)
    explore_policy = TensorDictSequential(policy, exploration)

    rollout = env.rollout(max_steps=20, policy=policy)
    print("\nrollout with policy, length:", rollout.batch_size[0])

    # --- Replay buffer -------------------------------------------------------------------------------
    from torchrl.data import ReplayBuffer, LazyTensorStorage
    buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10_000))

    # --- Collector: gathers experience in "batches" of frames_per_batch steps ------------------
    from torchrl.collectors import AsyncCollector
    collector = AsyncCollector(
        lambda: GymEnv("CartPole-v1", render_mode="human"),            # function that creates the environment
        explore_policy,
        frames_per_batch=200,
        total_frames=1_000,
    )

    # --- DQN loss + mini training loop (didactic, not optimized) ------------------------------------
    from torchrl.objectives import DQNLoss
    loss_module = DQNLoss(policy, action_space=env.action_spec, delay_value=False)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-3)

    for i, batch in enumerate(collector):         # each batch: a TensorDict [200]
        buffer.extend(batch)                      # store the transitions
        for _ in range(10):                       # a few updates per batch
            sample = buffer.sample(64)
            losses = loss_module(sample)          # TensorDict with the "loss" key
            losses["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()
        exploration.step(batch.numel())           # anneals epsilon
        collector.update_policy_weights_()        # syncs the weights (useful with multiprocessing)
        print(f"batch {i} | loss = {losses['loss'].item():.4f} "
              f"| epsilon = {float(exploration.eps):.2f}")

    collector.shutdown()
    env.close()


# =====================================================================
# Entry point
# =====================================================================
SECTIONS = {
    "numpy": tutorial_numpy,
    "pandas": tutorial_pandas,
    "matplotlib": tutorial_matplotlib,
    "scipy": tutorial_scipy,
    "pytorch": tutorial_pytorch,
    "torchrl": tutorial_torchrl,
}


def main():
    choices = sys.argv[1:] or list(SECTIONS)
    for name in choices:
        if name not in SECTIONS:
            print(f"Unknown section: {name}. Available: {', '.join(SECTIONS)}")
            continue
        try:
            SECTIONS[name]()
        except ImportError as e:
            print(f"[{name}] missing package: {e}")


# The __main__ guard is required: torchrl's collector may spawn subprocesses,
# which re-import this file.
if __name__ == "__main__":
    main()