# LINDA tuple spaces and dining philosophers

LINDA coordinates independent agents through a shared **tuple space**. `out`
adds a tuple, `in_` waits for and removes one matching tuple, and `rd` waits for
and reads one without removing it (`in_` has an underscore because `in` is a
Python keyword). Stored values are Python tuples of strings. A template is a
tuple of literal strings and/or compiled string regular expressions; its
length must match, and each regex must match the **whole** corresponding
field. Duplicates are allowed. A match is selected in insertion order, but
there is no guarantee of fairness between waiting agents.

## Blocking and asynchronous spaces

`TupleSpace` blocks the calling thread until a match arrives. To share it
between processes, use `TupleSpaceManager`: its server owns the space and
agents use its proxy. Start the manager under the usual `if __name__ ==
"__main__":` guard when using multiprocessing. The manager must remain alive
until all agents have finished.

```python
import re
from linda import TupleSpaceManager

with TupleSpaceManager() as manager:
    space = manager.TupleSpace()  # pass this proxy to worker processes
    space.out(("job", "42"))
    assert space.rd(("job", re.compile(r"\d+"))) == ("job", "42")
    assert space.in_(("job", "42")) == ("job", "42")
```

`in_` and `rd` accept an optional `timeout` in seconds and raise
`TimeoutError` if no match arrives. `try_in` atomically removes a match or
returns `None` immediately. A blocked synchronous call suspends the calling
thread; in a process running only that agent, the agent makes no further
progress until the call completes.

`AsyncTupleSpace` is a separate, in-process implementation for tasks on a
**single asyncio event loop**, not a process-shared proxy. Its operations are
coroutines. Awaiting a pending operation suspends only that task; other tasks
continue. To let the same agent do other work before awaiting the result,
schedule the operation as a task (a future):

```python
import asyncio
from linda import AsyncTupleSpace

async def agent():
    space = AsyncTupleSpace()
    pending = asyncio.create_task(space.in_(("answer", "42")))
    await asyncio.sleep(0)  # other work can run while the operation waits
    await space.out(("answer", "42"))
    print(await pending)

asyncio.run(agent())
```

Use `asyncio.wait_for` for an async timeout. Cancelling a waiting task does not
consume a tuple. Neither implementation promises distributed persistence or
fair scheduling.

## Dining philosophers

Philosophers sit around a table, each needing both adjacent chopsticks to eat.
Each chopstick is represented by a unique `("chopstick", "N")` tuple. Taking
one with `in_` removes it so that nobody else can take the same chopstick;
returning it with `out` makes it available again. Each philosopher runs in its
own process and communicates through the blocking, manager-hosted space.

Requirements: Python 3.8+ and Tkinter (on Debian/Ubuntu, install
`python3-tk`); no third-party packages. A graphical desktop is required. From
the repository root:

```sh
python scripts/blackboard/dining.py --scenario deadlock
python scripts/blackboard/dining.py --scenario livelock
python scripts/blackboard/dining.py --scenario success
```

Use `--philosophers N` (2–12, default 5), `--delay SECONDS` (default 0.7),
and `--meals N` (default 3, success only). For example:

```sh
python scripts/blackboard/dining.py --scenario success --philosophers 4 --meals 2 --delay 0.4
```

- **Deadlock:** everyone takes their left chopstick, then a barrier ensures
  everyone holds one before asking for the right. Every `in_` waits forever:
  no philosopher can eat or release their chopstick. Closing the window stops
  the processes.
- **Livelock:** everyone takes their left chopstick, observes that the right
  is unavailable using `try_in`, and politely puts the left one down. Barriers
  keep their retries in lockstep: they keep moving but never eat. The barriers
  deliberately make this unlucky schedule reproducible.
- **Success:** there are N−1 `("room", "N")` permits. A philosopher must
  remove a permit before taking chopsticks and return it afterward. At least
  one philosopher is outside the competition, breaking the circular wait.
  Each philosopher stops after the requested number of meals. This finite demo
  completes under ordinary scheduling; the tuple space does not guarantee
  starvation freedom for arbitrary, indefinitely running agents.

The GUI shows each philosopher's state and meal count, and the owner of each
chopstick. Event messages reflect actions asynchronously, so the display can
briefly lag behind the processes. Close the window to stop a run.

Run the automated checks from the repository root with:

```sh
python -m unittest discover -s scripts/blackboard -p 'test_*.py'
```
