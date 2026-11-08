import asyncio
import multiprocessing as mp
import queue
import re
import threading
import time
import unittest

from dining import philosopher
from linda import AsyncTupleSpace, TupleSpace, TupleSpaceManager


def process_reader(space, ready, result):
    ready.set()
    result.put(space.in_(("message", re.compile(r"\d+")), timeout=3))


class TupleSpaceTests(unittest.TestCase):
    def test_matching_duplicates_and_reads(self):
        space = TupleSpace()
        space.out(("number", "12"))
        space.out(("number", "12"))
        space.out(("number", "abc"))
        pattern = ("number", re.compile(r"\d+"))
        self.assertEqual(space.rd(pattern), ("number", "12"))
        self.assertEqual(space.rd(pattern), ("number", "12"))
        self.assertEqual(space.in_(pattern), ("number", "12"))
        self.assertEqual(space.in_(pattern), ("number", "12"))
        self.assertIsNone(space.try_in(pattern))
        self.assertEqual(space.in_(("number", "abc")), ("number", "abc"))
        space.out(("literal", "a.*"))
        self.assertIsNone(space.try_in(("literal", "abc")))
        self.assertEqual(space.in_(("literal", "a.*")), ("literal", "a.*"))
        space.out(())
        self.assertEqual(space.in_(()), ())

    def test_waiting_readers_and_timeout(self):
        space = TupleSpace()
        results = []
        readers = [threading.Thread(target=lambda: results.append(space.rd(("x",), 2)))
                   for _ in range(2)]
        for reader in readers:
            reader.start()
        space.out(("x",))
        for reader in readers:
            reader.join(3)
            self.assertFalse(reader.is_alive())
        self.assertEqual(results, [("x",), ("x",)])
        self.assertEqual(space.in_(("x",)), ("x",))
        with self.assertRaises(TimeoutError):
            space.in_(("x",), timeout=0)
        with self.assertRaises(ValueError):
            space.rd(("x",), timeout=-1)

    def test_validation(self):
        space = TupleSpace()
        with self.assertRaises(TypeError):
            space.out(("x", 1))
        with self.assertRaises(TypeError):
            space.in_([re.compile("x")], timeout=0)
        with self.assertRaises(TypeError):
            space.try_in((re.compile(b"x"),))

    def test_shared_across_spawned_processes(self):
        ctx = mp.get_context("spawn")
        with TupleSpaceManager(ctx=ctx) as manager:
            space = manager.TupleSpace()
            ready, result = ctx.Event(), ctx.Queue()
            process = ctx.Process(target=process_reader, args=(space, ready, result))
            process.start()
            try:
                self.assertTrue(ready.wait(3))
                self.assertTrue(process.is_alive())
                space.out(("message", "42"))
                self.assertEqual(result.get(timeout=3), ("message", "42"))
                process.join(3)
                self.assertEqual(process.exitcode, 0)
                self.assertIsNone(space.try_in(("message", "42")))
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join()
                result.close()


class AsyncTupleSpaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_tasks_keep_running_and_read_does_not_remove(self):
        space = AsyncTupleSpace()
        reader = asyncio.create_task(space.rd(("x", re.compile(r"\d+"))))
        await asyncio.sleep(0)
        self.assertFalse(reader.done())
        await space.out(("x", "123"))
        self.assertEqual(await asyncio.wait_for(reader, 1), ("x", "123"))
        self.assertEqual(await space.in_(("x", "123")), ("x", "123"))

    async def test_competing_consumers_and_cancellation(self):
        space = AsyncTupleSpace()
        first = asyncio.create_task(space.in_(("x",)))
        second = asyncio.create_task(space.in_(("x",)))
        await asyncio.sleep(0)
        await space.out(("x",))
        await asyncio.sleep(0)
        self.assertEqual(sum(task.done() for task in (first, second)), 1)
        pending = first if not first.done() else second
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        await space.out(("x",))
        self.assertEqual(await space.in_(("x",)), ("x",))
        self.assertEqual(len(space._waiters), 0)

    async def test_cancel_after_notification_does_not_lose_tuple(self):
        space = AsyncTupleSpace()
        task = asyncio.create_task(space.in_(("x",)))
        await asyncio.sleep(0)
        await space.out(("x",))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(await space.in_(("x",)), ("x",))


class DiningTests(unittest.TestCase):
    def run_scenario(self, scenario, count):
        ctx = mp.get_context("spawn")
        with TupleSpaceManager(ctx=ctx) as manager:
            space = manager.TupleSpace()
            for i in range(count):
                space.out(("chopstick", str(i)))
            if scenario == "success":
                for i in range(count - 1):
                    space.out(("room", str(i)))
            events, stop, barrier = ctx.Queue(), ctx.Event(), ctx.Barrier(count)
            processes = [ctx.Process(target=philosopher, args=(
                i, count, scenario, 2, 0.02, space, events, stop, barrier,
            )) for i in range(count)]
            for process in processes:
                process.start()
            received = []
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        event = events.get(timeout=0.1)
                    except queue.Empty:
                        if any(p.exitcode not in (None, 0) for p in processes):
                            self.fail("philosopher process failed")
                        continue
                    received.append(event)
                    if scenario == "success":
                        done = {i for i, kind, value in received
                                if kind == "state" and value == "finished"}
                    elif scenario == "deadlock":
                        done = {i for i, kind, value in received
                                if kind == "state" and value == f"waiting for C{(i + 1) % count}"}
                    else:
                        done = {i for i in range(count)
                                if sum(1 for p, kind, _ in received
                                       if p == i and kind == "put") >= 2}
                    if len(done) == count:
                        break
                else:
                    self.fail(f"{scenario} did not reach its expected state: {received}")
                self.assertFalse(any(kind == "error" for _, kind, _ in received))
                if scenario != "success":
                    self.assertFalse(any(kind == "meal" for _, kind, _ in received))
                    self.assertTrue(all(p.is_alive() for p in processes))
                else:
                    for i in range(count):
                        self.assertIn((i, "meal", 2), received)
                    for i in range(count):
                        self.assertEqual(space.rd(("chopstick", str(i)), timeout=1),
                                         ("chopstick", str(i)))
            finally:
                stop.set()
                barrier.abort()
                for process in processes:
                    process.join(3)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                events.close()

    def test_scenarios(self):
        for scenario in ("deadlock", "livelock", "success"):
            for count in (2, 5):
                with self.subTest(scenario=scenario, count=count):
                    self.run_scenario(scenario, count)


if __name__ == "__main__":
    unittest.main()
