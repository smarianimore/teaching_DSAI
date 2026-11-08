"""Small synchronous and asyncio LINDA tuple spaces."""

import asyncio
import math
import re
import threading
import time
from multiprocessing.managers import BaseManager
from typing import Optional, Tuple, Union

TupleValue = Tuple[str, ...]
Template = Tuple[Union[str, re.Pattern], ...]


def _validate_tuple(value: TupleValue) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        raise TypeError("a tuple must be a tuple of strings")


def _validate_template(template: Template) -> None:
    if not isinstance(template, tuple) or not all(
        isinstance(item, str)
        or (isinstance(item, re.Pattern) and isinstance(item.pattern, str))
        for item in template
    ):
        raise TypeError("a template must be a tuple of strings or compiled string regexes")


def _matches(value: TupleValue, template: Template) -> bool:
    return len(value) == len(template) and all(
        item == pattern if isinstance(pattern, str) else pattern.fullmatch(item) is not None
        for item, pattern in zip(value, template)
    )


class TupleSpace:
    """Blocking tuple space; share across processes with TupleSpaceManager."""

    def __init__(self) -> None:
        self._tuples = []
        self._condition = threading.Condition()

    def out(self, value: TupleValue) -> None:
        _validate_tuple(value)
        with self._condition:
            self._tuples.append(value)
            self._condition.notify_all()

    def _find(self, template: Template, remove: bool) -> Optional[TupleValue]:
        for index, value in enumerate(self._tuples):
            if _matches(value, template):
                return self._tuples.pop(index) if remove else value
        return None

    def _wait(self, template: Template, remove: bool, timeout: Optional[float]) -> TupleValue:
        _validate_template(template)
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be a finite non-negative number")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                value = self._find(template, remove)
                if value is not None:
                    return value
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("no matching tuple")
                self._condition.wait(remaining)

    def in_(self, template: Template, timeout: Optional[float] = None) -> TupleValue:
        """Remove and return a matching tuple, blocking until one is available."""
        return self._wait(template, True, timeout)

    def rd(self, template: Template, timeout: Optional[float] = None) -> TupleValue:
        """Return a matching tuple without removing it."""
        return self._wait(template, False, timeout)

    def try_in(self, template: Template) -> Optional[TupleValue]:
        """Atomically remove a match, or return None without waiting."""
        _validate_template(template)
        with self._condition:
            return self._find(template, True)


class TupleSpaceManager(BaseManager):
    """Context-managed server providing TupleSpace proxies to processes."""


TupleSpaceManager.register("TupleSpace", TupleSpace)


class AsyncTupleSpace:
    """Tuple space for tasks on a single asyncio event loop, not for processes."""

    def __init__(self) -> None:
        self._tuples = []
        self._waiters = set()

    async def out(self, value: TupleValue) -> None:
        _validate_tuple(value)
        self._tuples.append(value)
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)

    async def _wait(self, template: Template, remove: bool) -> TupleValue:
        _validate_template(template)
        while True:
            for index, value in enumerate(self._tuples):
                if _matches(value, template):
                    return self._tuples.pop(index) if remove else value
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.add(waiter)
            try:
                await waiter
            finally:
                self._waiters.remove(waiter)

    async def in_(self, template: Template) -> TupleValue:
        """Suspend this operation until a matching tuple can be removed."""
        return await self._wait(template, True)

    async def rd(self, template: Template) -> TupleValue:
        """Suspend this operation until a matching tuple can be read."""
        return await self._wait(template, False)
