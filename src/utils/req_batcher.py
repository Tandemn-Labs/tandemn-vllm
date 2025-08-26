import asyncio
import time
from collections.abc import Coroutine
from typing import Any, Callable, List, NamedTuple


class Request(NamedTuple):
    prompt: str
    max_tokens: int
    model_name: str


class Batcher:
    def __init__(
        self,
        model_name: str,
        max_req: int,
        max_time: int,
        process_req_fn: Callable[[List[Request]], Coroutine[Any, Any, int]],
    ):
        self.model_name = model_name
        self.max_req = max_req
        self.max_time = max_time
        self.process_req_fn = process_req_fn

        self._queue = []
        self._attr_lock = asyncio.Lock()
        self._start_time = None
        self._timer_task = None
        self._inflight = set()

    # Add request to the queue
    async def add(self, req: Request) -> None:
        async with self._attr_lock:
            self._queue.append(req)

            # First in queue, start timer
            if len(self._queue) == 1:
                self._start_time = time.monotonic()
                self._timer_task = asyncio.create_task(self._timer_coro())

            # Queue is now filled
            elif len(self._queue) >= self.max_req:
                self.flush_req()

            # Otherwise, just wait for timer or more requests

    # Wait till we hit timeout
    async def _timer_coro(self) -> None:
        try:
            diff = time.monotonic() - self._start_time
            if diff < self.max_time:
                await asyncio.sleep(self.max_time - diff)

            async with self._attr_lock:
                self.flush_req()

        except asyncio.CancelledError:
            pass  # Do nothing since cancel is intentional

    # This function is only called when lock is held
    def flush_req(self) -> None:
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        self._start_time = None

        queue = self._queue.copy()
        self._queue.clear()

        try:
            task = asyncio.create_task(self.process_req_fn(queue))
            self._inflight.add(task)

            def _done(t: asyncio.Task) -> None:
                self._inflight.discard(t)
                # Check if task had an exception
                if t.exception():
                    print(
                        f"Warning: Task in batcher failed with exception: {t.exception()}"
                    )

            task.add_done_callback(_done)

        except Exception as e:
            print(f"Error creating task in batcher: {e}")
            # Re-queue the requests if task creation failed
            self._queue.extend(queue)

    async def shutdown(self) -> None:
        """Clean shutdown of the batcher"""
        async with self._attr_lock:
            if self._timer_task:
                self._timer_task.cancel()
                self._timer_task = None

            # Process any remaining requests in queue
            if self._queue:
                self.flush_req()

        # Wait for all inflight tasks to complete
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
            self._inflight.clear()

    def get_queue_size(self) -> int:
        """Get current queue size (for monitoring)"""
        return len(self._queue)

    def get_inflight_count(self) -> int:
        """Get number of inflight requests (for monitoring)"""
        return len(self._inflight)
