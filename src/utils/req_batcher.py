import asyncio
import time
import uuid
from collections.abc import Coroutine
from typing import Any, Callable, Dict, NamedTuple


class Request(NamedTuple):
    id: str
    prompt: str
    model_name: str
    sampling_params: Dict[Any, Any]


class Batcher:
    def __init__(
        self,
        model_name: str,
        max_req: int,
        max_time: int,
        process_req_fn: Callable[Any, Coroutine[Any, Any, int]],
    ):
        self.model_name = model_name
        self.max_req = max_req
        self.max_time = max_time
        self.process_req_fn = process_req_fn
        self.busy = False  # Indicates if an inference is going on right now

        self._loop = asyncio.get_running_loop()
        self._queue = []
        self._attr_lock = asyncio.Lock()
        self._start_time = None
        self._timer_task = None
        self._inflight = set()

    # Add request to the queue
    async def add(self, req: Request) -> None:
        async with self._attr_lock:
            self._queue.append((req, time.monotonic()))
            print(f"batcher add() - len(queue): {len(self._queue())}")
            if len(self._queue) == 1:
                self._start_time = time.monotonic()

            # If no inference going on
            if not self.busy:
                # First in queue, start timer
                if len(self._queue) == 1:
                    self._timer_task = asyncio.create_task(self._timer_coro())

                # Queue is now filled
                elif len(self._queue) >= self.max_req:
                    self.flush_req()

                # Otherwise, just wait for timer or more requests

            # If busy, just adds and exits

    # Clear the busy status - Will flush out another batch if batch is full OR oldest req has waited long enough
    # Have to be super careful! This function is called from a different thread
    async def busy_clear(self):
        async with self._attr_lock:
            self.busy = False
            # Send out batch if queue can fill out batch
            if len(self._queue) >= self.max_req:
                self.flush_req()
            # If there is queue but not full, check if time is up (will instantly flush if time is up)
            elif len(self._queue):
                self._timer_task = asyncio.create_task(self._timer_coro())
            # If no queue, do nothing other than set busy flag to False

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
        except:
            raise

    # This function is only called when lock is held
    def flush_req(self) -> None:
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        # Take batch size out of queue and remove from queue
        queue = self._queue[: self.max_req].copy()
        del self._queue[: self.max_req]

        # Update start time with head of queue if any
        if len(self._queue):
            self._start_time = self._queue[0][1]
        else:
            self._start_time = None

        try:
            task = asyncio.create_task(
                self.process_req_fn(
                    batch_id=str(uuid.uuid4()),
                    model_name=self.model_name,
                    queue=[item[0] for item in queue],
                )
            )
            self._inflight.add(task)

            def _done(t: asyncio.Task) -> None:
                self._inflight.discard(t)
                # Check if task had an exception
                if t.exception():
                    print(
                        f"Warning: Task in batcher failed with exception: {t.exception()}"
                    )

            task.add_done_callback(_done)

            # Update busy status
            self.busy = True

        except Exception as e:
            print(f"Error creating task in batcher: {e}")

            import traceback

            traceback.print_exc()

            # Re-queue the requests if task creation failed
            self._queue.extend(queue)

    async def shutdown(self) -> None:
        """Clean shutdown of the batcher"""
        async with self._attr_lock:
            if self._timer_task:
                self._timer_task.cancel()
                self._timer_task = None

            # Continuously batch remaining requests and create one task each
            while self._queue:
                with self._attr_lock:
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
