"""Small process-local worker lifecycle; application data uses iceoryx2 only."""
import json
import signal
from threading import Event, current_thread, main_thread
import time


def log_record(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


class WorkerContext:
    def __init__(self, name: str):
        self.name = name
        self.stop_event = Event()
        self._handlers = {}

    def stop(self, *_args):
        self.stop_event.set()

    def install_signal_handlers(self):
        if current_thread() is main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._handlers[sig] = signal.signal(sig, self.stop)

    def restore_signal_handlers(self):
        for sig, handler in self._handlers.items():
            signal.signal(sig, handler)
        self._handlers.clear()


class SingleRateWorker:
    def __init__(self, context: WorkerContext, hz=None):
        if hz is not None and hz <= 0:
            raise ValueError("Worker rate must be positive")
        self.ctx, self.hz = context, hz

    def on_start(self):
        pass

    def step_once(self):
        raise NotImplementedError

    def on_stop(self):
        pass

    def run(self):
        self.ctx.install_signal_handlers()
        try:
            self.on_start()
            while not self.ctx.stop_event.is_set():
                started = time.monotonic()
                self.step_once()
                if self.hz is not None:
                    self.ctx.stop_event.wait(max(0.0, 1 / self.hz - (time.monotonic() - started)))
        finally:
            try:
                self.on_stop()
            finally:
                self.ctx.restore_signal_handlers()
