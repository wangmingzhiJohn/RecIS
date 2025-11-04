import queue
import threading
import time
from typing import Iterator

from torch.utils.data import IterableDataset


class _StopFlag:
    def __init__(self):
        self._stop = False

    def stop(self):
        return self._stop

    def set_stop(self, flag: bool):
        self._stop = flag


class _DataPrefetcher(threading.Thread):
    def __init__(self, queue: queue.Queue, iterator: Iterator, stop_flag: _StopFlag):
        super().__init__()
        self._queue = queue
        self.daemon = True
        self._iterator = iterator
        self._stop_flag = stop_flag

    def run(self):
        try:
            for input_data in self._iterator:
                put_done = False
                while not put_done:
                    if self._stop_flag.stop():
                        return
                    try:
                        self._queue.put(input_data, timeout=60)
                        put_done = True
                    except queue.Full:
                        pass
        except StopIteration:
            pass
        put_done = False
        while not put_done:
            if self._stop_flag.stop():
                return
            try:
                self._queue.put(None, timeout=60)
                put_done = True
            except queue.Full:
                pass
        while not self._stop_flag.stop():
            time.sleep(1)


class _PrefetchIterator:
    def __init__(self, dataset, input_iterator: Iterator, buffer_size=1):
        self._dataset = dataset
        self._input_iterator = input_iterator
        self._queue = queue.Queue(maxsize=buffer_size)
        self._stop_flag = _StopFlag()
        self._runner = _DataPrefetcher(
            self._queue, self._input_iterator, self._stop_flag
        )
        self._started = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self._started:
            self._runner.start()
            self._started = True
        if self._stop_flag.stop():
            raise StopIteration
        ret = self._queue.get()
        self._queue.task_done()
        if ret is None:
            self._stop_flag.set_stop(True)
            raise StopIteration
        return ret

    def __del__(self):
        self._stop_flag.set_stop(True)
        if self._started:
            self._runner.join()


class PrefetchDataset(IterableDataset):
    def __init__(self, input_dataset: IterableDataset, buffer_size=1):
        self._input_dataset = input_dataset
        self._buffer_size = buffer_size

    def __iter__(self) -> Iterator:
        return _PrefetchIterator(self, iter(self._input_dataset), self._buffer_size)
