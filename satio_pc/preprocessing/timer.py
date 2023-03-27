import time
from loguru import logger


class TaskTimer:

    def __init__(self, task_name, unit='minutes'):
        self.name = task_name
        self._total = 0

        self._running = False
        self._start = None

        self._unit = unit

        self._scaling = {'hours': 3600,
                         'minutes': 60,
                         'seconds': 1}.get(unit)

        if self._scaling is None:
            raise ValueError(f'Unknown unit `{unit}`')

    @property
    def _now(self):
        return time.time()

    def start(self):
        if self._running:
            pass
            # raise TaskTimerRunningError(
            #     'TaskTimer is running, should be stopped before starting')
        self._start = self._now
        self._running = True

    def stop(self):
        if not self._running:
            pass
            # raise TaskTimerRunningError(
            #     'TaskTimer isn't running, should be started before stopping')
        self._total += self._now - self._start
        self._running = False

    @property
    def _msg(self):
        return 'running for' if self._running else 'took'

    @property
    def total(self):
        if self._running:
            total = (self._now - self._start) / self._scaling
        else:
            total = self._total / self._scaling

        return total

    def log(self, level='INFO'):
        logger.log(level,
                   f"{self.name} {self._msg} {self.total:.2f} {self._unit}")

    def reset(self):
        self._start = None
        self._running = False
        self._total = 0


class FeaturesTimer():

    def __init__(self, resolution):

        self.load = TaskTimer(f'{resolution}m loading')
        self.composite = TaskTimer(f'{resolution}m compositing')
        self.speckle = TaskTimer(f'{resolution}m speckle filtering')
        self.interpolate = TaskTimer(f'{resolution}m interpolation')
        self.features = TaskTimer(f'{resolution}m features computation')
