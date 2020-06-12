from datetime import datetime
import numpy as np

class Timer(object):
    """Record training durations"""
    def __init__(self):
        self.running_times = []

    def start(self):
        self.starting_time = datetime.now()

    def stop(self):
        self.running_times.append(datetime.now() - self.starting_time)

    def sum(self):
        return np.sum(self.running_times)

    def avg(self):
        return np.average(self.running_times)

    def cumsum(self):
        return np.cumsum(self.running_times)