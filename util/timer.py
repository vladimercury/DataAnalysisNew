import time


class Timer:
    def __init__(self):
        self.last = time.time()

    def set_new(self):
        self.last = time.time()

    def get_diff(self):
        return time.time() - self.last

    def get_diff_str(self):
        diff = self.get_diff()
        if diff < 1e-3:
            return '%d μs' % (diff * 1e+6)
        if diff < 1:
            return '%d ms' % (diff * 1e+3)
        diff = int(diff)
        if diff < 60:
            return '%d s' % diff
        if diff < 3600:
            return '%d m %d s' % (diff / 60, diff % 60)
        diff_h = diff % 3600
        return '%d h %d m %d s' % (diff / 3600, diff_h / 60, diff % 60)
