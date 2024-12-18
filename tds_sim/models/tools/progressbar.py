import math
import sys

def _progressbar(status, total, scale=20):
    cnt = math.ceil(status / total * scale)
    return f"[{''.join(['='] * cnt)}{''.join([' '] * (scale - cnt))}]"

class ProgressBar(object):
    def __init__(self, total: int, desc: str='', scale: int=20, percentage: bool=True, print_interval: int=1):
        self.total = total
        self.desc = desc
        self.scale = scale

        self.percentage = percentage
        self.print_interval = print_interval

        self.status = 0

    def update(self, amt: int=1):
        self.status += amt

    def percent_fmt(self):
        return f" [{self.status/self.total*100:.2f}%]"

    def show(self):
        if self.status % self.print_interval != 0:
            return

        pbar = '\r' + self.desc + ' ' + _progressbar(status=self.status, total=self.total, scale=self.scale)
        if self.percentage:
            pbar += self.percent_fmt()
        if self.status == self.total:
            pbar += '\n'
        sys.stdout.write(pbar)