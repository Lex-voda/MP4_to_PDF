import time

class ProgressBar:
    def __init__(self):
        self.total = 0
        self.num_iters = 0
        self.progress = 0
        self.prefix = ''
        self.start_time = time.time()

    def update(self, progress, iter):
        self.progress += progress
        self.num_iters += iter
        self.display()

    def display(self):
        progress = self.progress / self.total
        num_blocks = int(progress * 40)

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time == 0:
            return

        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        print('\r', end='', flush=True)
        print(f"{self.prefix}: [{'#' * num_blocks}{' ' * (40 - num_blocks)}] {self.progress}:{self.total} {progress:.2%} frame:{self.num_iters} Elapsed: {elapsed_time_str}", end='', flush=True)