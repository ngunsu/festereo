import torch
import time


class TorchTimer(object):

    """Compute elapsed time for a given method """

    def __init__(self, times=100, warmup=5):
        """ Init torch timer

        Parameters:
        ----------
        times: int
            How many times will be run the method
        warmup: int
            How many the method will run before recording its time
        """
        super().__init__()
        self.times = times
        self.warmup = warmup
        self.reset()

    def reset(self):
        """ Reset variables
        Returns
        -------
        None
        """
        self.elapsed = torch.zeros(self.times)

    def run(self, method, *args):
        """ Benchmark a given method with its arguments

        Parameters
        ----------
        method : Object
            Method to benchmark
        args: list
            List of arguments of the method

        Returns
        -------
        elapsed time : [mean, std, network_output] in seconds
        """
        self.reset()
        for i in range(self.times + self.warmup):
            torch.cuda.synchronize()
            start = time.perf_counter()
            net_output = method(*args)
            torch.cuda.synchronize()
            end = time.perf_counter()
            if i > (self.warmup - 1):
                self.elapsed[i - self.warmup] = end - start
        return self.elapsed.mean().cpu(), self.elapsed.std().cpu(), net_output
