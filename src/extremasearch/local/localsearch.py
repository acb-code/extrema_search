# Local Search Algorithm - objects for local portion of multi-modal extrema search
# Author: Alex Braafladt
# Version: 2/11/23 v1
#
# References:
#   [1] https://botorch.org/tutorials/closed_loop_botorch_only
#
# Notes:
#   -Initial version, uses turbo or nei as local search inside a specific domain starting point
#   -Uses some support components from [1]


# imports
import torch
from dataclasses import dataclass
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf


# setup
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_SAMPLES = 256
N_CANDIDATES = min(5000, max(2000, 200 * 1))  # changes if dim =! 1
RAW_SAMPLES = 512
NUM_RESTARTS = 10
BATCH_SIZE = 1  # changes if batch size > 1 used
bounds = torch.tensor([0.0, 1.0], device=device, dtype=dtype).unsqueeze(-1)  # changes if input bounds not 0-1


def optimize_acqf_and_get_next_x(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # initialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    return new_x

