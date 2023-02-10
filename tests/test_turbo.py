# tests for turbo acquisition
import pytest

from extremasearch.acquisition.turbo import NewTurboState, generate_batch
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")