# tests for turbo acquisition
import pytest

from extremasearch.acquisition.turbo import NewTurboState, generate_batch
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_turbo_init_setup():
    state = NewTurboState(dim=1, batch_size=1, center=torch.tensor([0.5], dtype=dtype),
                          lb=torch.tensor([0.0], dtype=dtype), ub=torch.tensor([1.0], dtype=dtype))
    assert state is not None
    assert state.failure_tolerance <= 4/state.batch_size or\
           state.failure_tolerance <= float(state.dim) / state.batch_size
    assert state.lb.dtype == torch.double
    assert state.ub.dtype == torch.double
    assert state.center.dtype == torch.double
    assert state.lb.shape == (1,)
    assert state.ub.shape == (1,)
    assert state.center.shape == (1,)


def test_get_training_samples_in_region():
    state = NewTurboState(dim=1, batch_size=1, center=torch.tensor([0.5], dtype=dtype),
                          lb=torch.tensor([0.0], dtype=dtype), ub=torch.tensor([1.0], dtype=dtype))
    state.train_x = torch.tensor([])









