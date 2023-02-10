# tests for tead acquisition
import pytest

from extremasearch.acquisition.tead import finite_diff
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def example_1d_model():
    train_X = torch.rand(20, 1)
    train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
    model_example = SingleTaskGP(train_X, train_Y)
    mll_example = ExactMarginalLogLikelihood(model_example.likelihood, model_example)
    fit_gpytorch_mll(mll_example)
    return model_example


def test_finite_diff(example_1d_model):
    assert finite_diff(example_1d_model).shape == torch.Size([20])
    assert finite_diff(example_1d_model).type() == 'torch.FloatTensor'