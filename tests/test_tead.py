# tests for tead acquisition
import pytest

from extremasearch.acquisition.tead import finite_diff, knn, global_tead
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from scipy.stats.qmc import LatinHypercube

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def example_1d_model():
    dtype = torch.double
    train_X = torch.rand(20, 1, dtype=dtype)
    train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
    model_example = SingleTaskGP(train_X, train_Y)
    mll_example = ExactMarginalLogLikelihood(model_example.likelihood, model_example)
    fit_gpytorch_mll(mll_example)
    return model_example


def test_finite_diff(example_1d_model):
    assert finite_diff(example_1d_model).shape == torch.Size([20])
    assert finite_diff(example_1d_model).type() == 'torch.FloatTensor'


def test_knn(example_1d_model):
    x = example_1d_model.train_inputs[0].squeeze()
    lhs_sampler = LatinHypercube(d=1)
    num_cands = 2000
    cands = torch.from_numpy(lhs_sampler.random(n=num_cands))
    dists, inds = knn(x, cands)
    assert dists.shape == (2000, 1)
    assert dists.dtype == 'float64'
    assert inds.shape == (2000, 1)
    assert inds.dtype == 'int64'


def test_global_tead(example_1d_model):
    cands, scores = global_tead(example_1d_model, True)
    assert cands.shape == (2000, 1)
    assert cands.dtype == torch.double
    assert scores.shape == (2000, 1)
    assert scores.dtype == torch.double
    scores_bool = scores >= 0.0
    assert torch.all(scores_bool)
    cand_x = global_tead(example_1d_model, False)
    assert cand_x.shape == (1, 1)
    assert cand_x.dtype == torch.double



