# tests for turbo acquisition
import pytest

from extremasearch.acquisition.turbo import NewTurboState, generate_batch, new_update_state
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# test overall state object
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


def test_state_add_data():
    state = NewTurboState(dim=1, batch_size=1, center=torch.tensor([0.3], dtype=dtype), length=0.1,
                          lb=torch.tensor([0.25], dtype=dtype), ub=torch.tensor([0.35], dtype=dtype))
    state.train_x = torch.tensor([0.2, 0.3, 0.4], dtype=dtype).unsqueeze(-1)
    state.train_y = torch.tensor([0.4, 0.8, 0.2], dtype=dtype).unsqueeze(-1)
    state.best_value = torch.tensor([0.8], dtype=dtype).unsqueeze(-1)
    assert state.train_x.shape == (3, 1)
    assert state.train_y.shape == (3, 1)
    assert state.best_value >= torch.max(state.train_y)
    assert state.best_value.shape == (1, 1)

@pytest.fixture
def simple_turbo_state():
    state = NewTurboState(dim=1, batch_size=1, center=torch.tensor([0.3], dtype=dtype), length=0.1,
                          lb=torch.tensor([0.25], dtype=dtype), ub=torch.tensor([0.35], dtype=dtype))
    state.train_x = torch.tensor([0.2, 0.3, 0.4], dtype=dtype).unsqueeze(-1)
    state.train_y = torch.tensor([0.4, 0.8, 0.2], dtype=dtype).unsqueeze(-1)
    state.best_value = torch.tensor([0.8], dtype=dtype).unsqueeze(-1)
    return state


def test_get_training_samples_in_region(simple_turbo_state):
    x_in, y_in = simple_turbo_state.get_training_samples_in_region()
    assert x_in.shape == (1, 1)
    assert y_in.shape == (1, 1)
    assert x_in == torch.tensor([0.3], dtype=dtype).unsqueeze(-1)
    assert y_in == torch.tensor([0.8], dtype=dtype).unsqueeze(-1)
    assert x_in.dtype == torch.double
    assert y_in.dtype == torch.double


# test state update function
def test_update_state_single_success(simple_turbo_state):
    # tr success
    assert simple_turbo_state.success_counter == 0
    assert simple_turbo_state.best_value == torch.tensor(0.8, dtype=dtype)
    updated_x = torch.tensor([0.2, 0.3, 0.4, 0.28], dtype=dtype).unsqueeze(-1)
    updated_y = torch.tensor([0.4, 0.8, 0.2, 0.9], dtype=dtype).unsqueeze(-1)
    new_state = new_update_state(simple_turbo_state, updated_x, updated_y, torch.max(updated_y))
    assert new_state.success_counter == 1
    assert simple_turbo_state.success_counter == 1  # access and changes happen by reference
    assert new_state.best_value == torch.tensor(0.9, dtype=dtype)


def test_update_state_single_failure(simple_turbo_state):
    # tr failure
    assert simple_turbo_state.failure_counter == 0
    assert simple_turbo_state.best_value == torch.tensor(0.8, dtype=dtype)
    updated_x = torch.tensor([0.2, 0.3, 0.4, 0.28], dtype=dtype).unsqueeze(-1)
    updated_y = torch.tensor([0.4, 0.8, 0.2, 0.1], dtype=dtype).unsqueeze(-1)
    new_state = new_update_state(simple_turbo_state, updated_x, updated_y, torch.max(updated_y))
    assert new_state.failure_counter == 1
    assert simple_turbo_state.failure_counter == 1  # access and changes happen by reference
    assert new_state.best_value == torch.tensor(0.8, dtype=dtype)  # note that this is scalar (1,)


def test_update_state_success_counter(simple_turbo_state):
    # modify state to make trust region ready to expand
    simple_turbo_state.success_counter = simple_turbo_state.success_tolerance - 1
    original_length = simple_turbo_state.length
    old_lb = simple_turbo_state.lb
    old_ub = simple_turbo_state.ub
    updated_x = torch.tensor([0.2, 0.3, 0.4, 0.28], dtype=dtype).unsqueeze(-1)
    updated_y = torch.tensor([0.4, 0.8, 0.2, 0.9], dtype=dtype).unsqueeze(-1)
    new_state = new_update_state(simple_turbo_state, updated_x, updated_y, torch.max(updated_y))
    assert new_state.length == 2.0*original_length
    # note that lb and ub are static
    assert new_state.lb == old_lb
    assert new_state.ub == old_ub


def test_update_state_failure_counter(simple_turbo_state):
    # modify state to make trust region ready to expand
    simple_turbo_state.failure_counter = simple_turbo_state.failure_tolerance - 1
    original_length = simple_turbo_state.length
    old_lb = simple_turbo_state.lb
    old_ub = simple_turbo_state.ub
    updated_x = torch.tensor([0.2, 0.3, 0.4, 0.28], dtype=dtype).unsqueeze(-1)
    updated_y = torch.tensor([0.4, 0.8, 0.2, 0.1], dtype=dtype).unsqueeze(-1)
    new_state = new_update_state(simple_turbo_state, updated_x, updated_y, torch.max(updated_y))
    assert new_state.length == original_length/2.0
    assert new_state.failure_counter == 0
    # note that lb and ub are static
    assert new_state.lb == old_lb
    assert new_state.ub == old_ub


def test_turbo_restart(simple_turbo_state):
    # modify state to make ready to restart
    simple_turbo_state.length = simple_turbo_state.length_min + 1e-4
    simple_turbo_state.failure_counter = simple_turbo_state.failure_tolerance - 1
    original_length = simple_turbo_state.length
    updated_x = torch.tensor([0.2, 0.3, 0.4, 0.28], dtype=dtype).unsqueeze(-1)
    updated_y = torch.tensor([0.4, 0.8, 0.2, 0.1], dtype=dtype).unsqueeze(-1)
    new_state = new_update_state(simple_turbo_state, updated_x, updated_y, torch.max(updated_y))
    assert new_state.restart_triggered is True
    assert new_state.length == 0.5


# test acquisition function




