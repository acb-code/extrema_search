# Trust Region Bayesian Optimization (TuRBO)
# Author: Alex Braafladt
# Version: 2/10/23 v1
#
# References:
#   [1] [1] D. Eriksson, M. Pearce, J. R. Gardner, R. Turner, and M. Poloczek, “Scalable global optimization via local
#   Bayesian optimization,” Adv. Neural Inf. Process. Syst., vol. 32, no. NeurIPS, 2019. Available at:
#   https://github.com/uber-research/TuRBO (non-commercial license, but code not directly used either)
#   [2] https://botorch.org/tutorials/turbo_1 (MIT License)
#
# Notes:
#   -The initial approach is from [1], and code here adapts and modifies code from [2] which in turn adapts [1]
#   -The code here is simplified to 1-d, adds a local model for inside the trust region, and picks slightly different
#   parameters for the trust region update process

# imports
from dataclasses import dataclass
from typing import Any
import torch
import math
from botorch.generation import MaxPosteriorSampling
from botorch.models.gp_regression import ExactGP
from torch.quasirandom import SobolEngine
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf

# setup
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# objects
@dataclass
class NewTurboState:
    """Class to maintain the trust region for TuRBO"""
    dim: int
    batch_size: int
    center: torch.Tensor
    lb: torch.Tensor
    ub: torch.Tensor
    train_x: Any = None
    train_y: Any = None
    length: float = 0.3
    length_min: float = 0.25 ** 3
    length_max: float = 0.5  # modifying for [0,1] interval from 1.6 to 0.5
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # post initialized
    success_counter: int = 0
    success_tolerance: int = 4  # original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    domain_constraints: torch.Tensor = None

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))

    def get_training_samples_in_region(self):
        """query the training data and get the samples in the trust region - note this is for 1-d case"""
        # get the updated geometry of the trust region
        self.center = self.train_x[self.train_y.argmax(), :].clone()
        self.lb = torch.clamp(self.center - self.length / 2.0, 0.0, 1.0)
        self.ub = torch.clamp(self.center + self.length / 2.0, 0.0, 1.0)
        # get the indices of the evaluated data points in the trust region
        idx_below_ub = torch.where(self.train_x <= self.ub, True, False)
        idx_above_lb = torch.where(self.train_x >= self.lb, True, False)
        idx_in_tr = idx_below_ub & idx_above_lb
        # get the training points to use that are in the trust region
        train_x_tr = self.train_x[idx_in_tr].unsqueeze(-1)
        train_y_tr = self.train_y[idx_in_tr].unsqueeze(-1)
        # return the training points to use
        if len(train_x_tr) < 1:
            print("\nNot enough points in tr, using global data to fit model\n")
            return self.train_x.unsqueeze(-1), self.train_y.unsqueeze(-1)
            # depending on which {x, y} data are in current state, may need to revisit this
        else:
            return train_x_tr, train_y_tr


def new_update_state(state: NewTurboState, x_train, y_train, y_next):
    """Update the state of the trust region each iteration"""
    # check if the last iteration was successful and update attributes
    if torch.max(y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    # modify trust region geometry based on success or failure of last step
    if state.success_counter == state.success_tolerance:  # expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # shrink the trust region
        state.length /= 2.0
        state.failure_counter = 0
    # update the best value seen
    state.best_value = max(state.best_value, torch.max(y_next).item())
    # check if the trust region needs to restart
    if state.length < state.length_min:
        state.restart_triggered = True
        state.length = 0.5  # assumes x in [0, 1]
        print("\nTuRBO restart triggered")
    # update training data set
    state.train_x = x_train
    state.train_y = y_train
    return state


def turbo_region_bounds(model: ExactGP, x_center, length):
    """Get the bounds for the turbo trust region"""
    # scale the trust region to be proportional to the length scales
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * length / 2.0, 0.0, 1.0)
    return tr_lb, tr_ub


def turbo_thompson_sampling():
    """Convert candidates and trust region geometry to next sample point"""


def generate_batch(state: NewTurboState,  # trust region state
                   model: ExactGP,  # GP model
                   x,  # evaluated points on [0,1]
                   y,  # evaluated function values corresponding to x
                   batch_size,
                   n_candidates=None,
                   num_restarts=10,
                   raw_samples=512,
                   acqf='ts',  # 'ei' or 'ts'
                   ):
    """Acquisition function for TuRBO, wraps Thompson sampling or Expected Improvement constrained to
    trust region boundaries"""
    assert acqf in ("ts", "ei")
    assert x.min() >= 0.0 and x.max() <= 1.0 and torch.all(torch.isfinite(y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * x.shape[-1]))

    # # scale the trust region to be proportional to the length scales
    x_center = x[y.argmax(), :].clone()
    # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    # weights = weights / weights.mean()
    # # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    # tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    # tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    tr_lb, tr_ub = turbo_region_bounds(model, x_center, state.length)

    if acqf == 'ts':
        # thompson sampling
        dim = state.train_x.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        # create perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, 1, size=(len(ind),), device=device)] = 1
        # create candidate points from the perturbations and the mask
        x_cand = x_center.expand(n_candidates, dim).clone()
        x_cand[mask] = pert[mask]

        # sample the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            x_next = thompson_sampling(x_cand, num_samples=batch_size)

    elif acqf == 'ei':
        ei = qExpectedImprovement(model, y.max(), maximize=True)
        x_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return x_next


