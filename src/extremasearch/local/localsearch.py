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
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from extremasearch.acquisition.turbo import *
from extremasearch.acquisition.tead import *
from typing import Callable
from botorch.sampling import SobolQMCNormalSampler
from botorch.models.transforms import Normalize, Standardize

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


@dataclass
class LocalSearchState:
    """Holds the local search state information to be updated by running the search"""
    input_dim: int
    local_bounds: torch.Tensor
    x_local: torch.Tensor
    y_local: torch.Tensor
    trust_region: NewTurboState = None
    x_local_extreme: torch.Tensor = None
    y_local_extreme: torch.Tensor = None
    local_model: SingleTaskGP = None
    local_mll: ExactMarginalLogLikelihood = None
    most_recent_x_local: torch.Tensor = None
    most_recent_y_local: torch.Tensor = None


def initialize_model(train_x, train_obj, state_dict=None):
    """function to initialize the GP model"""
    model_obj = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    # load state_dict if it is not passed
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj


def initialize_scaled_model(train_x, train_obj, state_dict=None):
    """function to initialize the GP model with scaling on outputs and normalization on inputs"""
    model_obj = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]),
                             input_transform=Normalize(d=train_x.shape[-1]))
    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    # load state_dict if it is not passed
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj


@dataclass
class SearchIterationData:
    """Holds data on each search iteration to simplify adding and accessing the data"""
    x: torch.Tensor = None
    y: torch.Tensor = None
    acq_type: str = None
    iter_num: int = None


@dataclass
class LocalExtremeSearch:
    """Runs the local search to update the local search state"""
    max_local_evals: int
    min_local_init_evals: int
    local_state: LocalSearchState
    objective_function: Callable = None
    length_history: torch.Tensor = None
    tr_bound_history: torch.Tensor = None
    search_history: list[SearchIterationData] = None
    iteration_tracker: int = None

    def initialize_local_search(self):
        """Set up the local search object"""
        # set up state object
        # check initial data
        x_initial = self.local_state.x_local
        self.search_history = []
        if self.iteration_tracker is None:
            self.iteration_tracker = 0
        # if not enough initial data points in local subdomain, sample more randomly todo: sobol or lhs, ndim
        if x_initial.shape[0] < self.min_local_init_evals:
            # sample randomly in subdomain to get to min
            num_new = self.min_local_init_evals - x_initial.shape[0]
            print('Making ', num_new, ' mcs evaluations to start local search')
            new_x = torch.rand(num_new, 1, device=device, dtype=dtype)
            local_x = new_x * (self.local_state.local_bounds[1] - self.local_state.local_bounds[0])  # todo: ndim
            local_x = local_x + self.local_state.local_bounds[0]
            new_y = self.objective_function(local_x)
            # update local data set
            self.local_state.x_local = torch.cat((self.local_state.x_local, local_x), 0)
            self.local_state.y_local = torch.cat((self.local_state.y_local, new_y), 0)
            # update iteration data tracking
            self.iteration_tracker += local_x.shape[0]
            for i in range(0, self.iteration_tracker):
                current_data = SearchIterationData(x=local_x, y=new_y, acq_type='mcs', iter_num=self.iteration_tracker)
                self.search_history.append(current_data)
            # store new evaluations to add to global data set later
            self.local_state.most_recent_x_local = local_x
            self.local_state.most_recent_y_local = new_y
        else:
            # make sure new evaluations data reset
            self.local_state.most_recent_x_local = None
            self.local_state.most_recent_y_local = None
        # initialize gp model
        gp_x = self.local_state.x_local
        gp_y = self.local_state.y_local
        # self.local_state.local_mll, self.local_state.local_model = initialize_model(gp_x, gp_y, None)
        self.local_state.local_mll, self.local_state.local_model = initialize_scaled_model(gp_x, gp_y, None)

    def run_local_search(self, acq_type: str = 'turbo'):
        """Run the search based on the local starting point"""
        # initialize local search
        self.initialize_local_search()
        # local search loop
        converged = False
        local_iter = 0
        # initialize turbo trust region
        # self.local_state.trust_region = NewTurboState(dim=1, batch_size=1, center=0.5, lb=0.0, ub=1.0)
        # trying setting the bounds to constrain the turbo search to the selected subdomain
        self.local_state.trust_region = NewTurboState(dim=1,
                                                      batch_size=1,
                                                      center=torch.tensor([0.5], dtype=dtype),
                                                      lb=self.local_state.local_bounds[0],
                                                      ub=self.local_state.local_bounds[1],
                                                      length=0.25*(self.local_state.local_bounds[1] -
                                                                   self.local_state.local_bounds[0]),
                                                      domain_constraints=self.local_state.local_bounds,
                                                      length_max=0.5*(self.local_state.local_bounds[1] -
                                                                      self.local_state.local_bounds[0])
                                                      )
        self.local_state.trust_region = new_update_state(self.local_state.trust_region,
                                                         self.local_state.x_local,
                                                         self.local_state.y_local,
                                                         max(self.local_state.y_local)
                                                         )
        while not converged and local_iter <= self.max_local_evals:
            # fit the gp model
            self.fit_local_model()
            # run the acquisition function
            if acq_type == 'turbo':
                next_x = generate_batch(state=self.local_state.trust_region,
                                        model=self.local_state.local_model,
                                        x=self.local_state.x_local,
                                        y=self.local_state.y_local,
                                        batch_size=1,
                                        n_candidates=N_CANDIDATES,
                                        num_restarts=NUM_RESTARTS,
                                        raw_samples=RAW_SAMPLES,
                                        acqf='ts',
                                        )
                print('turbo')
            elif acq_type == 'tead':
                next_x = global_tead(self.local_state.local_model)
                print('tead')
            elif acq_type == 'nei':
                qmc_sampler = SobolQMCNormalSampler(MC_SAMPLES)
                qNEI = qNoisyExpectedImprovement(model=self.local_state.local_model,
                                                 X_baseline=self.local_state.x_local,
                                                 sampler=qmc_sampler,
                                                 )
                next_x = optimize_acqf_and_get_next_x(qNEI)
            else:
                print('Error: unknown acquisition function for local search')
            next_y = self.objective_function(next_x)
            # update x,y data points
            # local x,y values
            self.local_state.x_local = torch.cat((self.local_state.x_local, next_x), 0)
            self.local_state.y_local = torch.cat((self.local_state.y_local, next_y), 0)
            # search history
            current_data = SearchIterationData(x=next_x, y=next_y, acq_type=acq_type, iter_num=self.iteration_tracker)
            self.search_history.append(current_data)
            self.iteration_tracker += 1
            # new x,y values from this local search only
            if self.local_state.most_recent_x_local is not None:
                self.local_state.most_recent_x_local = torch.cat((self.local_state.most_recent_x_local, next_x), 0)
                self.local_state.most_recent_y_local = torch.cat((self.local_state.most_recent_y_local, next_y), 0)
            else:
                self.local_state.most_recent_x_local = next_x
                self.local_state.most_recent_y_local = next_y
            # update trust region
            # print(max(self.local_state.y_local))
            self.local_state.trust_region = new_update_state(self.local_state.trust_region,
                                                             x_train=self.local_state.x_local,
                                                             y_train=self.local_state.y_local,
                                                             y_next=max(self.local_state.y_local)
                                                             )
            # update local model - using only local training samples inside the updated trust region
            x_in_region, y_in_region = self.local_state.trust_region.get_training_samples_in_region()
            if acq_type == 'turbo':
                self.local_state.local_mll, self.local_state.local_model = initialize_scaled_model(x_in_region,
                                                                                                   y_in_region, None)
            elif acq_type == 'tead':
                self.local_state.local_mll, self.local_state.local_model = initialize_scaled_model(self.local_state.x_local,
                                                                                            self.local_state.y_local,
                                                                                            None)
            else:
                self.local_state.local_mll, self.local_state.local_model = initialize_scaled_model(self.local_state.x_local,
                                                                                            self.local_state.y_local,
                                                                                            None)
                print('Error: unknown acquisition function for local search')
            local_iter += 1
            if self.local_state.trust_region.restart_triggered:
                converged = True
        # exit local search and return results
        local_search_extreme_y, extreme_idx = torch.max(input=self.local_state.y_local, dim=0, keepdim=True)
        local_search_extreme_x = self.local_state.x_local[extreme_idx[0]]
        self.local_state.x_local_extreme = local_search_extreme_x
        self.local_state.y_local_extreme = local_search_extreme_y
        return self.search_history

    def fit_local_model(self):
        """Fit the local model"""
        # update local model including scaling and normalization from data in current trust region
        # get the data in the current trust region
        # set up the model and mll using the normalization and scaling

        fit_gpytorch_mll(self.local_state.local_mll)

