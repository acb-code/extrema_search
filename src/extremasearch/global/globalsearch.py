# Global Search Algorithm - objects for global portion of multi-modal extrema search
# Author: Alex Braafladt
# Version: 2/11/23 v1
#
# References:
#   [1] https://botorch.org/tutorials/closed_loop_botorch_only
#
# Notes:
#   -Initial version
#   -Uses some support components from [1]


# imports
from typing import Callable
import torch
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from extremasearch.acquisition.tead import *
from typing import Callable
from dataclasses import dataclass
from extremasearch.local.localsearch import LocalSearchState
import networkx as nx


# setup
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_initial_data_with_function(obj_func, n=10):
    """generate initial set of data with a given objective function"""
    train_x = torch.rand(n, 1, device=device, dtype=dtype)
    exact_obj = obj_func(train_x)
    train_obj = exact_obj
    best_observed_value = exact_obj.max().item()
    return train_x, train_obj, best_observed_value


@dataclass
class GlobalSearchNode:
    """Nodes for tree of partitions with local searches at each node"""
    local_search_state: LocalSearchState = None
    preselect_score: torch.Tensor = None
    select_score: torch.Tensor = None


@dataclass
class GlobalSearchState:
    """State includes tree-based data structure for holding local search info nodes,
    and global data for the entire domain"""
    partition_graph: nx.DiGraph = None
    num_levels: int = None
    current_node_num: int = 0
    global_model: SingleTaskGP = None
    global_mll: ExactMarginalLogLikelihood = None
    x_global: torch.Tensor = None
    y_global: torch.Tensor = None
    x_global_extreme: torch.Tensor = None
    y_global_extreme: torch.Tensor = None
    x_global_candidates: torch.Tensor = None
    tead_global_scores: torch.Tensor = None
    converged_extrema: list = None








