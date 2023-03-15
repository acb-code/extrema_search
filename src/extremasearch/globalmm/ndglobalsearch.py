# Global Search Algorithm - objects for global portion of multi-modal extrema search
# Update for n-dimensional case
# Author: Alex Braafladt
# Version: 2/11/23 v1
#          3/15/23 v2 n-dim input modification
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
from extremasearch.acquisition.qtead import *
from typing import Callable
from dataclasses import dataclass
from extremasearch.local.ndlocalsearch import NdLocalSearchState, NdLocalExtremeSearch, SearchIterationData
import networkx as nx
from botorch.models.transforms import Normalize, Standardize


# setup
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")









