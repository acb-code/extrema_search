# setting up n-dim, batch, Monte Carlo version of tead based on BoTorch acquisition function framework
#
# Author: Alex Braafladt
# Date: 3/4/2023
#
# TEAD references:
#   [1] S. Mo et al., “A Taylor Expansion-Based Adaptive Design Strategy for Global Surrogate Modeling With
#   Applications in Groundwater Modeling,” Water Resour. Res., vol. 53, no. 12, pp. 10802–10823, 2017,
#   doi: 10.1002/2017WR021622.
#   [2] J. N. Fuhg, A. Fau, and U. Nackenhorst, State-of-the-Art and Comparative Review of Adaptive Sampling Methods
#   for Kriging, vol. 28, no. 4. Springer Netherlands, 2021.
#   [3] https://github.com/FuhgJan/StateOfTheArtAdaptiveSampling (MIT License)
#
# BoTorch acquisition function references
#   [4] https://botorch.org/docs/acquisition
#   [5] M. Balandat et al., “BOTORCH: A framework for efficient Monte-Carlo Bayesian optimization,” Adv. Neural Inf.
#       Process. Syst., vol. 2020-Decem, no. MC, 2020.
#   [6] https://github.com/pytorch/botorch/discussions/1692
#   [7] https://botorch.org/tutorials/custom_acquisition
#
# Notes:
#   -The original algorithm is from [1], and reproduced in [2] and [3]
#   -The implementations here start from [3] (n-D) but make some simplifications and modifications
#   -The goal in this work is to translate the implementation from [3] to Python through the BoTorch acquisition
#    framework, including adding batch, Monte Carlo evaluation capability, so that the multi-start acquisition solvers
#    in the BoTorch framework can be used for fast evaluation of TEAD in n-dim
#   -Focusing here on SingleTaskGP from BoTorch as the model type, but using the more general Model type from BoTorch
#    that extends to additional probabilistic model options


# original tead imports
import torch
from botorch.models.gp_regression import ExactGP
from scipy.stats.qmc import LatinHypercube
from sklearn.neighbors import NearestNeighbors
import numpy as np

# imports for BoTorch MCAcquisitionFunction custom setup as in [7]
from typing import Optional

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor








