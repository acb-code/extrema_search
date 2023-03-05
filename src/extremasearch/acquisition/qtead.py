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


class qTaylorExpansionBasedAdaptiveDesign(MCAcquisitionFunction):
    def __init__(
            self,
            model: Model,
            sampler: Optional[MCSampler] = None,
    ) -> None:
        """Using the basic setup from [7] - global model"""
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the qTEAD on the candidate set `X`.
        `b` - t-batch number/dimension
        `q` - q-batch number/dimension
        `d` - input dimension (of objective function)
        `o` - output dimension (of objective function)

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Taylor Expansion-Based Adaptive Design
            values at the given design points `X`={x1, x2, ..., xq}
        """
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)  # n x b x q x o
        mean = posterior.mean  # b x q x o
        # TEAD calculations
        # finite diff gradient estimate at {x, y} data points - could be external, doesn't change as x_c change
        # generate candidate points x_c via LHS (2000-20000) - could be external? or is it captured in t-batches in opt?
        # find KNN for candidates x_c compared to {x, y} data points
        # todo figure out what this should look like


###############
# simplified n-D case
# setup
dtype = torch.double


# objects
def nfinite_diff(model):
    """get approx gradient at model training points"""
    # assumes 1d input... see implementation in [3] for update to n-D
    h = 1e-4
    x = model.train_inputs[0].squeeze()
    m = len(x)
    delta_s = torch.zeros(m)
    for i in range(0, m):
        x_lo = x[i] - h
        if x_lo < 0.0:
            x_lo = torch.DoubleTensor([0.0])
        x_hi = x[i] + h
        if x_hi > 1.0:
            x_hi = torch.DoubleTensor([1.0])
        y_lo = model(x_lo.unsqueeze(-1)).mean
        y_hi = model(x_hi.unsqueeze(-1)).mean
        delta_s[i] = (y_hi - y_lo)/(2*h)
    return delta_s


def nknn(ref, query, k=1):
    """Function to wrap KNN from sklearn for TEAD setup
    assumes 1d input dimension, torch Tensors, ref is training inputs
    query is candidate points, k is number of neighbors
    """
    knn_ex = NearestNeighbors(n_neighbors=k).fit(ref.squeeze().numpy().reshape(-1, 1))
    dists, indices = knn_ex.kneighbors(query.squeeze().numpy().reshape(-1, 1))
    return dists, indices


def nglobal_tead(model: ExactGP, get_all_candidates: bool = False):
    """Function version of initial implementation TEAD adaptive sampling algorithm
        Returns candidates and scores, not just final argmax score"""
    # get training set from model - assuming format expected from ExactGP here
    x_train = model.train_inputs[0].squeeze()
    y_train = model.train_targets
    # calculate gradient at each training point using finite difference
    grads = nfinite_diff(model)
    # (this may be available through torch built in behavior, but no time now
    # generate potential input candidates using LHC samples
    lhs_sampler = LatinHypercube(d=1)
    num_cands = 2000
    cands = torch.from_numpy(lhs_sampler.random(n=num_cands))
    # calculate nearest neighbors pairings for the candidates
    dists, inds = nknn(x_train, cands)
    # calculate the weighting
    sample_dists = []
    vals = x_train.numpy()
    for i in range(len(vals)):
        for j in range(len(vals)):
            sample_dists.append(np.linalg.norm(vals[i]-vals[j]))
    lmax = max(sample_dists)
    w = torch.ones(num_cands, 1, dtype=dtype) - dists/lmax
    # do taylor series expansions
    t = torch.zeros(num_cands, 1, dtype=dtype)
    s = torch.zeros(num_cands, 1, dtype=dtype)
    res = torch.zeros(num_cands, 1, dtype=dtype)
    for i in range(num_cands):
        g = x_train[inds[i]] - cands[i]
        t[i, 0] = y_train[inds[i]] + grads[inds[i]]*g
        s[i, 0] = model(cands[i]).mean
        res[i, 0] = torch.norm(s[i, 0] - t[i, 0])

    # compute score
    j = torch.from_numpy(dists/max(dists)) + w * (res / max(res))
    # pick point with max score
    if get_all_candidates:
        return cands, j
    else:
        return cands[torch.argmax(j)].unsqueeze(-1)






