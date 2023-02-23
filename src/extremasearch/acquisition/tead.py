# Taylor Expansion-Based Adaptive Design (TEAD) Acquisition Function for Bayesian Optimization
# Author: Alex Braafladt
# Version: 2/6/23 v1
#
# References:
#   [1] S. Mo et al., “A Taylor Expansion-Based Adaptive Design Strategy for Global Surrogate Modeling With
#   Applications in Groundwater Modeling,” Water Resour. Res., vol. 53, no. 12, pp. 10802–10823, 2017,
#   doi: 10.1002/2017WR021622.
#   [2] J. N. Fuhg, A. Fau, and U. Nackenhorst, State-of-the-Art and Comparative Review of Adaptive Sampling Methods
#   for Kriging, vol. 28, no. 4. Springer Netherlands, 2021.
#   [3] https://github.com/FuhgJan/StateOfTheArtAdaptiveSampling (MIT License)
#
# Notes:
#   -The original algorithm is from [1], and reproduced in [2] and [3]
#   -The implementations here start from [3] (n-D) but make some simplifications and modifications
#   -There are two implementations for v1 (both are 1D only): deterministic_tead returns only the new x to sample at
#    while global_tead returns all candidates and scores computed


# imports
import torch
from botorch.models.gp_regression import ExactGP
from scipy.stats.qmc import LatinHypercube
from sklearn.neighbors import NearestNeighbors
import numpy as np


# setup
dtype = torch.double


# objects
def finite_diff(model):
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


def knn(ref, query, k=1):
    """Function to wrap KNN from sklearn for TEAD setup
    assumes 1d input dimension, torch Tensors, ref is training inputs
    query is candidate points, k is number of neighbors
    """
    knn_ex = NearestNeighbors(n_neighbors=k).fit(ref.squeeze().numpy().reshape(-1, 1))
    dists, indices = knn_ex.kneighbors(query.squeeze().numpy().reshape(-1, 1))
    return dists, indices


def global_tead(model: ExactGP, get_all_candidates: bool = False):
    """Function version of initial implementation TEAD adaptive sampling algorithm
        Returns candidates and scores, not just final argmax score"""
    # get training set from model - assuming format expected from ExactGP here
    x_train = model.train_inputs[0].squeeze()
    y_train = model.train_targets
    # calculate gradient at each training point using finite difference
    grads = finite_diff(model)
    # (this may be available through torch built in behavior, but no time now
    # generate potential input candidates using LHC samples
    lhs_sampler = LatinHypercube(d=1)
    num_cands = 2000
    cands = torch.from_numpy(lhs_sampler.random(n=num_cands))
    # calculate nearest neighbors pairings for the candidates
    dists, inds = knn(x_train, cands)
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


def get_model_from_piecewise_set(graph, x):
    """Retrieves the local model from the piecewise set at the input point x"""
    graph_leaves = [n for n in graph if graph.out_degree[n] == 0]
    # bound_list = []
    # node_list = []
    for n in graph_leaves:
        # get current leaf node
        current_node = graph.nodes()[n]
        current_state = current_node['data']
        current_bounds = current_state.local_bounds
        if current_bounds[0] <= x < current_bounds[1]:
            return current_state.local_model
        # also save the bounds-node map to use for edge cases
        # bound_list.append(current_bounds)
        # node_list.append(n)
        if current_bounds[0] <= 0.001:
            low_node = n
        elif current_bounds[1] >= 0.999:
            high_node = n
    # handle slightly off bound queries
    if x < 0.0:
        # if a prediction lower than 0.0 needed
        return graph.nodes()[low_node]['data'].local_model
    elif x >= 1.0:
        # if a prediction higher than 1.0 needed
        return graph.nodes()[high_node]['data'].local_model
    print("Model lookup failed")


def finite_diff_piecewise(x_all, graph):
    """get approx gradient at model training points"""
    # assumes 1d input... see implementation in [3] for update to n-D
    h = 1e-4
    x = x_all.squeeze()
    m = len(x)
    delta_s = torch.zeros(m)
    for i in range(0, m):
        x_lo = x[i] - h
        if x_lo < 0.0:
            x_lo = torch.DoubleTensor([0.0])
        x_hi = x[i] + h
        if x_hi > 1.0:
            x_hi = torch.DoubleTensor([1.0])
        model_lo = get_model_from_piecewise_set(graph, x_lo.unsqueeze(-1))
        y_lo = model_lo(x_lo.unsqueeze(-1)).mean
        model_hi = get_model_from_piecewise_set(graph, x_hi.unsqueeze(-1))
        y_hi = model_hi(x_hi.unsqueeze(-1)).mean
        delta_s[i] = (y_hi - y_lo)/(2*h)
    return delta_s


def piecewise_tead(x_all, y_all, graph, get_all_candidates: bool = False):
    """Modification of global_tead to use the piecewise models from a graph"""
    # use {x, y} evaluations across space as in global_tead
    x_train = x_all.squeeze()
    y_train = y_all
    # calculate gradient at each training point using finite difference
    grads = finite_diff_piecewise(x_train, graph)
    # (this may be available through torch built in behavior, but no time now
    # generate potential input candidates using LHC samples
    lhs_sampler = LatinHypercube(d=1)
    num_cands = 2000
    cands = torch.from_numpy(lhs_sampler.random(n=num_cands))
    # calculate nearest neighbors pairings for the candidates
    dists, inds = knn(x_train, cands)
    # calculate the weighting
    sample_dists = []
    vals = x_train.numpy()
    for i in range(len(vals)):
        for j in range(len(vals)):
            sample_dists.append(np.linalg.norm(vals[i] - vals[j]))
    lmax = max(sample_dists)
    w = torch.ones(num_cands, 1, dtype=dtype) - dists / lmax
    # do taylor series expansions
    t = torch.zeros(num_cands, 1, dtype=dtype)
    s = torch.zeros(num_cands, 1, dtype=dtype)
    res = torch.zeros(num_cands, 1, dtype=dtype)
    for i in range(num_cands):
        g = x_train[inds[i]] - cands[i]
        t[i, 0] = y_train[inds[i]] + grads[inds[i]]*g
        cur_model = get_model_from_piecewise_set(graph, cands[i])
        s[i, 0] = cur_model(cands[i]).mean
        res[i, 0] = torch.norm(s[i, 0] - t[i, 0])

    # compute score
    j = torch.from_numpy(dists/max(dists)) + w * (res / max(res))
    # pick point with max score
    if get_all_candidates:
        return cands, j
    else:
        return cands[torch.argmax(j)].unsqueeze(-1)


