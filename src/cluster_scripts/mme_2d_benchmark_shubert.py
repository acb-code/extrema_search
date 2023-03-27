#%%
# script for literature 2d canonical problem for multimodal optimization looking for localized extreme
# events with agent-based military simulation
# Experiment 2.3 - Stochastic, jump discontinuity multimodal localized extrema test problem with 2d input dimension
# Part 1 - benchmark approaches (MCS, QMC, NEI, TEAD, TuRBO)
#
# Author: Alex Braafladt
# Initial creation: 3/23/23
#
# Goal: Benchmark state-of-the-art Bayesian Optimization approaches (and QMC/MCS) on a multimodal
#       optimization test function from the literature closest to my case - shubert [5]
#
# Notes:
# -Using the BoTorch framework [1] and NEI from [4]
# -TEAD technique from [2]
# -TuRBO technique from [3]
#
# References:
# [1] M. Balandat et al., “BOTORCH: A framework for efficient Monte-Carlo Bayesian optimization,”
#     Adv. Neural Inf. Process. Syst., vol. 2020-Decem, no. MC, 2020.
# [2] S. Mo et al., “A Taylor Expansion-Based Adaptive Design Strategy for Global Surrogate
#     Modeling With Applications in Groundwater Modeling,” Water Resour. Res., vol. 53, no.
#     12, pp. 10802–10823, 2017, doi: 10.1002/2017WR021622.
# [3] D. Eriksson, M. Pearce, J. R. Gardner, R. Turner, and M. Poloczek, “Scalable global
#     optimization via local Bayesian optimization,” Adv. Neural Inf. Process. Syst., vol.
#     32, no. NeurIPS, 2019.
# [4] B. Letham, B. Karrer, G. Ottoni, and E. Bakshy, “Constrained Bayesian optimization with noisy
#     experiments,” Bayesian Anal., vol. 14, no. 2, pp. 495–519, 2019, doi: 10.1214/18-BA1110.
# [5]
#%%
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import networkx as nx
from botorch.models.transforms import Normalize, Standardize


# setup
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
#%%
# # setup file i/o
# import os as os
# import datetime as dt
# # get current working directory
# wrkdir = os.getcwd()
# print('Current working directory: '+wrkdir)
# # set up a data save directory for all future runs
# newoutputdir = wrkdir+'\output'
# if not os.path.exists(newoutputdir):
#     os.makedirs(newoutputdir)
# # set up a new directory to store files for the current run - updates at each new full run of notebook
# curDatetime = dt.datetime.now()
# datasavedir = newoutputdir + r'\\' + '2.3_mme_2d_benchmark' + str(curDatetime.strftime('%Y%m%d%H%M%S'))
# if not os.path.exists(datasavedir):
#     os.makedirs(datasavedir)
# print('Data save directory: '+datasavedir)

import datetime as dt
curDatetime = dt.datetime.now()
strtime = str(curDatetime.strftime('%Y%m%d%H%M%S'))


# %%
# visualize canonical problem
# Shubert - dim*3^dim global optima, many local optima, many smooth but localized optima (~760)
# x_i in [-10, 10]
def shubert_plot_2d(x1, x2):
    x = [(x1 * 20.0) - 10., (x2 * 20.0) - 10.]
    outer = torch.ones_like(x1)
    for i in range(2):
        inner = torch.zeros_like(x1)
        for j in range(5):
            inner += j*torch.cos( (j+1)*x[i] + j)
        outer = outer * inner
    return outer / 186.73


from matplotlib import cm
x1 = torch.arange(0., 1., 0.005)
x2 = torch.arange(0., 1., 0.005)
X1, X2 = torch.meshgrid([x1, x2])
Z = shubert_plot_2d(X1, X2)
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(8,6))
# surf = ax.plot_surface(X1, X2, Z, cmap=cm.inferno, shade=False)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(8,6))

from matplotlib.colors import LightSource
ls = LightSource(270, 45)
rgb = ls.shade(Z.numpy(), cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(X1, X2, Z, cmap='Blues', shade=False, cstride=1, rstride=1, linewidth=0)

# %%
# set up function to work with botorch environment
def shubert_2d(X):
    x1, x2 = X[..., 0], X[..., 1]
    x = [(x1 * 20.0) - 10., (x2 * 20.0) - 10.]
    outer = torch.ones_like(x1)
    for i in range(2):
        inner = torch.zeros_like(x1)
        for j in range(5):
            inner += j*torch.cos( (j+1)*x[i] + j)
        outer = outer * inner
    return outer / 186.73


def outcome_objective(x):
    """wrapper for the outcome objective function"""
    return shubert_2d(x).type_as(x).unsqueeze(-1)


#%%
# set up the GP models for use
def initialize_model(train_x, train_obj, state_dict=None):
    """function to initialize the GP model"""
    model_obj = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    # load state_dict if it is not passed
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj

# set up the initial data set
def generate_initial_data(n=10):
    """generate initial set of data to get started with BO loop"""
    train_x = torch.rand(10, 2, device=device, dtype=dtype)
    exact_obj = outcome_objective(train_x)
    train_obj = exact_obj
    best_observed_value = exact_obj.max().item()
    return train_x, train_obj, best_observed_value

# set up the acquisition function operations for botorch
BATCH_SIZE = 1  # only doing one parallel objective function evaluation
from botorch.optim import optimize_acqf
NUM_RESTARTS = 10
RAW_SAMPLES = 512
bounds = torch.tensor([[0.0, 1.0],[0.0, 1.0]], device=device, dtype=dtype)
def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,# initialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    new_obj = outcome_objective(new_x)
    return new_x, new_obj

def update_random_observation():
    """Simulates a random policy by taking the current list of best values observed
    randomly, drawing a new random point, observing its value, and updating the list
    """
    rand_x = torch.rand(BATCH_SIZE, 2)  # drawing from random uniform
    next_random_best = outcome_objective(rand_x)
    return  rand_x, next_random_best
#%%
# metrics functions
def count_number_peaks_observed_2d(x_obs, y_obs, num_known_peaks=3):
    """Function to count the number of peaks observed for the mme_noise_jump_1d function"""
    peak1, peak2, peak3 = False, False, False
    for x, y in zip(x_obs, y_obs):
        if 0.64 <= x[0] <= 0.66 and 0.64 <= x[1] <= 0.66:
            if y >= 1.37:
                peak1 = True
        elif 0.24 <= x[0] <= 0.26 and 0.34 <= x[1] <= 0.36:
            if y >= 1.06:
                peak2 = True
        elif 0.79 <= x[0] <= 0.81 and 0.14 <= x[1] <= 0.16:
            if y >= 1.36:
                peak3 = True
    num_peaks_observed = peak1 + peak2 + peak3
    return num_peaks_observed


def count_evaluations_for_all_peaks_2d(x_obs, y_obs):
    """Count the number of function evaluations before finding all peaks"""
    i = 0
    num_evals_for_all_peaks = x_obs.shape[0]
    peak1, peak2, peak3 = False, False, False
    for x, y in zip(x_obs, y_obs):
        if 0.64 <= x[0] <= 0.66 and 0.64 <= x[1] <= 0.66:
            if y >= 1.37:
                peak1 = True
        elif 0.24 <= x[0] <= 0.26 and 0.34 <= x[1] <= 0.36:
            if y >= 1.06:
                peak2 = True
        elif 0.79 <= x[0] <= 0.81 and 0.14 <= x[1] <= 0.16:
            if y >= 1.36:
                peak3 = True
        i += 1
        if peak1 and peak2 and peak3:
            num_evals_for_all_peaks = i
            break
    return num_evals_for_all_peaks
#%%
# set up Bayesian optimization loop
from extremasearch.acquisition.nturbo import ngenerate_batch, NdTurboState, nd_new_update_state
from extremasearch.acquisition.qtead import nglobal_tead
import time
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from extremasearch.local.localsearch import initialize_scaled_model

N_TRIALS = 4
N_BATCH = 390 # note +10 for total iteration limit
MC_SAMPLES = 256
dim = 2
N_CANDIDATES = min(5000, max(2000, 200 * dim))
BATCH_SIZE = 1

verbose = True
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# set up arrays to hold arrays of data
# best values
best_observed_all_nei, best_observed_random_all, best_observed_all_tead = [], [], []
best_observed_all_turbo = []
# {x, y} values
x_observed_all_random, x_observed_all_nei, x_observed_all_tead = [], [], []
y_observed_all_random, y_observed_all_nei, y_observed_all_tead = [], [], []
x_observed_all_turbo = []
y_observed_all_turbo = []

# mmo results
distinct_peaks_random, distinct_peaks_nei, distinct_peaks_tead = [], [], []
function_evals_random, function_evals_nei, function_evals_tead = [], [], []
distinct_peaks_turbo = []
function_evals_turbo = []

# going to average over N_TRIALS
for trial in range(1, N_TRIALS + 1):
    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    # set up arrays to hold data for this trial
    best_observed_nei, best_observed_random, best_observed_tead = [], [], []
    best_observed_turbo = []

    # generate initial training data and initial model for this trial
    # nei
    # use same initial training points
    train_x_nei, train_obj_nei, best_observed_value_nei = generate_initial_data(n=10)
    mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei)
    # tead
    # use same initial points
    train_x_tead, train_obj_tead = train_x_nei, train_obj_nei
    best_observed_value_tead = best_observed_value_nei
    mll_tead, model_tead = initialize_model(train_x_tead, train_obj_tead)
    # turbo
    train_x_turbo, train_obj_turbo = train_x_nei, train_obj_nei
    best_observed_value_turbo = best_observed_value_nei
    # turbo initialization
    state = NdTurboState(dim=2, batch_size=BATCH_SIZE, center=torch.tensor(0.5), lb=torch.tensor(0.0), ub=torch.tensor(1.0))
    state = nd_new_update_state(state, train_x_turbo, train_obj_turbo, torch.DoubleTensor([best_observed_value_turbo]))
    tr_x_turbo, tr_obj_turbo = state.get_training_samples_in_region()
    mll_turbo, model_turbo = initialize_scaled_model(tr_x_turbo, tr_obj_turbo)
    # random
    train_x_random, train_obj_random = train_x_nei, train_obj_nei
    best_observed_value_random = best_observed_value_nei

    # start collection of results from random initialization
    best_observed_nei.append(best_observed_value_nei)
    best_observed_random.append(best_observed_value_nei)
    best_observed_tead.append(best_observed_value_tead)
    best_observed_turbo.append(best_observed_value_turbo)
    best_observed_random.append(best_observed_value_random)

    # for N_BATCH rounds of BO after the initial random batch
    for iteration in range(1, N_BATCH + 1):
        t0 = time.monotonic()
        # fit models
        fit_gpytorch_mll(mll_nei)
        fit_gpytorch_mll(mll_tead)
        fit_gpytorch_mll(mll_turbo)

        # set up the sampler to use with the acq funcs
        qmc_sampler = SobolQMCNormalSampler(sample_shape=MC_SAMPLES)

        # set up the acquisition functions
        qNEI = qNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=train_x_nei,
            sampler=qmc_sampler,
        )

        # optimize the acquisition functions
        new_x_nei, new_obj_nei = optimize_acqf_and_get_observation(qNEI)
        new_x_tead = nglobal_tead(model_tead)
        new_obj_tead = outcome_objective(new_x_tead)
        new_x_turbo = ngenerate_batch(
            state=state,
            model=model_turbo,
            dim=state.train_x.shape[-1],
            x=state.train_x,
            y=state.train_y,
            batch_size=BATCH_SIZE,
            n_candidates=N_CANDIDATES,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            acqf='ts',
        )
        new_obj_turbo = outcome_objective(new_x_turbo)
        # random
        new_x_random, new_obj_random = update_random_observation()

        # update training points
        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
        train_x_tead = torch.cat([train_x_tead, new_x_tead])
        train_obj_tead = torch.cat([train_obj_tead, new_obj_tead])
        train_x_turbo = torch.cat([train_x_turbo, new_x_turbo])
        train_obj_turbo = torch.cat([train_obj_turbo, new_obj_turbo])
        train_x_random = torch.cat([train_x_random, new_x_random])
        train_obj_random = torch.cat([train_obj_random, new_obj_random])

        # update trust region
        state = nd_new_update_state(state, x_train=train_x_turbo, y_train=train_obj_turbo, y_next=new_obj_turbo)

        # update progress data
        best_value_random = train_obj_random.max()
        best_observed_random.append(best_value_random)
        best_value_nei = train_obj_nei.max()
        best_observed_nei.append(best_value_nei)
        best_value_tead = train_obj_tead.max()
        best_observed_tead.append(best_value_tead)
        best_value_turbo = train_obj_turbo.max()
        best_observed_turbo.append(best_value_turbo)

        # reinitialize the models so that they're ready to fit on the next iteration
        # state dict passed to speed up fitting
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            model_nei.state_dict(),
        )
        mll_tead, model_tead = initialize_model(
            train_x_tead,
            train_obj_tead,
            model_tead.state_dict(),
        )
        # update trust region model
        tr_x_turbo, tr_obj_turbo = state.get_training_samples_in_region()
        mll_turbo, model_turbo = initialize_scaled_model(tr_x_turbo, tr_obj_turbo)

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (random, qNEI, TEAD, TuRBO) = "
                f"({best_value_random:>4.2f}, {best_value_nei:>4.2f}, {best_value_tead:>4.2f}, {state.best_value:>4.2f}), "
                f"time = {t1-t0:>4.2f}.", end=""
            )
        else:
            print(".", end="")

    # collect observations from trial
    best_observed_all_nei.append(best_observed_nei)
    best_observed_random_all.append(best_observed_random)
    best_observed_all_tead.append(best_observed_tead)
    best_observed_all_turbo.append(best_observed_turbo)
    x_observed_all_random.append(train_x_random)
    x_observed_all_nei.append(train_x_nei)
    x_observed_all_tead.append(train_x_tead)
    x_observed_all_turbo.append(train_x_turbo)
    y_observed_all_random.append(train_obj_random)
    y_observed_all_nei.append(train_obj_nei)
    y_observed_all_tead.append(train_obj_tead)
    y_observed_all_turbo.append(train_obj_turbo)
    # collect metrics
    distinct_peaks_random.append(count_number_peaks_observed_2d(train_x_random, train_obj_random))
    distinct_peaks_nei.append(count_number_peaks_observed_2d(train_x_nei, train_obj_nei))
    distinct_peaks_tead.append(count_number_peaks_observed_2d(train_x_tead, train_obj_tead))
    distinct_peaks_turbo.append(count_number_peaks_observed_2d(train_x_turbo, train_obj_turbo))
    function_evals_random.append(count_evaluations_for_all_peaks_2d(train_x_random, train_obj_random))
    function_evals_nei.append(count_evaluations_for_all_peaks_2d(train_x_nei, train_obj_nei))
    function_evals_tead.append(count_evaluations_for_all_peaks_2d(train_x_tead, train_obj_tead))
    function_evals_turbo.append(count_evaluations_for_all_peaks_2d(train_x_turbo, train_obj_turbo))
#%%
# abbreviated QMC loop with Sobol sequences
from torch.quasirandom import SobolEngine

best_observed_all_sobol = []
x_observed_all_sobol, y_observed_all_sobol = [], []
distinct_peaks_sobol = []
function_evals_sobol = []
for trial in range(1, N_TRIALS + 1):
    print(f"\nSobol Trial {trial:>2} of {N_TRIALS} ", end="")
    best_observed_sobol = []
    # get sobol random samples for the trial
    X_Sobol = SobolEngine(dimension=2, scramble=True).draw(N_BATCH+10).to(dtype=dtype, device=device)
    Y_Sobol = torch.tensor([outcome_objective(x) for x in X_Sobol], dtype=dtype, device=device)
    curMax = torch.max(Y_Sobol[:10])
    for y_val in Y_Sobol[10:]:
        curMax = torch.max(curMax,y_val)
        best_observed_sobol.append(curMax.squeeze())
    best_observed_all_sobol.append(best_observed_sobol)
    x_observed_all_sobol.append(X_Sobol)
    y_observed_all_sobol.append(Y_Sobol)
    distinct_peaks_sobol.append(count_number_peaks_observed_2d(X_Sobol, Y_Sobol))
    function_evals_sobol.append(count_evaluations_for_all_peaks_2d(X_Sobol, Y_Sobol))
#%%
# collect output data
# calculate summary metrics
# peak ratio
num_known_peaks = 3
peak_ratio_random = sum(distinct_peaks_random) / (num_known_peaks * N_TRIALS)
peak_ratio_nei = sum(distinct_peaks_nei) / (num_known_peaks * N_TRIALS)
peak_ratio_tead = sum(distinct_peaks_tead) / (num_known_peaks * N_TRIALS)
peak_ratio_turbo = sum(distinct_peaks_turbo) / (num_known_peaks * N_TRIALS)
peak_ratio_sobol = sum(distinct_peaks_sobol) / (num_known_peaks * N_TRIALS)
print(f"Peak ratio: Random {peak_ratio_random:>4.2f}, NEI {peak_ratio_nei:>4.2f}, TEAD {peak_ratio_tead:>4.2f}, TuRBO {peak_ratio_turbo:>4.2f}, QMC {peak_ratio_sobol:>4.2f} ")
# success rate
num_successes_random = 0
num_successes_nei = 0
num_successes_tead = 0
num_successes_turbo = 0
num_successes_sobol = 0
for i in range(N_TRIALS):
    if distinct_peaks_random[i] == 3:
        num_successes_random += 1
    if distinct_peaks_nei[i] == 3:
        num_successes_nei += 1
    if distinct_peaks_tead[i] == 3:
        num_successes_tead += 1
    if distinct_peaks_turbo[i] == 3:
        num_successes_turbo += 1
    if distinct_peaks_sobol[i] == 3:
        num_successes_sobol += 1
success_ratio_random = num_successes_random / N_TRIALS
success_ratio_nei = num_successes_nei / N_TRIALS
success_ratio_tead = num_successes_tead / N_TRIALS
success_ratio_turbo = num_successes_turbo / N_TRIALS
success_ratio_sobol = num_successes_sobol / N_TRIALS
print(f"Success ratio: Random {success_ratio_random:>4.2f}, NEI {success_ratio_nei:>4.2f}, TEAD {success_ratio_tead:>4.2f}, TuRBO {success_ratio_turbo:>4.2f}, Sobol {success_ratio_sobol:>4.2f}")
# function evaluations
fe_random = sum(function_evals_random)/N_TRIALS
fe_nei = sum(function_evals_nei)/N_TRIALS
fe_tead = sum(function_evals_tead)/N_TRIALS
fe_turbo = sum(function_evals_turbo)/N_TRIALS
fe_sobol = sum(function_evals_sobol)/N_TRIALS
print(f"Average function evaluations: Random {fe_random}, NEI {fe_nei}, TEAD {fe_tead}, TuRBO {fe_turbo}, Sobol {fe_sobol}")
#%%
# save data to file for potential future use
methods_list = ['MCS', 'NEI', 'TEAD', 'TuRBO', 'QMC']
pr_list = [peak_ratio_random, peak_ratio_nei, peak_ratio_tead, peak_ratio_turbo, peak_ratio_sobol]
peak_list = [distinct_peaks_random, distinct_peaks_nei, distinct_peaks_tead, distinct_peaks_turbo, distinct_peaks_sobol]
success_list = [num_successes_random, num_successes_nei, num_successes_tead, num_successes_turbo, num_successes_sobol]
num_reps = N_TRIALS
sr_list = [success_ratio_random, success_ratio_nei, success_ratio_tead, success_ratio_turbo, success_ratio_sobol]
fe_list = [function_evals_random, function_evals_nei, function_evals_tead, function_evals_turbo, function_evals_sobol]

torch.save([methods_list, pr_list, peak_list, success_list, num_reps, sr_list, fe_list], '2d_bench_mme_metrics_shubert'+strtime+'.pt')

#%%
fit_gpytorch_mll(mll_nei)
fit_gpytorch_mll(mll_tead)
fit_gpytorch_mll(mll_turbo)
None
#%%
# plot evaluations for tead - surface
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z = shubert_plot_2d(X1, X2)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10,8))
surf = ax.plot_surface(X1, X2, Z, cmap='Blues', shade=False, cstride=1, rstride=1, linewidth=0, alpha=0.2)
ax.scatter(train_x_tead[:,0].numpy()[0:10], train_x_tead[:,1].numpy()[0:10], train_obj_tead.numpy()[0:10], marker='.', color='tab:orange', s=40)
ax.scatter(train_x_tead[:,0].numpy()[10:], train_x_tead[:,1].numpy()[10:], train_obj_tead.numpy()[10:], marker='.', color='g', s=40)
# plt.savefig(datasavedir + '/'+'2d_mme_noise_tead_evals'+'.png')
plt.savefig(strtime+'2d_mme_noise_tead_evals'+'.png')
#%%
# plot evaluations for tead contour
fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, Z, levels=15, cmap='Blues')
cbar = fig1.colorbar(CS)

ax1.scatter(train_x_tead[:,0].numpy()[0:10], train_x_tead[:,1].numpy()[0:10], s=15, marker='o', color='tab:orange', zorder=2)
ax1.scatter(train_x_tead[:,0].numpy()[10:], train_x_tead[:,1].numpy()[10:], s=15, marker='o', color='g', zorder=2)
ax1.set_xlim([-0.01,1.01])
ax1.set_ylim([-0.01,1.01])
# plt.savefig(datasavedir + '/'+'2d_mme_noise_tead_evals_contour'+'.png')
plt.savefig(strtime+'2d_mme_noise_tead_evals_contour'+'.png')
#%%
# plot model contours tead - mean
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z_tead_model = model_tead.posterior(torch.stack([X1.flatten(), X2.flatten()], dim=1)).mean
Z_tead_shaped = Z_tead_model.unflatten(dim=0, sizes=(100,100)).squeeze()

fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, Z_tead_shaped.detach().numpy(), levels=15, cmap='viridis')
cbar = fig1.colorbar(CS)
# plt.savefig(datasavedir + '/'+'2d_mme_noise_tead_model_mean_contour'+'.png')
plt.savefig(strtime+'2d_mme_noise_tead_model_mean_contour'+'.png')
#%%
# plot model contours tead - standard deviation
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z_tead_model = model_tead.posterior(torch.stack([X1.flatten(), X2.flatten()], dim=1)).variance
Z_tead_shaped = Z_tead_model.unflatten(dim=0, sizes=(100,100)).squeeze()

fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, np.sqrt(Z_tead_shaped.detach().numpy()), levels=15, cmap='viridis')
cbar = fig1.colorbar(CS)
# plt.savefig(datasavedir + '/'+'2d_mme_noise_tead_model_stdev_contour'+'.png')
plt.savefig(strtime+'2d_mme_noise_tead_model_stdev_contour'+'.png')
#%%
# plot evaluations for nei
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z = shubert_plot_2d(X1, X2)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10,8))
surf = ax.plot_surface(X1, X2, Z, cmap='Blues', shade=False, cstride=1, rstride=1, linewidth=0, alpha=0.2)
ax.scatter(train_x_nei[:,0].numpy(), train_x_nei[:,1].numpy(), train_obj_nei.numpy(), marker='.', color='g')
# plt.savefig(datasavedir + '/'+'2d_mme_noise_nei_evals_surface'+'.png')
plt.savefig(strtime+'2d_mme_noise_nei_evals_surface'+'.png')
#%%
# plot evaluations for nei contour
fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, Z, levels=15, cmap='Blues')
cbar = fig1.colorbar(CS)
ax1.scatter(train_x_nei[:,0].numpy()[0:10], train_x_nei[:,1].numpy()[0:10], s=15, marker='o', color='tab:orange', zorder=2)
ax1.scatter(train_x_nei[:,0].numpy()[10:], train_x_nei[:,1].numpy()[10:], s=15, marker='o', color='g', zorder=2)
ax1.set_xlim([-0.01,1.01])
ax1.set_ylim([-0.01,1.01])
# plt.savefig(datasavedir + '/'+'2d_mme_noise_nei_evals_contour'+'.png')
plt.savefig(strtime+'2d_mme_noise_nei_evals_contour'+'.png')
#%%
# plot model contours nei - mean
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z_nei_model = model_nei.posterior(torch.stack([X1.flatten(), X2.flatten()], dim=1)).mean
Z_nei_shaped = Z_nei_model.unflatten(dim=0, sizes=(100,100)).squeeze()

fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, Z_nei_shaped.detach().numpy(), levels=15, cmap='viridis')
cbar = fig1.colorbar(CS)
plt.savefig(strtime+'2d_mme_noise_nei_model_mean_contour'+'.png')
# plt.savefig(datasavedir + '/'+'2d_mme_noise_nei_model_mean_contour'+'.png')
#%%
# plot model contours nei - stdev
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z_nei_model = model_nei.posterior(torch.stack([X1.flatten(), X2.flatten()], dim=1)).variance
Z_nei_shaped = Z_nei_model.unflatten(dim=0, sizes=(100,100)).squeeze()

fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, np.sqrt(Z_nei_shaped.detach().numpy()), levels=15, cmap='viridis')
cbar = fig1.colorbar(CS)
# plt.savefig(datasavedir + '/'+'2d_mme_noise_nei_model_stdev_contour'+'.png')
plt.savefig(strtime+'2d_mme_noise_nei_model_stdev_contour'+'.png')
#%%
# plot evaluations for turbo - surface
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z = shubert_plot_2d(X1, X2)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10,8))
surf = ax.plot_surface(X1, X2, Z, cmap='Blues', shade=False, cstride=1, rstride=1, linewidth=0, alpha=0.2)
ax.scatter(train_x_turbo[:,0].numpy()[0:10], train_x_turbo[:,1].numpy()[0:10], train_obj_turbo.numpy()[0:10], marker='.', color='tab:orange', s=40)
ax.scatter(train_x_turbo[:,0].numpy()[10:], train_x_turbo[:,1].numpy()[10:], train_obj_turbo.numpy()[10:], marker='.', color='g', s=40)
# plt.savefig(datasavedir + '/'+'2d_mme_noise_turbo_evals_surface'+'.png')
plt.savefig(strtime+'2d_mme_noise_turbo_evals_surface'+'.png')
#%%
# plot evaluations for tead contour
fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, Z, levels=15, cmap='Blues')
cbar = fig1.colorbar(CS)

ax1.scatter(train_x_turbo[:,0].numpy()[0:10], train_x_turbo[:,1].numpy()[0:10], s=15, marker='o', color='tab:orange', zorder=2)
ax1.scatter(train_x_turbo[:,0].numpy()[10:], train_x_turbo[:,1].numpy()[10:], s=15, marker='o', color='g', zorder=2)
ax1.set_xlim([-0.01,1.01])
ax1.set_ylim([-0.01,1.01])
# plt.savefig(datasavedir + '/'+'2d_mme_noise_turbo_evals_contour'+'.png')
plt.savefig(strtime+'2d_mme_noise_turbo_evals_contour'+'.png')
#%%
# plot model contours turbo - mean
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z_tead_model = model_turbo.posterior(torch.stack([X1.flatten(), X2.flatten()], dim=1)).mean
Z_tead_shaped = Z_tead_model.unflatten(dim=0, sizes=(100,100)).squeeze()

fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, Z_tead_shaped.detach().numpy(), levels=15, cmap='viridis')
cbar = fig1.colorbar(CS)
# plt.savefig(datasavedir + '/'+'2d_mme_noise_turbo_model_mean_contour'+'.png')
plt.savefig(strtime+'2d_mme_noise_turbo_model_mean_contour'+'.png')
#%%
# plot model contours turbo - standard deviation
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z_tead_model = model_turbo.posterior(torch.stack([X1.flatten(), X2.flatten()], dim=1)).variance
Z_tead_shaped = Z_tead_model.unflatten(dim=0, sizes=(100,100)).squeeze()

fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, np.sqrt(Z_tead_shaped.detach().numpy()), levels=15, cmap='viridis')
cbar = fig1.colorbar(CS)
plt.savefig(strtime+'2d_mme_noise_turbo_model_stdev_contour'+'.png')
#%%
# plot evaluations for sobol - surface
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z = shubert_plot_2d(X1, X2)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10,8))
surf = ax.plot_surface(X1, X2, Z, cmap='Blues', shade=False, cstride=1, rstride=1, linewidth=0, alpha=0.2)
ax.scatter(X_Sobol[:,0].numpy(), X_Sobol[:,1].numpy(), Y_Sobol.numpy(), marker='.', color='g', s=40)
plt.savefig(strtime+'2d_mme_noise_qmc_sobol_evals_surface'+'.png')
#%%
# plot evaluations for sobol contour
fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, Z, levels=15, cmap='Blues')
cbar = fig1.colorbar(CS)

ax1.scatter(X_Sobol[:,0].numpy(), X_Sobol[:,1].numpy(), s=15, marker='o', color='g', zorder=2)
ax1.set_xlim([-0.01,1.01])
ax1.set_ylim([-0.01,1.01])
plt.savefig(strtime+'2d_mme_noise_qmc_evals_contour'+'.png')
#%%
# plot evaluations for mcs - surface
x1 = torch.arange(0.0, 1.0, 0.01)
x2 = torch.arange(0.0, 1.0, 0.01)
X1, X2 = torch.meshgrid([x1, x2])
Z = shubert_plot_2d(X1, X2)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10,8))
surf = ax.plot_surface(X1, X2, Z, cmap='Blues', shade=False, cstride=1, rstride=1, linewidth=0, alpha=0.2)
ax.scatter(train_x_random[:,0].numpy(), train_x_random[:,1].numpy(), train_obj_random.numpy(), marker='.', color='g', s=40)
plt.savefig(strtime+'2d_mme_noise_mcs_evals_surface'+'.png')
#%%
# plot evaluations for mcs contour
fig1, ax1 = plt.subplots()
CS = ax1.contour(X1, X2, Z, levels=15, cmap='Blues')
cbar = fig1.colorbar(CS)
ax1.scatter(train_x_random[:,0].numpy(), train_x_random[:,1].numpy(), s=15, marker='o', color='g', zorder=2)
ax1.set_xlim([-0.01,1.01])
ax1.set_ylim([-0.01,1.01])
plt.savefig(strtime+'2d_mme_noise_mcs_evals_contour'+'.png')
