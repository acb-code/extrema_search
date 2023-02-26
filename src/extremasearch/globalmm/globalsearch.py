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
from extremasearch.local.localsearch import LocalSearchState, LocalExtremeSearch, initialize_model
import networkx as nx
from botorch.models.transforms import Normalize, Standardize


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


@dataclass
class MultimodalExtremaSearch:
    num_batch: int
    x_bounds: torch.Tensor
    obj_func: Callable = None
    input_dim: int = 1
    max_local_evals: int = 10
    min_local_init_evals: int = 2
    num_penalty_life: int = 3
    split_dim_selector: str = 'random'
    global_state: GlobalSearchState = None
    iteration_num: int = 0
    num_initial_samples: int = 10
    current_max_preselect_node: int = None
    total_iteration_limit: int = 100
    current_iteration: int = 0
    penalty_radius: torch.Tensor = torch.tensor([0.5], dtype=dtype)
    reset_penalty_radius: float = 0.25
    num_previous_extrema: int = 4
    penalty_reset_size: float = 0.025
    tead_model_type: str = 'piecewise'
    num_tead_explore_evals: int = 5

    def initialize_global_state(self, x_init, y_init):
        """Set up the initial GlobalSearchState object"""
        # start global state object using initial {x,y}
        self.global_state = GlobalSearchState(x_global=x_init, y_global=y_init)
        # set up graph tracking subdomains of search space
        self.global_state.partition_graph = nx.DiGraph()
        initial_local_data = LocalSearchState(input_dim=1, local_bounds=self.x_bounds,
                                              x_local=self.global_state.x_global,
                                              y_local=self.global_state.y_global,)
        initial_local_search = LocalExtremeSearch(self.max_local_evals, self.min_local_init_evals,
                                                  initial_local_data, self.obj_func)
        self.global_state.partition_graph.add_node(0, data=initial_local_data, search=initial_local_search)
        self.global_state.num_levels = 1
        # run initial local search
        # self.global_state.partition_graph.nodes[0]['search'].run_local_search()  # use turbo
        self.global_state.partition_graph.nodes[0]['search'].run_local_search(acq_type='tead')  # attempt with tead
        # update global search state
        x_new = self.global_state.partition_graph.nodes[0]['search'].local_state.most_recent_x_local
        y_new = self.global_state.partition_graph.nodes[0]['search'].local_state.most_recent_y_local
        self.global_state.x_global = torch.cat((self.global_state.x_global, x_new), 0)
        self.global_state.y_global = torch.cat((self.global_state.y_global, y_new), 0)
        # fit initial model
        self.global_state.global_mll, self.global_state.global_model = initialize_model(self.global_state.x_global,
                                                                                        self.global_state.y_global,
                                                                                        None)
        self.fit_global_model()
        # initialize tensors for extrema
        init_extreme_y, extreme_idx = torch.max(input=self.global_state.y_global, dim=0, keepdim=True)
        init_extreme_x = self.global_state.x_global[extreme_idx[0]]
        # current global extreme
        self.global_state.x_global_extreme = init_extreme_x
        self.global_state.y_global_extreme = init_extreme_y
        # converged extrema tracker
        self.global_state.converged_extrema = [(init_extreme_x, init_extreme_y)]

    def initialize_global_search(self):
        """Set up the initial state of the global search"""
        # collect initial samples
        print('Collecting initial samples')
        x_init, y_init, best_y = generate_initial_data_with_function(self.obj_func, n=self.num_initial_samples)
        # set up global state
        print('Setting up global state')
        self.initialize_global_state(x_init, y_init)

    def run_global_tead_exploration_evaluations(self):
        """Evaluate the best TEAD points for a handful of iterations prior to allocating a local search
        with the goal of using TEAD uniformity and nonlinearity foci to uncover potential localize extrema"""
        tead_iter = 0
        self.fit_global_model()
        self.fit_global_model_partitioned()
        while tead_iter < self.num_tead_explore_evals:
            # update model fit and compute TEAD acquisition function on candidates from across input space
            # choose between global model using all data, or piecewise partitioned data model joint
            if self.tead_model_type == 'global':
                x_candidates, x_tead_scores = global_tead(self.global_state.global_model, get_all_candidates=True)
            elif self.tead_model_type == 'piecewise':
                x_candidates, x_tead_scores = piecewise_tead(self.global_state.x_global, self.global_state.y_global,
                                                             self.global_state.partition_graph, get_all_candidates=True)
            else:
                print('TEAD type selection not recognized')
            # apply penalty
            current_penalty = self.get_penalty(x_candidates, x_tead_scores)
            x_penalized_scores = x_tead_scores * current_penalty
            # evaluate objective at best penalized TEAD point
            score_max, max_idx = torch.max(input=x_penalized_scores, dim=0, keepdim=True)
            x_max = x_candidates[max_idx[0]]
            y_new = self.obj_func(x_max)
            # update global state using new objective evaluation
            self.current_iteration += 1
            self.global_state.x_global = torch.cat((self.global_state.x_global, x_max), 0)
            self.global_state.y_global = torch.cat((self.global_state.y_global, y_new), 0)
            self.update_extreme()
            self.fit_global_model()
            self.fit_global_model_partitioned()
            tead_iter += 1
            print("Completed: TEAD exploration iteration")

    def compute_global_preselect_scores(self):
        """Use global value algorithm to assign value to points across input space"""
        # compute TEAD acquisition function on candidates from across input space
        # choose between global model using all data, or piecewise partitioned data model joint
        if self.tead_model_type == 'global':
            x_candidates, x_tead_scores = global_tead(self.global_state.global_model, get_all_candidates=True)
        elif self.tead_model_type == 'piecewise':
            x_candidates, x_tead_scores = piecewise_tead(self.global_state.x_global, self.global_state.y_global,
                                                         self.global_state.partition_graph, get_all_candidates=True)
        else:
            print('TEAD type selection not recognized')
        self.global_state.x_global_candidates = x_candidates
        # get penalty
        current_penalty = self.get_penalty(x_candidates, x_tead_scores)
        self.global_state.tead_global_scores = x_tead_scores * current_penalty

    # def get_candidates_and_scores_from_node(self, graph, n): # todo see if there's a function to create here
    #     current_node = graph.nodes()[n]
    #     current_state = current_node['data']
    #     current_bounds = current_state.local_bounds
    #     mask_cands = torch.ge(x_cands, current_bounds[0]) & torch.lt(x_cands, current_bounds[1])
    #     current_cands = x_cands[mask_cands].unsqueeze(-1)
    #     current_scores = x_cand_scores[mask_cands].unsqueeze(-1)
    def compute_subdomain_preselect_scores(self):
        """Determine the per-leaf-node pre-select score data and assign to nodes"""
        # get leaf nodes in graph
        graph = self.global_state.partition_graph
        graph_leaves = [n for n in graph if graph.out_degree[n] == 0]
        # separate candidates and scores between leaves and compute average scores (generate more if none for node)
        x_cands = self.global_state.x_global_candidates
        x_cand_scores = self.global_state.tead_global_scores
        current_max_score = 0
        for n in graph_leaves:
            # get current leaf node
            current_node = graph.nodes()[n]
            current_state = current_node['data']
            current_bounds = current_state.local_bounds
            mask_cands = torch.ge(x_cands, current_bounds[0]) & torch.lt(x_cands, current_bounds[1])
            current_cands = x_cands[mask_cands].unsqueeze(-1)
            current_scores = x_cand_scores[mask_cands].unsqueeze(-1)
            tead_max_score = torch.max(x_cand_scores[mask_cands])
            current_ave_score = current_scores.mean()
            current_ave_score = tead_max_score
            current_score = tead_max_score
            # add {cand, score} and ave(score) data to current leaf node in tree
            current_node['cands'] = (current_cands, current_scores)
            current_node['preselect'] = current_score
            print("PRESELECT: For node ", n, " tead score: ", current_score)
            # save the index of the max score node to use for partitioning
            if current_score > current_max_score:
                self.current_max_preselect_node = n
                current_max_score = current_score
        print("PRESELECT: Selecting node ", self.current_max_preselect_node,
              " for split with score ", current_max_score.item())

    def determine_penalty_radius(self, extrema_set):
        """determine what the penalty radius should be based on the current extrema set"""
        if len(extrema_set) <= 1:
            self.penalty_radius = torch.tensor([0.25])
        else:
            # calculate the distances between all the extrema
            extrema_tensor = torch.tensor(extrema_set)  # dim0=extrema index, dim1: 0=x, 1=y
            dists = []
            x_vals = extrema_tensor[:, 0]
            for i in range(len(x_vals)):
                for j in range(len(x_vals)):
                    dists.append(torch.linalg.norm(x_vals[i]-x_vals[j]))
            max_dist = torch.max(torch.tensor(dists))
            new_rad = 0.25*max_dist
            # check if the radius is too small
            if new_rad < self.penalty_reset_size:
                self.penalty_radius = torch.tensor([0.25], dtype=dtype)
            else:
                self.penalty_radius = new_rad

    def get_penalty(self, x, score):
        """Modify a combination of x locations and associated scores with penalty function"""
        past_extrema = self.global_state.converged_extrema
        # check how many past extrema are available, and if 4+, get the most recent 4
        if past_extrema is not None:
            num_extrema = len(past_extrema)
            if num_extrema >= 4:
                penalized_extrema = past_extrema[-4:]
            else:
                penalized_extrema = past_extrema
            # update current penalty radius
            self.determine_penalty_radius(penalized_extrema)
            cur_rad = self.penalty_radius
            # create penalty function on domain of x (candidates, discrete in this case)
            penalty = torch.ones_like(score)
            for extrema in penalized_extrema:
                x_ext, y_ext = extrema
                lower_bound = x_ext - cur_rad
                upper_bound = x_ext + cur_rad
                # find x in penalty domain
                idx_to_penalize = torch.ge(x, lower_bound) & torch.lt(x, upper_bound)
                penalty[idx_to_penalize] *= 0.5
            return penalty
        else:
            # do not apply penalty if no past extrema converged
            print('No past extrema to penalize search near')

    def evaluate_objective_at_split(self, node, l_bound, r_bound):
        """Run the objective evaluation with global metric at global search partition"""
        # get candidates and scores off of graph
        initial_x_cands, initial_scores = node['cands']
        # get indices for each node
        x_cands_left_mask = torch.ge(initial_x_cands, l_bound[0]) & torch.lt(initial_x_cands, l_bound[1])
        x_cands_right_mask = torch.ge(initial_x_cands, r_bound[0]) & torch.lt(initial_x_cands, r_bound[1])
        # get partitioned candidates and scores
        x_cands_left = initial_x_cands[x_cands_left_mask].unsqueeze(-1)
        x_cands_right = initial_x_cands[x_cands_right_mask].unsqueeze(-1)
        scores_left = initial_scores[x_cands_left_mask].unsqueeze(-1)
        scores_right = initial_scores[x_cands_right_mask].unsqueeze(-1)
        # find argmax_x (score_i)
        score_max_left, max_idx_left = torch.max(input=scores_left, dim=0, keepdim=True)
        # score_max_left, max_idx_left = torch.max(input=scores_left)
        x_max_left = x_cands_left[max_idx_left[0]]
        score_max_right, max_idx_right = torch.max(input=scores_right, dim=0, keepdim=True)
        # score_max_right, max_idx_right = torch.max(input=scores_right)
        x_max_right = x_cands_right[max_idx_right[0]]
        # evaluate obj func
        y_new_left = self.obj_func(x_max_left)
        y_new_right = self.obj_func(x_max_right)
        self.current_iteration += 2
        return x_max_left, y_new_left, x_max_right, y_new_right

    def update_extreme(self):
        """Check current extreme and current  x, y values and update extreme
            Importantly - this is without updating the penalty extreme set"""
        new_extreme_y, extreme_idx = torch.max(input=self.global_state.y_global, dim=0, keepdim=True)
        new_extreme_x = self.global_state.x_global[extreme_idx[0]]
        current_extreme_y = self.global_state.y_global_extreme
        if new_extreme_y > current_extreme_y:
            self.global_state.x_global_extreme = new_extreme_x
            self.global_state.y_global_extreme = new_extreme_y

    def get_new_search(self, node_bound):
        """Select the domain and set up a search object for the node"""
        x_node = self.global_state.x_global
        node_mask = torch.ge(x_node, node_bound[0]) & torch.lt(x_node, node_bound[1])
        x_node = x_node[node_mask].unsqueeze(-1)
        y_left = self.global_state.y_global[node_mask].unsqueeze(-1)
        node_local_data = LocalSearchState(input_dim=1, local_bounds=node_bound,
                                           x_local=x_node,
                                           y_local=y_left, )
        return node_local_data

    def run_global_partition_step(self):
        """Run the partitioning step and split a subdomain using the pre-select scores"""
        # get node to partition
        graph = self.global_state.partition_graph
        node_to_split = graph.nodes()[self.current_max_preselect_node]
        # todo: make work for dim>1 (select dim to split)
        # split node halfway through bounds to get new subdomain bounds
        prior_bounds = node_to_split['data'].local_bounds
        lower_bound = prior_bounds[0]
        upper_bound = prior_bounds[1]
        center_bound = (upper_bound - lower_bound)/2. + lower_bound
        left_bound = torch.tensor([lower_bound, center_bound])
        right_bound = torch.tensor([center_bound, upper_bound])
        print("PARTITION: Splitting node ", self.current_max_preselect_node, " from ", lower_bound.item(), " to ",
              center_bound.item(), " to ", upper_bound.item())
        # set up new nodes
        left_local_data = self.get_new_search(left_bound)
        right_local_data = self.get_new_search(right_bound)
        current_max_node_num = graph.number_of_nodes() - 1
        self.global_state.partition_graph.add_node(current_max_node_num+1, data=left_local_data)
        self.global_state.partition_graph.add_edge(self.current_max_preselect_node, current_max_node_num+1)
        print("PARTITION: New node ", current_max_node_num+1, " new edge from ", self.current_max_preselect_node,
              " to ", current_max_node_num+1)
        self.global_state.partition_graph.add_node(current_max_node_num+2, data=right_local_data)
        self.global_state.partition_graph.add_edge(self.current_max_preselect_node, current_max_node_num+2)
        print("PARTITION: New node ", current_max_node_num+2, " new edge from ", self.current_max_preselect_node,
              " to ", current_max_node_num+2)
        self.global_state.num_levels += 1
        # evaluate objective at max TEAD score in each new node
        x_max_left, y_new_left, x_max_right, y_new_right = self.evaluate_objective_at_split(node_to_split,
                                                                                            left_bound,
                                                                                            right_bound)
        self.global_state.x_global = torch.cat((self.global_state.x_global, x_max_left), 0)
        self.global_state.y_global = torch.cat((self.global_state.y_global, y_new_left), 0)
        self.global_state.x_global = torch.cat((self.global_state.x_global, x_max_right), 0)
        self.global_state.y_global = torch.cat((self.global_state.y_global, y_new_right), 0)
        # check if new data points change the best observed global extreme
        self.update_extreme()

    def compute_global_select_scores(self):
        """Use global value algorithm to assign value to subdomains for local search"""
        self.fit_global_model()
        self.fit_global_model_partitioned()
        # option to use NEI or to use TEAD
        # todo: set up option, right now just TEAD
        # have option for two approaches for tead
        if self.tead_model_type == 'global':
            x_candidates, x_tead_scores = global_tead(self.global_state.global_model, get_all_candidates=True)
        elif self.tead_model_type == 'piecewise':
            x_candidates, x_tead_scores = piecewise_tead(self.global_state.x_global, self.global_state.y_global,
                                                         self.global_state.partition_graph, get_all_candidates=True)
        else:
            print('Global acquisition for select scores not recognized')
        # adding penalty
        current_penalty = self.get_penalty(x_candidates, x_tead_scores)
        x_tead_scores *= current_penalty
        # separate and use scores based on leaves
        graph = self.global_state.partition_graph
        graph_leaves = [n for n in graph if graph.out_degree[n] == 0]
        best_n = None
        best_score = torch.tensor([[0.]])
        for n in graph_leaves:
            # get current leaf node
            current_node = graph.nodes()[n]
            current_state = current_node['data']
            current_bounds = current_state.local_bounds
            mask_cands = torch.ge(x_candidates, current_bounds[0]) & torch.lt(x_candidates, current_bounds[1])
            current_cands = x_candidates[mask_cands].unsqueeze(-1)
            current_scores = x_tead_scores[mask_cands].unsqueeze(-1)
            cur_max_tead, max_tead_idx = torch.max(input=current_scores, dim=0, keepdim=True)
            cur_max_x = current_cands[max_tead_idx[0]]
            # add {cand, score} and max(score) data to current leaf node in tree
            current_node['select_cands'] = (current_cands, current_scores)
            current_node['select'] = (cur_max_x, cur_max_tead)
            print("SELECT: For node ", n, " best x: ", cur_max_x.item(), " best tead: ", cur_max_tead.item())
            if cur_max_tead > best_score:
                best_score = cur_max_tead
                best_n = n
        print("SELECT: Found ", best_n, " as best with score ", best_score.item())
        return best_n

    def reset_leaf_node_local_data(self):
        """Go through leaf nodes and update LocalSearchState to have local {x, y}
        based on global {x, y} that are within bounds"""
        graph = self.global_state.partition_graph
        graph_leaves = [n for n in graph if graph.out_degree[n] == 0]
        for n in graph_leaves:
            current_node = graph.nodes()[n]
            current_state = current_node['data']
            current_bounds = current_state.local_bounds
            all_x = self.global_state.x_global
            all_y = self.global_state.y_global
            mask_local = torch.ge(all_x, current_bounds[0]) & torch.lt(all_x, current_bounds[1])
            local_x = all_x[mask_local]
            local_y = all_y[mask_local]
            current_state.x_local = local_x.unsqueeze(-1)
            current_state.y_local = local_y.unsqueeze(-1)

    def run_selected_local_search(self, node):
        """Select and run the local search in a specified subdomain"""
        # get node to run search in
        graph = self.global_state.partition_graph
        current_node = graph.nodes()[node]
        print("LOCALSEARCH: Running search in node ", node)
        current_local_data = current_node['data']
        # set up local search
        # run initial tead points
        num_initial_tead = 1
        local_pre_search = LocalExtremeSearch(num_initial_tead, self.min_local_init_evals,
                                              current_local_data, self.obj_func)
        current_node['presearch'] = local_pre_search
        local_pre_search.run_local_search('tead')
        # update state for running turbo search - is this needed? Or is the object pointed to
        # automatically based on the connection between the current_local_data object
        # current_node['data'] = LocalSearchState(input_dim=1, local_bounds=self.x_bounds,
        #                                       x_local=self.global_state.x_global,
        #                                       y_local=self.global_state.y_global,)
        # run turbo points
        local_search = LocalExtremeSearch(self.max_local_evals, self.min_local_init_evals,
                                          current_local_data, self.obj_func)
        current_node['search'] = local_search
        # run local search
        # local_search.run_local_search()
        local_search.run_local_search()
        # update global search state
        x_new = current_node['search'].local_state.most_recent_x_local
        y_new = current_node['search'].local_state.most_recent_y_local
        self.current_iteration += int(x_new.shape[0])
        print("LOCALSEARCH: Completed ", int(x_new.shape[0]), " new evaluations between ", torch.min(x_new).item(),
              " and ", torch.max(x_new).item())
        self.global_state.x_global = torch.cat((self.global_state.x_global, x_new), 0)
        self.global_state.y_global = torch.cat((self.global_state.y_global, y_new), 0)
        # update current global extrema
        extreme_y, extreme_idx = torch.max(input=self.global_state.y_global, dim=0, keepdim=True)
        extreme_x = self.global_state.x_global[extreme_idx[0]]
        # current global extreme
        self.global_state.x_global_extreme = extreme_x
        self.global_state.y_global_extreme = extreme_y
        # add local converged extrema to converged extrema tracker
        local_extreme_y, local_extreme_idx = torch.max(input=y_new, dim=0, keepdim=True)
        local_extreme_x = x_new[local_extreme_idx[0]]
        self.global_state.converged_extrema.append((local_extreme_x, local_extreme_y))
        # reset leaf node local_x and local_y based on bounds and global {x, y}
        self.reset_leaf_node_local_data()
        # fit the piecewise models across the full space
        self.fit_global_model_partitioned()

    def run_global_search(self):
        """Run the subroutines for the global search"""
        self.initialize_global_search()
        while self.current_iteration <= self.total_iteration_limit:
            print('Fitting global model')
            self.fit_global_model()
            self.fit_global_model_partitioned()
            print('Exploring full space')
            self.run_global_tead_exploration_evaluations()
            print('Calculating pre-select scores')
            self.compute_global_preselect_scores()
            print('Calculating subdomain pre-select scores')
            self.compute_subdomain_preselect_scores()
            print('Running partition')
            self.run_global_partition_step()
            print('Running subdomain selection')
            n_to_search = self.compute_global_select_scores()
            print('Running local search')
            self.run_selected_local_search(n_to_search)

    def fit_global_model(self):
        """Fit the global surrogate model"""
        fit_gpytorch_mll(self.global_state.global_mll)

    def fit_global_model_partitioned(self):
        """Fit a global model by combining the local models into a piecewise set
        -requires that the local {x, y} values are updated prior to calling this function"""
        graph = self.global_state.partition_graph
        graph_leaves = [n for n in graph if graph.out_degree[n] == 0]
        for n in graph_leaves:
            current_node = graph.nodes()[n]
            current_state = current_node['data']
            current_state.local_model = SingleTaskGP(current_state.x_local, current_state.y_local,
                                                     input_transform=Normalize(d=current_state.x_local.shape[-1]),
                                                     outcome_transform=Standardize(m=current_state.y_local.shape[-1]))
            current_state.local_mll = ExactMarginalLogLikelihood(current_state.local_model.likelihood,
                                                                 current_state.local_model)
            fit_gpytorch_mll(current_state.local_mll)

    def get_bounds_of_leaves(self):
        """Get the (node, bounds) pairs"""
        graph = self.global_state.partition_graph
        graph_leaves = [n for n in graph if graph.out_degree[n] == 0]
        bound_list = []
        node_list = []
        for n in graph_leaves:
            # get current leaf node
            current_node = graph.nodes()[n]
            current_state = current_node['data']
            current_bounds = current_state.local_bounds
            node_list.append(n)
            bound_list.append(current_bounds)
        return node_list, bound_list

    # moved to tead module
    # def get_model_from_piecewise_set(self, x):
    #     """Retrieves the local model from the piecewise set at the input point x"""
    #     graph = self.global_state.partition_graph
    #     graph_leaves = [n for n in graph if graph.out_degree[n] == 0]
    #     for n in graph_leaves:
    #         # get current leaf node
    #         current_node = graph.nodes()[n]
    #         current_state = current_node['data']
    #         current_bounds = current_state.local_bounds
    #         if current_bounds[0] <= x < current_bounds[1]:
    #             return current_state.local_model
    #     print("Using single global model")
    #     return self.global_state.global_model






