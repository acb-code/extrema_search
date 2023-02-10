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

# setup

# objects