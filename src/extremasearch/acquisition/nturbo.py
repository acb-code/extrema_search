# setting up n-dim version of turbo based on BoTorch setup combined with modifications from 1-d case
#
# Author: Alex Braafladt
# Date: 3/13/2023
#
# TuRBO references:
#   [1] D. Eriksson, M. Pearce, J. R. Gardner, R. Turner, and M. Poloczek, “Scalable global optimization via local
#   Bayesian optimization,” Adv. Neural Inf. Process. Syst., vol. 32, no. NeurIPS, 2019.
#   [2] https://github.com/uber-research/TuRBO (restrictive license)
#   [3] https://botorch.org/tutorials/turbo_1 (MIT license)
#   [4] https://proceedings.neurips.cc/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html
#   [5] https://proceedings.neurips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Supplemental.zip
#
# Notes:
#   -Original algorithm from [1] and [2], modified MIT Licensed version from [3], supplemental material in [4][5]
#   -Baseline algorithm from [3] modified to use local-only GP by me (and as considered hypothetically in [1]) and
#    uses slightly different parameters/logic for the trust region update process
#   -This implementation extends a 1-D version to n-D with a separate example notebook for a 2-D case












