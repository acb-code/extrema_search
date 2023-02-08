from extremasearch.acquisition.tead import finite_diff
import torch
from botorch.models import SingleTaskGP

train_X = torch.rand(20, 2)
train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
model_example = SingleTaskGP(train_X, train_Y)

print(finite_diff(model_example))
