import sys
from python_environment_check import check_packages
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
# Check recommended package versions:
d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'torch': '1.8',
}
check_packages(d)

# ## Simplifying implementations of common architectures via the torch.nn module
# 
# 
# ### Implementing models based on nn.Sequential
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU()
)

# #### Configuring layers
# 
#  * Initializers `nn.init`: https://pytorch.org/docs/stable/nn.init.html 
#  * L1 Regularizers `nn.L1Loss`: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
#  * L2 Regularizers `weight_decay`: https://pytorch.org/docs/stable/optim.html
#  * Activations: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity  
#  
nn.init.xavier_uniform_(model[0].weight)

l1_weight = 0.01
l1_penalty = l1_weight * model[2].weight.abs().sum()

# #### Compiling a model
# 
#  * Optimizers `torch.optim`:  https://pytorch.org/docs/stable/optim.html#algorithms
#  * Loss Functins `tf.keras.losses`: https://pytorch.org/docs/stable/nn.html#loss-functions
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# ## Solving an XOR classification problem
np.random.seed(1)
torch.manual_seed(1)
x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1]<0] = 0

n_train = 100
x_train = torch.tensor(x[:n_train, :], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.float32)
x_valid = torch.tensor(x[n_train:, :], dtype=torch.float32)
y_valid = torch.tensor(y[n_train:], dtype=torch.float32)

fig = plt.figure(figsize=(6, 6))
plt.plot(x[y==0, 0], 
         x[y==0, 1], 'o', alpha=0.75, markersize=10)
plt.plot(x[y==1, 0], 
         x[y==1, 1], '<', alpha=0.75, markersize=10)
plt.xlabel(r'$x_1$', size=15)
plt.ylabel(r'$x_2$', size=15)
#plt.savefig('figures/13_02.png', dpi=300)
plt.show()