from python_environment_check import check_packages
import torch
import torch.nn as nn

# Check recommended package versions:
d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'torch': '1.8',
}
check_packages(d)

# ### Creating a graph in PyTorch
# 
# 
def compute_z(a, b, c):
   r1 = torch.sub(a, b)
   r2 = torch.mul(r1, 2)
   z = torch.add(r2, c)
   return z

print('Scalar Inputs:', compute_z(torch.tensor(1), torch.tensor(2), torch.tensor(3)))
print('Rank 1 Inputs:', compute_z(torch.tensor([1]), torch.tensor([2]), torch.tensor([3])))
print('Rank 2 Inputs:', compute_z(torch.tensor([[1]]), torch.tensor([[2]]), torch.tensor([[3]])))

# ## PyTorch Tensor objects for storing and updating model parameters
a = torch.tensor(3.14, requires_grad=True)
b = torch.tensor([1.0, 2.0, 3.0], requires_grad=True) 
print(a)
print(b)

w = torch.tensor([1.0, 2.0, 3.0])
print(w.requires_grad)
w.requires_grad_()
print(w.requires_grad)

torch.manual_seed(1)
w = torch.empty(2, 3)
nn.init.xavier_normal_(w)
print("Weight : ", w)

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.empty(2, 3, requires_grad=True)
        nn.init.xavier_normal_(self.w1)
        self.w2 = torch.empty(1, 2, requires_grad=True)
        nn.init.xavier_normal_(self.w2)

# ## Computing gradients via automatic differentiation and GradientTape
# 
# ### Computing the gradients of the loss with respect to trainable variables
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True) 
x = torch.tensor([1.4])
y = torch.tensor([2.1])
z = torch.add(torch.mul(w, x), b)
 
loss = (y-z).pow(2).sum()
loss.backward()

print('dL/dw : ', w.grad)
print('dL/db : ', b.grad)

# verifying the computed gradient dL/dw
print(2 * x * ((w * x + b) - y))