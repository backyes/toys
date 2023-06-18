import torch
from torch import nn

# this
my_data = torch.tensor([1,2,3], dtype=torch.float32).repeat(1000, 1)
weights = torch.rand((3,2))
output_mm = torch.matmul(my_data, weights)
print(output_mm)

linear = nn.Linear(3, 2)
output_linear = linear(my_data)
print(output_linear)
print(torch.eq(output_linear, output_mm))


# torch.nn.Linear, torch.matmul
linear = torch.nn.Linear(3, 3)
inputs = torch.rand(3, 3)

output_linear = linear(inputs)
print(output_linear)
output_mm = torch.matmul(inputs, linear.weight.T).add(linear.bias)
print(output_mm)
print(linear.weight.T)
print(linear.bias)

print(torch.eq(output_linear, output_mm))


# https://www.zhihu.com/question/66782101
# toch.
