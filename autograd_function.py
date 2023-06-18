import torch
from torch.autograd import Function
import torch.nn as nn

# https://blog.csdn.net/guofei_fly/article/details/105098728

# 张量函数的前向和反向过程
class SquareFun(Function):
    
    @staticmethod
    def forward(ctx, x1, x2):
        result = x1 ** 3 + x2 ** 2
        ctx.save_for_backward(x1, x2)
        return result
    
    @staticmethod
    def backward(ctx, grad_out):
        x1, x2 = ctx.saved_tensors
        return 3 * x1**2 * grad_out, 2 * x2 * grad_out

# 定义对应的module层
class Square(nn.Module):
    def forward(self, x1, x2):
        return SquareFun.apply(x1, x2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = Square()
    
    def forward(self, x1, x2):
        y = self.layer1(x1, x2)
        y = y.sum()
        return y


net = Net()
x1 = torch.tensor([1,2,3], requires_grad=True, dtype=torch.float32)
x2 = torch.tensor([4,5,6], requires_grad=True, dtype=torch.float32)
out = net(x1, x2)

print(out)  # tensor(113., grad_fn=<SumBackward0>)
out.backward()

print(x1.grad)  # tensor([ 3., 12., 27.])
print(x2.grad)  # tensor([ 8., 10., 12.])
