#https://zhuanlan.zhihu.com/p/321449610
import torch

class Sigmoid(torch.autograd.Function):
                                                
    @staticmethod
    def forward(ctx, x): 
        output = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        output,  = ctx.saved_tensors
        grad_x = output * (1 - output) * grad_output
        return grad_x

test_input = torch.randn(4, requires_grad=True)     # tensor([-0.4646, -0.4403,  1.2525, -0.5953], requires_grad=True)
print(torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-3))   # pass
print(torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-3))    # pass
print(torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-4))    # fail
print(torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-4))    # fail


test_input = torch.randn(4, requires_grad=True, dtype=torch.float64)    # tensor([-0.4646, -0.4403,  1.2525, -0.5953], dtype=torch.float64, requires_grad=True)
print(torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-4))   # pass
print(torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-4))    # pass

print(torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-6))    # pass
print(torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-6))    # pass
