from torchvision.models import resnet50
import torch

net = resnet50().cuda(0)
num = 128
inp = torch.ones([num, 3, 224, 224]).cuda(0)
net(inp)                                        # 若不开torch.no_grad()，batch_size为128时就会OOM (在1080 Ti上)

net = resnet50().cuda(1)
num = 512
inp = torch.ones([num, 3, 224, 224]).cuda(1)    
with torch.no_grad():                           # 打开torch.no_grad()后，batch_size为512时依然能跑inference (节约超过4倍显存)
    net(inp)
