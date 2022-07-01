from torch.utils.tensorboard import SummaryWriter
from outside.resnet_model import ACBResNet50, ResNet50
import torch
from torchvision.models import resnet50
import numpy as np

# 需要pip tensorboard才行，

if __name__ == '__main__':
    writer = SummaryWriter('runs/exp4')
    net = resnet50()
    # net2 = ResNet50()
    image = torch.ones((1, 3, 32, 32))
    writer.add_graph(net, image)
    # writer.add_graph(net2, image)
    writer.close()
    # 等程序结束后 控制台在能看到runs的目录运行: tensorboard --logdir=runs

