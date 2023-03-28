'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        # Uncomment the following lines, replace the ? with correct values.
        #self.conv1 = nn.Conv2d(
        #    ?, planes, kernel_size=3, stride=?, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(?)
        #self.conv2 = nn.Conv2d(planes, ?, kernel_size=3,
        #                       stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(?)

        #self.shortcut = nn.Sequential()
        #if stride != 1 or in_planes != planes:
        #    self.shortcut = nn.Sequential(
        #        nn.Conv2d(?, ?,
        #                  kernel_size=1, stride=stride, bias=False),
        #        nn.BatchNorm2d(?)
        #    )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        # 2. Go through conv2, bn
        # 3. Combine with shortcut output, and go through relu
        raise NotImplementedError


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Uncomment the following lines and replace the ? with correct values
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
        #                       stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(?)
        #self.layer1 = self._make_layer(64, 64, stride=1)
        #self.layer2 = self._make_layer(?, 128, stride=2)
        #self.layer3 = self._make_layer(?, 256, stride=2)
        #self.layer4 = self._make_layer(?, 512, stride=2)
        #self.linear = nn.Linear(?, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        raise NotImplementedError

    def visualize(self, logdir):
        """ Visualize the kernel in the desired directory """
        raise NotImplementedError
