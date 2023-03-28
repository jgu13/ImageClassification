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
        super().__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(
           in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, planes,
                         kernel_size=1, stride=stride, bias=False), # make sure F(x) and x have the same number of channels
               nn.BatchNorm2d(planes)
           )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        fx = nn.ReLU()(self.bn1(self.conv1(x)))
        # 2. Go through conv2, bn
        fx = self.bn2(self.conv2(fx))
        # 3. Combine with shortcut output, and go through relu
        output = nn.ReLU()(fx + self.shortcut(x))
        return output


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Uncomment the following lines and replace the ? with correct values
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        images = nn.ReLU()(self.bn1(self.conv1(images)))
        layer1_out = self.layer1(images)
        # print("Layer1: ", layer1_out.size())
        layer2_out = self.layer2(layer1_out)
        # print("Layer2: ", layer2_out.size())
        layer3_out = self.layer3(layer2_out)
        # print("Layer3: ", layer3_out.size())
        layer4_out = self.layer4(layer3_out)
        # print("Layer4: ", layer4_out.size())
        img_size = (int(images.shape[2]/8), int(images.shape[3]/8))
        avgpool_out = nn.AvgPool2d(img_size)(layer4_out)
        # print("Average Pooling: ", avgpool_out.size())
        flattened = torch.flatten(avgpool_out, 1)
        # print("Flattened: ", flattened.size())
        logits = self.linear(flattened)
        return logits

    def get_weights(self):
        # get weights of each layer
        filters = {}
        filters['conv1'] = self.conv1.weight
        return filters

    def visualize(self, logdir=None):
        """ Visualize the kernel in the desired directory """
        import matplotlib as mpl
        import os.path as osp
        filters = self.get_weights()
        first_conv_layer = filters['conv1']
        fmin, fmax = first_conv_layer.min(), first_conv_layer.max()
        first_conv_layer = (first_conv_layer - fmin) / (fmax - fmin)
        ncols = 8
        nrows = 8 if (first_conv_layer.size()[0] % 8 == 0) else 9
        fig, axs = mpl.pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
        axs = axs.ravel()
        for i in range(first_conv_layer.size()[0]):
            f = first_conv_layer[i, :,:,:].detach().cpu().numpy()
            f = f.mean(axis=-1) # average across channels
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].imshow(f, cmap='gray')
        if logdir != None:
          mpl.pyplot.savefig(osp.join(logdir, 'visualize_first_conv_layer.png'))
        mpl.pyplot.show()
        
