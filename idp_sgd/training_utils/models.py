from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import math


class VGG(nn.Module):
    def __init__(self, architecture_name="VGG7", dataset_name="MNIST"):
        super(VGG, self).__init__()

        self.num_classes = 10  # NOTE : HARDCODE
        if dataset_name == "MNIST" or dataset_name == "Fashion_MNIST":
            self.in_channels = 1
        elif dataset_name == "CIFAR10":
            self.in_channels = 3
        else:
            raise Exception(f"Dataset not known {dataset_name}")
        if architecture_name == 'VGG3':
            self.cfg = [64, 'M', 128, 128, 'M']
        elif architecture_name == 'VGG5':
            self.cfg = [64, 'M', 128, 'M', 256, 'M', 512]
        elif architecture_name == 'VGG7':
            self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512]
        elif architecture_name == 'VGG9':
            self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 'M', 512, 'M']
        elif architecture_name == 'VGG11':
            self.cfg = [
                64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
            ]
        elif architecture_name == 'VGG13':
            self.cfg = [
                64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
                512, 'M'
            ]
        elif architecture_name == 'VGG16':
            self.cfg = [
                64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M'
            ]
        elif architecture_name == 'VGG19':
            self.cfg = [
                64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
                512, 512, 'M', 512, 512, 512, 512, 'M'
            ]
        else:
            raise Exception("VGG architecture not known")
        self.features = self.make_layers()
        self.classifier = nn.Linear(
            128 if architecture_name == 'VGG3' else 512, self.num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def make_layers(self):
        layers = []
        in_channels = self.in_channels
        for c in self.cfg:
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv = nn.Conv2d(in_channels,
                                 c,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)
                layers += [conv, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                in_channels = c
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x


def clip_data(data, max_norm):
    norms = torch.norm(data.reshape(data.shape[0], -1), dim=-1)
    scale = (max_norm / norms).clamp(max=1.0)
    data *= scale.reshape(-1, 1, 1, 1)
    return data


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class StandardizeLayer(nn.Module):
    def __init__(self, bn_stats):
        super(StandardizeLayer, self).__init__()
        self.bn_stats = bn_stats

    def forward(self, x):
        return standardize(x, self.bn_stats)


class ClipLayer(nn.Module):
    def __init__(self, max_norm):
        super(ClipLayer, self).__init__()
        self.max_norm = max_norm

    def forward(self, x):
        return clip_data(x, self.max_norm)


class CIFAR10_CNN(nn.Module):
    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels,
                                         affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(),
                                            nn.Linear(hidden, 10))
        else:
            self.classifier = nn.Linear(c * 4 * 4, 10)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MNIST_CNN(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels,
                                         affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size,
                                   stride=stride, padding=pad)

                layers += [conv2d, nn.Tanh()]
                c = filters

        self.features = nn.Sequential(*layers)

        hidden = 32
        self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                        nn.Tanh(),
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


CNNS = {
    "cifar10": CIFAR10_CNN,
    "fmnist": MNIST_CNN,
    "mnist": MNIST_CNN,
}

if __name__ == "__main__":
    mnist_model = MNIST_CNN()
