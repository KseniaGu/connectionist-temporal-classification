import torch.nn as nn
import torch

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class RecurrentConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(RecurrentConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_size), nn.LeakyReLU(),
            nn.Conv2d(output_size, output_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_size), nn.LeakyReLU()
        )

    def forward(self, x):
        return  self.conv_block(x)


class Model(nn.Module):
    def __init__(self, alphabet_size=63, use_log_softmax=False):
        super(Model, self).__init__()

        sftmax_layer = nn.LogSoftmax() if use_log_softmax else nn.Sequential()

        self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.LeakyReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.LeakyReLU(),
                                  nn.MaxPool2d(2, stride=2, padding=1),
                                  nn.Conv2d(32, 64, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(64), nn.LeakyReLU(),
                                  RecurrentConv(64, 64),
                                  nn.MaxPool2d(2, stride=2, padding=1),
                                  nn.Conv2d(64, 128, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(128), nn.LeakyReLU(),
                                  RecurrentConv(128, 128),
                                  nn.MaxPool2d(2, stride=(2, 1), padding=(1,0)),
                                  nn.Conv2d(128, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256), nn.LeakyReLU(),
                                  RecurrentConv(256, 256),
                                  nn.MaxPool2d(2, stride=(2, 1), padding=(1, 0)),
                                  nn.Conv2d(256, 512, (3, 2), padding=1, bias=False),nn.LeakyReLU(),
                                  nn.Conv2d(512, 512, (5, 1), padding=(2,0), bias=False),nn.LeakyReLU(),
                                  nn.Conv2d(512, alphabet_size, (7, 1), padding=(2, 0), bias=False),
                                  sftmax_layer
                                  )

        for m in self.modules():
            if isinstance(m, nn.Sequential):
                m.apply(init_weights)
            else:
                init_weights(m)

    def forward(self, x):
        out = self.conv(x)
        return out

