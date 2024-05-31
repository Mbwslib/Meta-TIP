import torch
import torch.nn as nn

class get_features(nn.Module):
    def __init__(self, input_nc=3, ngf=64):
        super(get_features, self).__init__()
        self.input_nc = input_nc

        self.model_layer0 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False)

        self.model_layer1 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                          nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                                          nn.InstanceNorm2d(ngf * 2))

        self.model_layer2 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                          nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                                          nn.InstanceNorm2d(ngf * 4))

        self.model_layer3 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                          nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
                                          nn.InstanceNorm2d(ngf * 8))

        self.model_layer4 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                          nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
                                          nn.InstanceNorm2d(ngf * 8))

        self.model_layer5 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                          nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
                                          nn.InstanceNorm2d(ngf * 8))

        self.model_layer6 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                          nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
                                          nn.InstanceNorm2d(ngf * 8))

        self.model_layer7 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                          nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False))

    def forward(self, x):
        x0 = self.model_layer0(x)
        x1 = self.model_layer1(x0)
        x2 = self.model_layer2(x1)
        x3 = self.model_layer3(x2)
        x4 = self.model_layer4(x3)
        x5 = self.model_layer5(x4)
        x6 = self.model_layer6(x5)
        x7 = self.model_layer7(x6)

        return x0, x1, x2, x3, x4, x5, x6, x7


class skip_connections(nn.Module):
    def __init__(self, ngf=64):
        super(skip_connections, self).__init__()

        self.up_layer0 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1,
                                                          bias=False))
        self.upnorm0 = nn.InstanceNorm2d(ngf * 8)

        self.up_layer1 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1,
                                                          bias=False))
        self.upnorm1 = nn.InstanceNorm2d(ngf * 8)

        self.up_layer2 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1,
                                                          bias=False))
        self.upnorm2 = nn.InstanceNorm2d(ngf * 8)

        self.up_layer3 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1,
                                                          bias=False))
        self.upnorm3 = nn.InstanceNorm2d(ngf * 8)

        self.up_layer4 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1,
                                                          bias=False))
        self.upnorm4 = nn.InstanceNorm2d(ngf * 4)

        self.up_layer5 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1,
                                                          bias=False))
        self.upnorm5 = nn.InstanceNorm2d(ngf * 2)

        self.up_layer6 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1, bias=False))
        self.upnorm6 = nn.InstanceNorm2d(ngf)

        self.model_out = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 2, 3, kernel_size=4, stride=2, padding=1), )
        # nn.Tanh())

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7):
        ret7 = self.up_layer0(x7)
        ret6 = self.upnorm0(ret7)
        ret6 = torch.cat([x6, ret6], 1)

        ret6 = self.up_layer1(ret6)
        ret5 = self.upnorm1(ret6)
        ret5 = torch.cat([x5, ret5], 1)

        ret5 = self.up_layer2(ret5)
        ret4 = self.upnorm2(ret5)
        ret4 = torch.cat([x4, ret4], 1)

        ret4 = self.up_layer3(ret4)
        ret3 = self.upnorm3(ret4)
        ret3 = torch.cat([x3, ret3], 1)

        ret3 = self.up_layer4(ret3)
        ret2 = self.upnorm4(ret3)
        ret2 = torch.cat([x2, ret2], 1)

        ret2 = self.up_layer5(ret2)
        ret1 = self.upnorm5(ret2)
        ret1 = torch.cat([x1, ret1], 1)

        ret1 = self.up_layer6(ret1)
        ret0 = self.upnorm6(ret1)
        ret0 = torch.cat([x0, ret0], 1)

        out = self.model_out(ret0)

        return ret7, ret6, ret5, ret4, ret3, ret2, ret1, out




