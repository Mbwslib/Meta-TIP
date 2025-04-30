import torch
import torch.nn as nn
import torch.nn.functional as F


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
                                       nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False))
        self.upnorm0 = nn.InstanceNorm2d(ngf * 8)


        self.up_layer1 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False))
        self.upnorm1 = nn.InstanceNorm2d(ngf * 8)

        self.up_layer2 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False))
        self.upnorm2 = nn.InstanceNorm2d(ngf * 8)

        self.up_layer3 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False))
        self.upnorm3 = nn.InstanceNorm2d(ngf * 8)

        self.up_layer4 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.upnorm4 = nn.InstanceNorm2d(ngf * 4)

        self.up_layer5 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.upnorm5 = nn.InstanceNorm2d(ngf * 2)

        self.up_layer6 = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1, bias=False))
        self.upnorm6 = nn.InstanceNorm2d(ngf)

        self.model_out = nn.Sequential(nn.ReLU(True),
                                       nn.ConvTranspose2d(ngf * 2, 3, kernel_size=4, stride=2, padding=1),)
                                       #nn.Tanh())

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
    
class MASA_Projection(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        super(MASA_Projection, self).__init__()

        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps
        self.softmax = nn.Softmax(dim=-1)

    def get_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])  # (B, C)
        num = torch.sum(mask, dim=[2, 3])  # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask) * mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var + self.eps)

    def material_aware_attention(self, fx, bx):

        bs, C, height, width = bx.size()
        fx = fx.view(bs, C, -1).permute(0, 2, 1)  # (bs, 1, c)
        bx = bx.view(bs, C, -1)  # (bs, c, h*w)
        energy = torch.bmm(fx, bx)
        attention = torch.bernoulli(self.softmax(energy)).view(bs, 1, height, width)
        #out = (attention * bx).view(bs, C, height, width)

        return attention


    def forward(self, fx, bx, mask):
        mask = F.interpolate(mask.detach(), size=fx.size()[2:], mode='nearest')
        mean_fore, std_fore = self.get_mean_std(fx * mask, mask)  # (bs, c, 1, 1), (bs, c, 1, 1)

        att = self.material_aware_attention(mean_fore, bx)
        mean_back, std_back = self.get_mean_std(bx * att, att)
        #normalized = (att_bx - mean_back) / std_back
        #normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) + self.background_beta[None, :, None, None]) * (1 - mask)

        normalized = (fx - mean_fore) / std_fore * std_back + mean_back
        #normalized_foreground = (normalized * (1 + self.foreground_gamma[None, :, None, None]) + self.foreground_beta[None, :, None, None]) * mask

        return normalized


if __name__ == '__main__':

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   net = MASA(dims_in=3).to(device)
   mask = torch.tensor([[[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0]]]], dtype=torch.float32)
   print(mask)
   fout, bout = net(torch.randn(1, 3, 4, 4).to(device), torch.randn(1, 3, 4, 4).to(device), mask.to(device))





