import torch
import torch.nn as nn
from model_utils import *

class TIP(nn.Module):
    def __init__(self, use_attention=True):
        super(TIP, self).__init__()

        self.use_attention = use_attention

        self.extract_feat = get_features()
        self.fg_reconstruction = skip_connections()
        self.bg_reconstruction = skip_connections()

        self.tip7 = MASA_Projection(dims_in=512)
        self.tip6 = MASA_Projection(dims_in=512)
        self.tip5 = MASA_Projection(dims_in=512)
        self.tip4 = MASA_Projection(dims_in=512)
        self.tip3 = MASA_Projection(dims_in=256)
        self.tip2 = MASA_Projection(dims_in=128)
        self.tip1 = MASA_Projection(dims_in=64)

        self.up_path1 = nn.Sequential(nn.ReLU(True),
                                      nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False))
        self.up_path2 = nn.Sequential(nn.ReLU(True),
                                      nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False))
        self.up_path3 = nn.Sequential(nn.ReLU(True),
                                      nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False))
        self.up_path4 = nn.Sequential(nn.ReLU(True),
                                      nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False))
        if use_attention:
            self.up_path4att = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1), nn.Sigmoid())

        self.up_path5 = nn.Sequential(nn.ReLU(True),
                                      nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False))
        if use_attention:
            self.up_path5att = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.Sigmoid())

        self.up_path6 = nn.Sequential(nn.ReLU(True),
                                      nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False))
        if use_attention:
            self.up_path6att = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, stride=1), nn.Sigmoid())

        self.fusion_out = nn.Sequential(nn.ReLU(True),
                                        nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
                                        nn.Tanh())

        self.init_weights()
    def init_weights(self):
        for name, param in self.named_parameters():

            if 'weight' in name:
                nn.init.normal_(param, 0.0, 0.01)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)

    def get_fore_back(self, fx, mask):

        bs, c, height, width = fx.size()
        fore = fx * mask
        fx_f_feats = nn.AvgPool2d(kernel_size=(height, width))(fore)
        back = fx * (1 - mask)
        fx_b_feats = nn.AvgPool2d(kernel_size=(height, width))(back)

        return fx_f_feats.view(bs, c), fx_b_feats.view(bs, c)


    def forward(self, fx, bx, mask):

        fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7 = self.extract_feat(fx)
        bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7 = self.extract_feat(bx)

        fret7, fret6, fret5, fret4, fret3, fret2, fret1, fout = self.fg_reconstruction(fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7)
        bret7, bret6, bret5, bret4, bret3, bret2, bret1, bout = self.bg_reconstruction(bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7)

        fx_f_feats, fx_b_feats = self.get_fore_back(fout, mask)

        f7 = self.tip7(fret7, bret7, mask)
        f6 = self.tip6(fret6, bret6, mask)
        f5 = self.tip5(fret5, bret5, mask)
        f4 = self.tip4(fret4, bret4, mask)
        f3 = self.tip3(fret3, bret3, mask)
        f2 = self.tip2(fret2, bret2, mask)
        f1 = self.tip1(fret1, bret1, mask)

        f6 = torch.cat([self.up_path1(f7), f6], 1)
        f5 = torch.cat([self.up_path2(f6), f5], 1)
        f4 = torch.cat([self.up_path3(f5), f4], 1)
        f3 = torch.cat([self.up_path4(f4), f3], 1)
        if self.use_attention:
            f3 = self.up_path4att(f3) * f3
        f2 = torch.cat([self.up_path5(f3), f2], 1)
        if self.use_attention:
            f2 = self.up_path5att(f2) * f2
        f1 = torch.cat([self.up_path6(f2), f1], 1)
        if self.use_attention:
            f1 = self.up_path6att(f1) * f1

        out = self.fusion_out(f1)

        out = out * mask + bx[:, :3, :, :] * (1 - mask)

        return torch.tanh(fout), torch.tanh(bout), fx_f_feats, fx_b_feats, out

if __name__ == '__main__':

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   net = TIP().to(device)