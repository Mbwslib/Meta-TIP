import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def cos_sim(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)

class SimMinLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):

        sim = cos_sim(embedded_bg, embedded_fg)
        loss = -torch.log(1 - sim)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

class SimMaxLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.reduction = reduction

    def forward(self, embedded_bg):

        sim = cos_sim(embedded_bg, embedded_bg)
        loss = -torch.log(sim)
        loss[loss < 0] = 0

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class RegionL1Loss(nn.Module):
    def __init__(self, is_fore, reduction='mean'):
        super(RegionL1Loss, self).__init__()
        self.is_fore = is_fore
        self.reduction = reduction
        self.l1loss = nn.L1Loss(reduction=self.reduction)

    def forward(self, pred, gt, mask):
        num = torch.sum(mask)  # (B, C)
        if self.is_fore:
            return self.l1loss(pred * mask, gt * mask) / num

        else:
            return self.l1loss(pred, gt)

# class unsupervised_projection_loss(nn.Module):
#     def __init__(self, feature_model, eps=1e-5, reduction='mean'):
#         super(unsupervised_projection_loss, self).__init__()
#         self.eps = eps
#         self.feature_model = feature_model
#         self.reduction = reduction
#
#     def get_weight(self, fore, back, feature_model):
#
#         c1 = 1 / 2
#         c2 = 1
#
#         fore = (fore + 1) / 2
#         back = (back + 1) / 2
#
#         with torch.no_grad():
#             feat_1 = feature_model(fore)
#             feat_2 = feature_model(back)
#
#             for i in range(len(feat_1)):
#                 m1 = torch.mean(features_grad(feat_1[i]).pow(2), dim=[1, 2, 3])
#                 m2 = torch.mean(features_grad(feat_2[i]).pow(2), dim=[1, 2, 3])
#                 if i == 0:
#                     w1 = torch.unsqueeze(m1, dim=-1)
#                     w2 = torch.unsqueeze(m2, dim=-1)
#                 else:
#                     w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
#                     w2 = torch.cat((w2, torch.unsqueeze(m2, dim=-1)), dim=-1)
#             weight_1 = torch.mean(w1, dim=-1) / c1
#             weight_2 = torch.mean(w2, dim=-1) / c2
#             weight_list = torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
#             weight_list = F.softmax(weight_list, dim=-1)
#
#         return weight_list[:, 0].view(4, 1, 1, 1), weight_list[:, 1].view(4, 1, 1, 1)
#
#     def similarity_metric(self, t, f, b):
#
#         similarity = torch.log(t + ((f * b) / (t)) + f + b)
#
#         return similarity
#     def forward(self, target, source_f, source_b, mask):
#         target = torch.pow(target * mask, 2) + self.eps
#
#         #w1, w2 = self.get_weight(source_f, source_b, self.feature_model)
#         source_f = torch.pow(source_f * mask, 4)
#         source_b = torch.pow(source_b * mask, 4)
#
#         unsupervised_loss = self.similarity_metric(target, source_f, source_b)
#
#         if self.reduction == 'mean':
#             return torch.mean(unsupervised_loss)
#         elif self.reduction == 'sum':
#             return torch.sum(unsupervised_loss)
#         else:
#             return unsupervised_loss

class unsupervised_projection_loss2(nn.Module):
    def __init__(self, feature_model, eps=1e-5, reduction='sum'):
        super(unsupervised_projection_loss2, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.feature_model = feature_model
        self.softmax = nn.Softmax(dim=1)
        self.l1loss = nn.L1Loss(reduction=self.reduction)

    def get_weight(self, fore, back, feature_model, mask):

        c1 = 1 / 2
        c2 = 1

        fore = (fore + 1) / 2
        back = (back + 1) / 2

        with torch.no_grad():
            feat_1 = torch.cat((fore, fore, fore), dim=1)
            feat_1 = feature_model(feat_1)
            feat_2 = torch.cat((back, back, back), dim=1)
            feat_2 = feature_model(feat_2)

            for i in range(len(feat_1)):
                m1 = torch.mean(features_grad(feat_1[i]).pow(2), dim=[1, 2, 3])
                m2 = torch.mean(features_grad(feat_2[i]).pow(2), dim=[1, 2, 3])
                if i == 0:
                    w1 = torch.unsqueeze(m1, dim=-1)
                    w2 = torch.unsqueeze(m2, dim=-1)
                else:
                    w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
                    w2 = torch.cat((w2, torch.unsqueeze(m2, dim=-1)), dim=-1)
            weight_1 = torch.mean(w1, dim=-1) / c1
            weight_2 = torch.mean(w2, dim=-1) / c2
            weight_list = torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
            weight_list = F.softmax(weight_list, dim=-1)

        return weight_list

    def get_R_G_B(self, x):

        img_R = x[:, 0:1, :, :]
        img_G = x[:, 1:2, :, :]
        img_B = x[:, 2:3, :, :]

        return img_R, img_G, img_B

    def forward(self, out, fout, bout, mask):
        num = torch.sum(mask)  # (B, C)

        out = self.normalize_(out)
        fout = self.normalize_(fout)
        bout = self.normalize_(bout)

        fout_0, fout_1, fout_2 = fout[:, 0:1, :, :], fout[:, 1:2, :, :], fout[:, 2:3, :, :]
        bout_0, bout_1, bout_2 = bout[:, 0:1, :, :], bout[:, 1:2, :, :], bout[:, 2:3, :, :]
        R = self.softmax(torch.cat([fout_0, bout_0], dim=1)) * 2.0
        G = self.softmax(torch.cat([fout_1, bout_1], dim=1)) * 2.0
        B = self.softmax(torch.cat([fout_2, bout_2], dim=1)) * 2.0

        fR = torch.pow(fout_0, R[:, 0:1, :, :])
        fG = torch.pow(fout_1, G[:, 0:1, :, :])
        fB = torch.pow(fout_2, B[:, 0:1, :, :])

        f = torch.cat([fR, fG, fB], dim=1)

        bR = torch.pow(bout_0, R[:, 1:2, :, :])
        bG = torch.pow(bout_1, G[:, 1:2, :, :])
        bB = torch.pow(bout_2, B[:, 1:2, :, :])

        b = torch.cat([bR, bG, bB], dim=1)

        gt = f * b
        loss = self.l1loss(out * mask, gt * mask) / num

        #-------------

        fore_R, fore_G, fore_B = self.get_R_G_B(fout)
        back_R, back_G, back_B = self.get_R_G_B(bout)

        R_weight = self.get_weight(fore_R, back_R, self.feature_model, mask)
        G_weight = self.get_weight(fore_G, back_G, self.feature_model, mask)
        B_weight = self.get_weight(fore_B, back_B, self.feature_model, mask)

        out = out + self.eps
        loss = torch.log(out + torch.pow((fout * mask), 4*w1) / out) + torch.log(out + torch.pow(bout, 4*w2) / out)

        return loss





