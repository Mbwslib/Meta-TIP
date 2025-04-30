import torch
import numpy as np
from model import E2ETIP
from loss import *
from dataset import *
from utils import *
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from vgg import *
from skimage.metrics import structural_similarity
from get_Qabf import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_epoch = 0
epochs = 12
learning_rate = 2e-4
batch_size = 4
print_freq = 5
best_psnr = 0.
checkpoint = None

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_checkpoint(epoch, model, optimizer):

    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             }

    filename = 'ep%03d_checkpoint.pth.tar' % (epoch + 1)
    torch.save(state, './checkpoint/' + filename)

def main():

    global best_psnr, checkpoint, start_epoch

    if checkpoint is None:

        model = E2ETIP()

        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    else:

        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)

    #feature_model = vgg16().cuda()
    #feature_model.load_state_dict(torch.load('vgg16.pth'))

    criterion = [RegionL1Loss(is_fore=False).to(device), RegionL1Loss(is_fore=False).to(device),
                 SimMinLoss().to(device), SimMaxLoss().to(device), unsupervised_projection_loss(feature_model=None).to(device)]


    train_loader = DataLoader(data_loader(), batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(data_loader_test(), batch_size=batch_size, shuffle=True)

    for epoch in range(start_epoch, epochs):

        train(train_loader, model, criterion, optimizer, scheduler, epoch)
        #psnr = validate(test_loader, model, criterion)
        #print("The now learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

        # Check if there was an improvement
        #is_best = psnr > best_psnr
        #best_psnr = max(psnr, best_psnr)
        #if is_best:
        save_checkpoint(epoch, model, optimizer)

def train(train_loader, model, criterion, optimizer, scheduler, epoch):

    model.train()

    losses = list()
    recon1 = list()
    recon2 = list()
    contra1 = list()
    contra2 = list()
    unsuper = list()

    qabf_acc = list()
    #mse_acc = list()
    psnr_acc = list()
    ssim_acc = list()

    for i, (fore_n, mask_n, back_n) in enumerate(train_loader):

        fore_n = fore_n.to(device)
        mask_n = mask_n.to(device)
        back_n = back_n.to(device)

        optimizer.zero_grad()

        fout, bout, fx_f_feats, fx_b_feats, out = model(fore_n, back_n, mask_n)

        recon_loss1 = criterion[0](fout, fore_n, mask_n)
        recon_loss2 = criterion[1](bout, back_n, mask_n)
        contra_loss1 = criterion[2](fx_f_feats, fx_b_feats)
        contra_loss2 = criterion[3](fx_b_feats)
        unsuper_loss = criterion[4](out, fout, bout, mask_n)

        loss = 10 * recon_loss1 + 10 * recon_loss2 + 10 * contra_loss1 + 10 * contra_loss2 + unsuper_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item() * 10)
        recon1.append(recon_loss1.item() * 10)
        recon2.append(recon_loss2.item() * 10)
        contra1.append(contra_loss1.item() * 10)
        contra2.append(contra_loss2.item() * 10)
        unsuper.append(unsuper_loss.item())



        for i_img in range(back_n.size(0)):
            gt, pred, mask = back_n[i_img:i_img+1], out[i_img:i_img+1], mask_n[i_img:i_img+1]
            #mse_score_op = mean_squared_error(tensor2im(pred, is_norm=False), tensor2im(gt, is_norm=False))
            psnr_score_op = peak_signal_noise_ratio(tensor2im(pred, is_norm=False), tensor2im(gt, is_norm=False), data_range=255)

            single_fore = tensor2im(get_canvas(pred, mask), is_norm=False)
            qabf_score_op = compute_Qabf(single_fore, tensor2im(gt, is_norm=False), tensor2im(pred, is_norm=False))
            ssim_score_op = structural_similarity(tensor2im(pred, is_norm=False), tensor2im(gt, is_norm=False), multichannel=True)

            qabf_acc.append(qabf_score_op)
            #mse_acc.append(mse_score_op)
            psnr_acc.append(psnr_score_op)
            ssim_acc.append(ssim_score_op)

        # print status
        if i is not 0 and i % print_freq == 0:
            print('Epoch: [{}]/[{}/{}]\t'
                  'Loss: {:.3f}\t'
                  'f_recon_loss: {:.3f} - b_recon_loss: {:.3f} - fb_contra_loss: {:.3f} - bb_contra_loss: {:.3f} - unsurper_loss: {:.3f}\t'
                  'PSNR_Accuracy: {:.3f} - Qabf_Accuracy: {:.3f} - SSIM_Accuracy: {:.3f}'
                  .format(epoch, i, len(train_loader),
                          sum(losses) / len(losses),
                          sum(recon1) / len(recon1),
                          sum(recon2) / len(recon2),
                          sum(contra1) / len(contra1),
                          sum(contra2) / len(contra2),
                          sum(unsuper) / len(unsuper),
                          sum(psnr_acc) / len(psnr_acc),
                          sum(qabf_acc) / len(qabf_acc),
                          sum(ssim_acc) / len(ssim_acc)))
    scheduler.step()

def validate(test_loader, model, criterion):

    model.eval()

    val_losses = list()
    val_mse_acc = list()
    val_psnr_acc = list()

    with torch.no_grad():
        for i, (fore_n, mask_n, back_n) in enumerate(test_loader):
            fore_n = fore_n.to(device)
            mask_n = mask_n.to(device)
            back_n = back_n.to(device)

            fout, bout, fx_f_feats, fx_b_feats, out = model(fore_n, back_n, mask_n)

            recon_loss1 = criterion[0](fout, fore_n, mask_n)
            recon_loss2 = criterion[1](bout, back_n, mask_n)
            contra_loss1 = criterion[2](fx_f_feats, fx_b_feats)
            contra_loss2 = criterion[3](fx_b_feats)
            unsuper_loss = criterion[4](out, fout, bout, mask_n)

            loss = recon_loss1 + recon_loss2 + 100 * contra_loss1 + 100 * contra_loss2 + unsuper_loss

            val_losses.append(loss.item())

            for i_img in range(back_n.size(0)):
                gt, pred = back_n[i_img:i_img + 1], out[i_img:i_img + 1]
                mse_score_op = mean_squared_error(tensor2im(pred, is_norm=False), tensor2im(gt, is_norm=False))
                psnr_score_op = peak_signal_noise_ratio(tensor2im(pred, is_norm=False), tensor2im(gt, is_norm=False), data_range=255)

                val_mse_acc.append(mse_score_op)
                val_psnr_acc.append(psnr_score_op)

        print('Loss: {:.3f}\t'
              'MSE_Accuracy: {:.3f} - PSNR_Accuracy: {:.3f}'.format(sum(val_losses) / len(val_losses),
                                                                    sum(val_mse_acc) / len(val_mse_acc),
                                                                    sum(val_psnr_acc) / len(val_psnr_acc)))

        print('--------------------------------------------------------')
    return sum(val_psnr_acc) / len(val_psnr_acc)


if __name__ == '__main__':
    setup_seed(19)
    main()