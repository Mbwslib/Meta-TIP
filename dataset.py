import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from PIL import Image
import torchvision.transforms as transforms


class data_loader(Dataset):
    def __init__(self):
        super(data_loader, self).__init__()

        self.size = 256

        self.fore = os.listdir('./data/pix_fore/')
        self.mask = os.listdir('./data/pix_mask/')
        self.back = os.listdir('./data/sixray_back/')

        self.trans = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        assert len(self.fore) == len(self.mask) == len(self.back)

        return len(self.fore)

    def __getitem__(self, item):
        fore = Image.open('./data/pix_fore/' + self.fore[item]).convert('RGB').resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open('./data/pix_mask/' + self.mask[item]).convert('1').resize((self.size, self.size), Image.BILINEAR)

        back = Image.open('./data/sixray_back/' + self.back[item]).convert('RGB').resize((self.size, self.size), Image.BILINEAR)

        fore_n = self.normalize(self.trans(fore))
        mask_n = self.trans(mask)

        back_n = self.normalize(self.trans(back))

        return fore_n, mask_n, back_n

class data_loader_test(data_loader):

    def __init__(self):
        self.fore = os.listdir('./data/pix_fore/')
        self.mask = os.listdir('./data/pix_mask/')
        self.back = os.listdir('./data/sixray_back/')

    def __getitem__(self, item):
        fore = Image.open('./data/pix_fore/' + self.fore[item]).convert('RGB').resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open('./data/pix_mask/' + self.mask[item]).convert('1').resize((self.size, self.size), Image.BILINEAR)

        back = Image.open('./data/sixray_back/' + self.back[item]).convert('RGB').resize((self.size, self.size), Image.BILINEAR)

        fore_n = self.normalize(self.trans(fore))
        mask_n = self.trans(mask)

        back_n = self.normalize(self.trans(back))

        return fore_n, mask_n, back_n



