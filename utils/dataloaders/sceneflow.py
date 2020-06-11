from torch.utils.data import Dataset
from PIL import Image
from . import listflowfile as lt
import numpy as np
import random


class SceneflowLoader(Dataset):

    """Sceneflow stereo dataset manager"""

    def __init__(self, dataset, dataset_path, training, validation, transform=None, downsample_training=False):
        """
        Parameters
        ----------
        dataset: str
            Kitti2012 or kitti2015
        dataset_path: str
            Kitti dataset path
        training: bool
            Loads training images
        validation: bool
            Loads validation data
        transform: torchvision.transforms
            Transform to be applied to all pair
        downsample_training: bool
            Downsample during training. Some networks dont't need big images to converge faster

        Returns
        -------
        None
        """
        Dataset.__init__(self)

        self.dataset = dataset
        self.dataset_path = dataset_path
        self.training = training
        self.validation = validation
        self.transform = transform
        self.downsample_training = downsample_training

        # Load list of images
        tr_l, tr_r, tr_l_disp, test_l, test_r, test_l_disp = lt.dataloader(dataset_path)

        self.l_im_paths = []
        self.r_im_paths = []
        self.l_disp_paths = []

        if self.training:
            self.l_im_paths = self.l_im_paths + tr_l
            self.r_im_paths = self.r_im_paths + tr_r
            self.l_disp_paths = self.l_disp_paths + tr_l_disp
        if self.validation:
            self.l_im_paths = self.l_im_paths + test_l
            self.r_im_paths = self.r_im_paths + test_r
            self.l_disp_paths = self.l_disp_paths + test_l_disp

    def disparity_loader(self, path):
        path_prefix = path.split('.')[0]
        # print(path_prefix)
        path1 = path_prefix + '_exception_assign_minus_1.npy'
        path2 = path_prefix + '.npy'
        path3 = path_prefix + '.pfm'
        import os.path as ospath
        if ospath.exists(path1):
            return np.load(path1)
        else:

            # from readpfm import readPFMreadPFM
            from .readpfm import readPFM
            data, _ = readPFM(path3)
            np.save(path2, data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if j - data[i][j] < 0:
                        data[i][j] = -1
            np.save(path1, data)
            return data

    def load_pair(self, idx):
        l_im = Image.open(self.l_im_paths[idx])
        r_im = Image.open(self.r_im_paths[idx])
        l_disp = self.disparity_loader(self.l_disp_paths[idx])
        l_disp = np.ascontiguousarray(l_disp, dtype=np.float32)
        l_disp = np.expand_dims(l_disp, axis=0)

        if self.downsample_training:
            w, h = l_im.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            l_im = l_im.crop((x1, y1, x1 + tw, y1 + th))
            r_im = r_im.crop((x1, y1, x1 + tw, y1 + th))

            l_disp = l_disp[:, y1:y1 + th, x1:x1 + tw]
        return l_im, r_im, l_disp

    def __len__(self):
        return len(self.l_im_paths)

    def __getitem__(self, index):
        l_im, r_im, l_disp = self.load_pair(index)

        if self.transform is not None:
            l_im = self.transform(l_im)
            r_im = self.transform(r_im)

        return l_im, r_im, l_disp
