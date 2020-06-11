from torch.utils.data import Dataset
from os.path import join
from PIL import Image
import cv2
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class KittiLoader(Dataset):

    """Kitti 2012 and 2015 stereo dataset manager"""

    def __init__(self, dataset, dataset_path, training, validation, transform=None, all_in_ram=True, downsample_training=False):
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
        all_in_ram: bool
            If true all images are stored in the ram

        Returns
        -------
        None
        """
        # Store dataset info
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.transform = transform
        self.all_in_ram = all_in_ram

        # Set proper range
        lower_limit = 0
        upper_limit = 194
        if self.dataset == 'kitti2015':
            upper_limit = 200
        if validation and not training:
            lower_limit = 160
        if training and not validation:
            upper_limit = 160

        self.images_path = []
        if self.dataset == 'kitti2015':

            # Generate names
            for idx in range(lower_limit, upper_limit):
                l_im_path = join(self.dataset_path, 'training', 'image_2', f'{idx:06d}_10.png')
                r_im_path = join(self.dataset_path, 'training', 'image_3', f'{idx:06d}_10.png')
                d_gt_noc_path = join(self.dataset_path, 'training', 'disp_noc_0', f'{idx:06d}_10.png')
                pair = dict(l_im_path=l_im_path, r_im_path=r_im_path, d_gt_noc_path=d_gt_noc_path)
                self.images_path.append(pair)

        elif self.dataset == 'kitti2012':

            # Generate names
            for idx in range(lower_limit, upper_limit):
                l_im_path = join(self.dataset_path, 'training', 'colored_0', f'{idx:06d}_10.png')
                r_im_path = join(self.dataset_path, 'training', 'colored_1', f'{idx:06d}_10.png')
                d_gt_noc_path = join(self.dataset_path, 'training', 'disp_noc', f'{idx:06d}_10.png')
                pair = dict(l_im_path=l_im_path, r_im_path=r_im_path, d_gt_noc_path=d_gt_noc_path)
                self.images_path.append(pair)

        # Load all images in the RAM
        if self.all_in_ram:
            self.pairs = []
            for p in self.images_path:
                l_im, r_im, d_gt_noc = self.load_pair(p)
                pair = dict()
                pair['l_im'] = l_im
                pair['r_im'] = r_im
                pair['d_gt_noc'] = d_gt_noc
                self.pairs.append(pair)

    def load_pair(self, paths):
        l_im = Image.open(paths['l_im_path'])
        r_im = Image.open(paths['r_im_path'])
        d_gt_noc = cv2.imread(paths['d_gt_noc_path'], cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)
        d_gt_noc = d_gt_noc.astype(np.float32) / 256

        crop_width = 1216
        crop_height = 368

        d_gt_noc = d_gt_noc[d_gt_noc.shape[0] - crop_height:d_gt_noc.shape[0], d_gt_noc.shape[1] - crop_width: d_gt_noc.shape[1]]
        d_gt_noc = np.expand_dims(d_gt_noc, axis=0)

        w, h = l_im.size
        l_im = l_im.crop((w - crop_width, h - crop_height, w, h))
        r_im = r_im.crop((w - crop_width, h - crop_height, w, h))

        return l_im, r_im, d_gt_noc

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        if not self.all_in_ram:
            l_im, r_im, d_gt_noc = self.load_pair(self.images_path[idx])
        else:
            l_im = self.pairs[idx]['l_im']
            r_im = self.pairs[idx]['r_im']
            d_gt_noc = self.pairs[idx]['d_gt_noc']

        if self.transform is not None:
            l_im = self.transform(l_im)
            r_im = self.transform(r_im)

        return l_im, r_im, d_gt_noc
