from os.path import join
from abc import abstractmethod
from utils.dataloaders.kitti import KittiLoader
from utils.dataloaders.sceneflow import SceneflowLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.trainer.seed import seed_everything
import torch
from utils.metrics import metrics


class PLBase(pl.LightningModule):

    """Pytorch Lighting base template"""

    # -------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------
    def __init__(self, hparams):
        super().__init__()
        seed_everything(hparams.seed)
        self.hparams = hparams
        self.prepare_datasets()

    # -------------------------------------------------------------------
    # Get torchvision transform for the dataset
    # -------------------------------------------------------------------
    @abstractmethod
    def get_transform(self):
        pass

    # -------------------------------------------------------------------
    #  Compute EPE
    # -------------------------------------------------------------------
    def compute_epe(self, d_gt, d_est, max_disp=192):
        return metrics.compute_epe(d_gt, d_est, max_disp)

    # -------------------------------------------------------------------
    #  Compute Err3
    # -------------------------------------------------------------------
    def compute_err(self, d_gt, d_est, tau, max_disp=192):
        return metrics.compute_err(d_gt, d_est, tau, max_disp)

    # -------------------------------------------------------------------
    #  Prepare stereo dataset
    # -------------------------------------------------------------------
    def prepare_datasets(self):
        transform = self.get_transform()
        loader = KittiLoader
        if self.hparams.dataset == 'sceneflow':
            loader = SceneflowLoader

        dataset_path = join(self.hparams.datasets_path, self.hparams.dataset)

        self.full_train_loader = loader(dataset=self.hparams.dataset,
                                        dataset_path=dataset_path,
                                        training=True,
                                        validation=False,
                                        transform=transform,
                                        downsample_training=True)
        self.test_dataset = loader(dataset=self.hparams.dataset,
                                   dataset_path=dataset_path,
                                   training=False,
                                   validation=True,
                                   transform=transform)

        train_size = int(len(self.full_train_loader) * 0.9)
        lengths = [train_size, len(self.full_train_loader) - train_size]
        self.train_dataset, self.val_dataset = random_split(self.full_train_loader, lengths)

    # -------------------------------------------------------------------
    # PL dataloaders
    # -------------------------------------------------------------------
    @pl.data_loader  # Decorator used only when data doesn't change
    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(self.train_dataset,
                                             batch_size=self.hparams.batch_size,
                                             shuffle=self.hparams.shuffle,
                                             num_workers=self.hparams.num_workers,
                                             drop_last=self.hparams.drop_last)
        return loader

    @pl.data_loader  # Decorator used only when data doesn't change
    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(self.val_dataset,
                                             batch_size=self.hparams.batch_size,
                                             shuffle=False,
                                             num_workers=self.hparams.num_workers,
                                             drop_last=False)
        return loader

    @pl.data_loader  # Decorator used only when data doesn't change
    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(self.test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=self.hparams.num_workers,
                                             drop_last=False)
        return loader
