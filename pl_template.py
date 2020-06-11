import torch
from utils.pl.pl_base import PLBase
from torchvision import transforms
from model.default import DefaultModel
from adabound import AdaBound


class FEStereo(PLBase):

    # -------------------------------------------------------------------
    # Training details - Network definition
    # -------------------------------------------------------------------
    def __init__(self, hparams):
        """ Constructor

        Params
        ------
        hparams:
            Contains the training configuration details
        """
        super().__init__(hparams)

        # Prepare network
        if self.hparams.model_type == 'default':
            self.net = DefaultModel(self.hparams.max_disp)

        # Additional log
        self.avg_train_loss = None

    # -------------------------------------------------------------------
    #  Get image transform
    # -------------------------------------------------------------------
    def get_transform(self):
        # Prepare transform
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
        return transform

    # -------------------------------------------------------------------
    # Training details - Forward
    # -------------------------------------------------------------------
    def forward(self, left, right):
        return self.net.forward(left, right)

    # -------------------------------------------------------------------
    # Training details - Train step
    # -------------------------------------------------------------------
    def training_step(self, batch, batch_nb):
        im_left, im_right, disp_l = batch
        output = self.forward(im_left, im_right)

        if self.hparams.dataset == 'sceneflow':
            mask = disp_l < self.hparams.max_disp
        else:
            mask = disp_l > 0
        mask.detach_()

        loss = torch.nn.functional.smooth_l1_loss(output[mask], disp_l[mask], size_average=True)

        if batch_nb == 0:
            self.avg_train_loss = 0

        self.avg_train_loss += loss.item()

        # Loss per stage to print in progresss bar
        pb_dict = dict()
        pb_dict[f'avg_train_loss'] = f'({self.avg_train_loss/(batch_nb+1):4f})'

        return {'loss': loss,
                'progress_bar': pb_dict,
                'log': {'loss': loss}}

    # -------------------------------------------------------------------
    # Training details - Validation step
    # -------------------------------------------------------------------
    def validation_step(self, batch, batch_nb):
        im_left, im_right, disp_l = batch
        output = self.forward(im_left, im_right)

        epe = self.compute_epe(disp_l, output, max_disp=self.hparams.max_disp)
        err3 = self.compute_err(disp_l, output, max_disp=self.hparams.max_disp, tau=3)

        return {'epe': epe, 'err3': err3}

    # -------------------------------------------------------------------
    # Training details - Validation ends
    # -------------------------------------------------------------------
    def validation_epoch_end(self, outputs):
        avg_epe = torch.stack([x['epe'] for x in outputs]).mean()
        avg_err3 = torch.stack([x['err3'] for x in outputs]).mean()
        return {'val_loss': avg_epe,
                'progress_bar': {'val_loss': avg_epe},
                'log': {'val_epe': avg_epe, 'val_err3': avg_err3}}

    # -------------------------------------------------------------------
    # Training details - Optimizer
    # -------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.hparams.lr
        gamma_step = self.hparams.gamma_step
        gamma = self.hparams.gamma
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
            if self.hparams.scheduler == 'steplr':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=gamma_step, gamma=gamma)
            elif self.hparams.scheduler == 'multisteplr':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(200, 300)), gamma=gamma)
            elif self.hparams.scheduler == 'plateau':
                return [optimizer], [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=25)]
            return [optimizer], [scheduler]
        elif self.hparams.optimizer == 'adabound':
            optimizer = AdaBound(self.parameters(), lr=lr, final_lr=0.1)
            return optimizer

    # -------------------------------------------------------------------
    # Test details - Test step
    # -------------------------------------------------------------------
    def test_step(self, batch, batch_nb):
        im_left, im_right, disp_l = batch
        output = self.forward(im_left, im_right)

        epe = self.compute_epe(disp_l, output, max_disp=self.hparams.max_disp)
        err2 = self.compute_err(disp_l, output, tau=2)
        err3 = self.compute_err(disp_l, output, tau=3)
        err4 = self.compute_err(disp_l, output, tau=4)
        err5 = self.compute_err(disp_l, output, tau=5)

        return {'epe': epe, 'err2': err2, 'err3': err3, 'err4': err4, 'err5': err5}

    # -------------------------------------------------------------------
    # Test details - Test ends
    # -------------------------------------------------------------------
    def test_epoch_end(self, outputs):
        avg_epe = torch.stack([x['epe'] for x in outputs]).mean()
        avg_err2 = torch.stack([x['err2'] for x in outputs]).mean()
        avg_err3 = torch.stack([x['err3'] for x in outputs]).mean()
        avg_err4 = torch.stack([x['err4'] for x in outputs]).mean()
        avg_err5 = torch.stack([x['err5'] for x in outputs]).mean()

        return {'test_err': avg_epe,
                'progress_bar': {'test_err': avg_epe},
                'log': {'test_err2': avg_err2, 'test_err3': avg_err3, 'test_err4': avg_err4,
                        'test_err5': avg_err5, 'test_epe': avg_epe}}
