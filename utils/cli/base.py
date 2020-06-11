import torch
import time
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning import loggers


class CLIBase():

    """Base for command line interfaces of implemented models"""

    def __init__(self, model, model_name, hparams):
        """Constructor

        Parameters
        ----------
            model (class): Pytorch Lighting template model
            model_name (str): Name of the model
            hparams (Namespace): Parameters list

        Returns
        -------
            None

        """
        # Set arguments
        self.model = model
        self.model_name = model_name
        self.model_type = hparams.model_type
        self.hparams = hparams

    def train(self):
        """Train the network"""

        hparams = self.hparams
        dataset = hparams.dataset

        exp_id = time.time()
        # Checkpoint callback
        filepath = f'./output/{self.model_name}_{self.model_type}/{dataset}/checkpoints/{exp_id}/'
        checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                              save_top_k=hparams.save_top_k,
                                              verbose=True,
                                              monitor='val_loss',
                                              mode='min',
                                              prefix=f'{hparams.dataset}')

        logger = False
        if not hparams.justtest:
            logger = loggers.TensorBoardLogger(f'/output/{self.model_name}_{self.model_type}/{dataset}/log/')

        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            min_delta=0.00,
                                            patience=hparams.patience,
                                            verbose=True,
                                            mode='min')
        # Set trainer
        trainer = Trainer(gpus=1,
                          train_percent_check=1.0,
                          val_percent_check=1.0,
                          test_percent_check=1.0,
                          overfit_pct=0.01 if hparams.debug else 0.0,
                          check_val_every_n_epoch=hparams.epochs_per_val,
                          min_epochs=hparams.min_epochs,
                          max_epochs=hparams.max_epochs,
                          resume_from_checkpoint=hparams.resume,
                          auto_lr_find=hparams.auto_lr_find,
                          logger=logger,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop_callback)

        if hparams.pretrained != '':
            checkpoint = torch.load(hparams.pretrained)
            self.model.load_state_dict(checkpoint['state_dict'])

        if not hparams.justtest:
            trainer.fit(self.model)
            trainer.test(self.model)
        else:
            trainer.test(self.model)
