from utils.cli.base import CLIBase
from argparse import Namespace
from pl_template import FEStereo
import click
import os

def_dataset_path = os.environ['HOME'] + '/datasets/'


class TrainTestManager(CLIBase):

    """TrainTestManager deals with the training and testing process of the network"""

    model_name = 'festero'

    def __init__(self, hparams):
        model = FEStereo(hparams)
        super().__init__(model, self.model_name, hparams)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--num_workers', default=16, type=int, help="Number of CPUs to use")
@click.option('--shuffle/--no-shuffle', default=True, help="Shuffle while training")
@click.option('--drop_last/--no-drop_last', default=True, help="Drop last batch during training")
@click.option('--dataset', type=click.Choice(['sceneflow', 'kitti2012', 'kitti2015']),
              default='sceneflow')
@click.option('--model_type', type=click.Choice(['default']), default='default')
@click.option('--datasets_path', default=def_dataset_path)
@click.option('--exp_id', default=1, help='Experiment ID')
@click.option('--min_epochs', default=10, help='Minimum number of epochs during training')
@click.option('--max_epochs', default=20, help='Maximun number of epochs during training')
@click.option('--epochs_per_val', default=1, help='Check validation every epochs_per_val')
@click.option('--max_disp', default=192, help='Maximum disparity')
@click.option('--batch_size', default=6, help='Batch size')
@click.option('--seed', default=1, help='Seed')
@click.option('--optimizer', type=click.Choice(['adam', 'adabound']), default='adam')
@click.option('--scheduler', type=click.Choice(['steplr', 'multisteplr', 'plateau']), default='steplr')
@click.option('--lr', default=5e-3, help='Learning rate')
@click.option('--gamma', default=0.1, help='Learning rate step gamma')
@click.option('--gamma_step', default=10, help='Learning rate step')
@click.option('--auto_lr_find/--no-auto_lr_find', default=False, help='Auto find initial lr')
@click.option('--debug/--no-debug', default=False, help='Number of stages')
@click.option('--justtest/--no-justtest', default=False, help='just run test')
@click.option('--resume', default=None, required=False, type=str, help='Checkpoint to resume training')
@click.option('--patience', default=5, help='Early stopping')
@click.option('--save_top_k', default=1, help='Save best k models')
@click.option('--pretrained', default='', help='Pretrained weights path')
def festereo_train(**args):
    hparams = Namespace(**args)

    ttmanager = TrainTestManager(hparams=hparams)

    ttmanager.train()


if __name__ == "__main__":
    cli()
