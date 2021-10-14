# from __future__ import print_function
import argparse
import torch.nn as nn
import torch.optim as optim
import pprint

from mcrgan.default import _C as config
from mcrgan.default import update_config
import torch
from mcrgan.trainer import MCRTrainer
from mcrgan.datasets import get_dataloader
from mcrgan.models import get_models


def run_ldr():
    """the default setting is running the binary LDR on different data-sets."""
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    dataloader, dataset = get_dataloader(
        data_name=config.DATA.DATASET,
        root=config.DATA.ROOT,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.CUDNN.WORKERS
    )

    # Define models and optimizers
    netD, netG = get_models(config.DATA.DATASET, device)

    optD = optim.Adam(netD.parameters(), config.TRAIN.LR, betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))
    optG = optim.Adam(netG.parameters(), config.TRAIN.LR, betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))

    # Start training
    trainer = MCRTrainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=config.TRAIN.INNER_LOOP,
        num_steps=config.TRAIN.ITERATION,
        lr_decay=config.TRAIN.LR_DECAY,
        print_steps=config.TRAIN.SHOW_STEPS,
        vis_steps=config.TRAIN.SHOW_STEPS,
        log_steps=config.TRAIN.SHOW_STEPS,
        save_steps=5000,
        dataloader=dataloader,
        log_dir=config.LOG_DIR,
        device=device,
        num_class=config.MODEL.NUM_CLASSES,
        mode=config.LOSS.MODE,
    )
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='should add the .yaml file',
                        required=True,
                        type=str,
                        )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    #
    # return args


if __name__ == '__main__':

    parse_args()
    # pprint.pformat(config)

    run_ldr()
