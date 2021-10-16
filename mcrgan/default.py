# Modified based on the MDEQ repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.LOG_DIR = ''
# _C.GPUS = (0,)

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True
_C.CUDNN.WORKERS = 16

# dataset
_C.DATA = CN()
_C.DATA.ROOT = ''
_C.DATA.DATASET = ''
_C.DATA.IMAGE_SIZE = [32, 32]
_C.DATA.NC = 3

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.CIFAR_BACKBONE = ''
_C.MODEL.INIT = ''
# _C.MODEL.NZ = 100  # Size of z latent vector (i.e. size of generator input)
# _C.MODEL.NGF = 64  # Size of feature maps in generator
# _C.MODEL.NDF = 64  # Size of feature maps in discriminator

# loss
_C.LOSS = CN()
_C.LOSS.MODE = 0  # 0 for LDR-binary, 1 for LDR multi
_C.LOSS.GAM1 = 1.
_C.LOSS.GAM2 = 1.
_C.LOSS.GAM3 = 1.
_C.LOSS.EPS = 0.5

# training
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.LR = 0.00015
_C.TRAIN.BETA1 = 0.0  # Beta1 hyperparam for Adam optimizers
_C.TRAIN.BETA2 = 0.9  # Beta2 hyperparam for Adam optimizers
_C.TRAIN.ITERATION = 450000  # number of total iterations
_C.TRAIN.INNER_LOOP = 1
_C.TRAIN.LR_DECAY = 'linear'
_C.TRAIN.SHOW_STEPS = 100

# evaluation
_C.EVAL = CN()
_C.EVAL.DATA_SAMPLE = 50000
_C.EVAL.NETD_CKPT = ''
_C.EVAL.NETG_CKPT = ''


def update_config(cfg, args):
    cfg.defrost()
    if args.cfg:
        cfg.merge_from_file(args.cfg)

    # if args.testModel:
    #     cfg.TEST.MODEL_FILE = args.testModel

    cfg.merge_from_list(args.opts)

    cfg.freeze()
