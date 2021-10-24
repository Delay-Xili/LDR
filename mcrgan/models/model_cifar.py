import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_mimicry.nets.dcgan import dcgan_base
from mcrgan.default import _C as cfg

from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.nets import sngan
from torch_mimicry.nets.sngan.sngan_32 import SNGANDiscriminator32, SNGANGenerator32
from torch_mimicry.modules.resblocks import GBlock as mimicry_GBlock

from .model_resnet import Generator, Discriminator


class Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x)


class GBlock1(mimicry_GBlock):

    def __init__(self, in_channels, out_channels, activation='relu', hidden_channels=None,
                 upsample=False, num_classes=0, spectral_norm=False):
        super(GBlock1, self).__init__(in_channels, out_channels, hidden_channels, upsample, num_classes, spectral_norm)

        if activation == 'relu':
            pass
        elif activation == 'lrelu':
            print("building leaky relu")
            self.activation = nn.LeakyReLU(negative_slope=cfg.MODEL.L_RELU_P)
        else:
            raise ValueError()


class customSNGANGenerator32(SNGANGenerator32):

    def __init__(self, nz=128, ngf=256, activation='relu', **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=4, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock1(self.ngf, self.ngf, activation=activation, upsample=True)
        self.block3 = GBlock1(self.ngf, self.ngf, activation=activation, upsample=True)
        self.block4 = GBlock1(self.ngf, self.ngf, activation=activation, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.c5 = nn.Conv2d(self.ngf, 3, 3, 1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU(True)
        elif activation == 'lrelu':
            print("building leaky relu")
            self.activation = nn.LeakyReLU(negative_slope=cfg.MODEL.L_RELU_P)
        else:
            raise ValueError()

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)


class customSNGANDiscriminator32(SNGANDiscriminator32):

    def __init__(self, nz=128, ndf=128, **kwargs):
        super(customSNGANDiscriminator32, self).__init__(ndf, **kwargs)
        self.nz = nz
        self.l5 = nn.Sequential(SNLinear(self.ndf, nz), Norm())


def weights_init_mnist_model(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=False,
                 upsample=False
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.upsample = upsample

        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        if self.hidden_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            self.identity = nn.Sequential()

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        if self.hidden_channels:
            h = self.identity(h)
        return h


class DBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=False,
                 downsample=False,
                 BN=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.downsample = downsample

        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False)

        if BN:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Sequential()

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        if self.hidden_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.identity = nn.Sequential()

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))

        if self.hidden_channels:
            h = self.identity(h)

        return h


class GeneratorCIFAR(dcgan_base.DCGANBaseGenerator):

    def __init__(self, nz=128, ngf=64, iden=False, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        self.main = nn.Sequential(
            GBlock(nz, ngf * 8, hidden_channels=iden, upsample=False),
            GBlock(ngf * 8, ngf * 4, hidden_channels=iden, upsample=True),
            GBlock(ngf * 4, ngf * 2, hidden_channels=iden, upsample=True),
            nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        if cfg.MODEL.INIT == 'mini_dcgan':
            self.main.apply(weights_init_mnist_model)
        elif cfg.MODEL.INIT == 'kaiming':
            pass
        else:
            raise ValueError

    def forward(self, x):

        return self.main(x.view(x.shape[0], -1, 1, 1))


class DiscriminatorCIFAR(dcgan_base.DCGANBaseDiscriminator):

    def __init__(self, nz=128, ndf=64, iden=False, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        self.nz = nz
        self.main = nn.Sequential(
            DBlock(3, ndf, hidden_channels=iden, downsample=True, BN=False),
            DBlock(ndf, ndf * 2, hidden_channels=iden, downsample=True),
            DBlock(ndf * 2, ndf * 4, hidden_channels=iden, downsample=True),

            nn.Conv2d(ndf * 4, nz, 4, 1, 0, bias=False),
            nn.Flatten()
        )

        if cfg.MODEL.INIT == 'mini_dcgan':
            self.main.apply(weights_init_mnist_model)
        elif cfg.MODEL.INIT == 'kaiming':
            pass
        else:
            raise ValueError

    def forward(self, x):

        return F.normalize(self.main(x))


def get_cifar_model():

    if cfg.MODEL.CIFAR_BACKBONE == 'mini_dcgan':
        print("building the mini_dcgan model...")
        netG = GeneratorCIFAR()
        netD = DiscriminatorCIFAR()
    elif cfg.MODEL.CIFAR_BACKBONE == 'mini_dcgan_double':
        print("building the mini_dcgan_double model...")
        netG = GeneratorCIFAR(iden=True)
        netD = DiscriminatorCIFAR(iden=True)
    elif cfg.MODEL.CIFAR_BACKBONE == 'mimicry_sngan':
        print("building the mimicry_sngan model...")
        netG = sngan.SNGANGenerator32()
        netD = customSNGANDiscriminator32()

    elif cfg.MODEL.CIFAR_BACKBONE == 'lrelu_sngan':
        print("building the lrelu_sngan model...")
        netG = customSNGANGenerator32(activation='lrelu')
        netD = customSNGANDiscriminator32()

    elif cfg.MODEL.CIFAR_BACKBONE == 'work_sngan':
        print("building the work_sngan model...")
        netG = Generator(128)
        netD = Discriminator(128)

    else:
        raise ValueError()

    return netG, netD
