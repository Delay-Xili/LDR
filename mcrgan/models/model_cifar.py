import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_mimicry.nets.dcgan import dcgan_base
from mcrgan.default import _C as cfg

from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.nets import sngan
from torch_mimicry.nets.sngan.sngan_128 import SNGANDiscriminator128
from torch_mimicry.nets.sngan.sngan_48 import SNGANDiscriminator48
from torch_mimicry.nets.sngan.sngan_32 import SNGANDiscriminator32


class Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x)


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

    else:
        raise ValueError()

    return netG, netD
