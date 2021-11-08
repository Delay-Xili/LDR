import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mimicry.nets.dcgan import dcgan_base
from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.nets import sngan
from torch_mimicry.nets.sngan.sngan_128 import SNGANDiscriminator128
from torch_mimicry.nets.sngan.sngan_48 import SNGANDiscriminator48
from torch_mimicry.nets.sngan.sngan_32 import SNGANDiscriminator32

from .model_cifar import get_cifar_model
from mcrgan.default import _C as cfg


class Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x)


class customSNGANDiscriminator128(SNGANDiscriminator128):

    def __init__(self, nz=128, ndf=1024, **kwargs):
        super(customSNGANDiscriminator128, self).__init__(ndf, **kwargs)
        self.nz = nz
        self.l7 = nn.Sequential(SNLinear(self.ndf, nz), Norm())


class customSNGANDiscriminator48(SNGANDiscriminator48):

    def __init__(self, nz=128, ndf=1024, **kwargs):
        super(customSNGANDiscriminator48, self).__init__(ndf, **kwargs)
        self.nz = nz
        self.l5 = nn.Sequential(SNLinear(self.ndf, nz), Norm())


class customSNGANDiscriminator32(SNGANDiscriminator32):

    def __init__(self, nz=128, ndf=128, **kwargs):
        super(customSNGANDiscriminator32, self).__init__(ndf, **kwargs)
        self.nz = nz
        self.l5 = nn.Sequential(SNLinear(self.ndf, nz), Norm())



class GeneratorMNIST(dcgan_base.DCGANBaseGenerator):
    r"""
    ResNet backbone generator for ResNet DCGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=64, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        # h = h.view(x.shape[0], -1, 1, 1)
        return self.main(x.view(x.shape[0], -1, 1, 1))


class DiscriminatorMNIST(dcgan_base.DCGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for ResNet DCGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ndf=64, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        self.nz = nz
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, nz, 4, 1, 0, bias=False),
            nn.Flatten()  # new
            # nn.LeakyReLU(0.2, inplace=True), #New
            # nn.Linear(ndf, ndf, bias=False)
            # nn.Sigmoid()
        )

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        return F.normalize(self.main(x))


def weights_init_mnist_model(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_models(data_name, device):

    if data_name in ["cifar10", "cifar10_data_aug"]:

        netG, netD = get_cifar_model()
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))
    elif data_name == 'mnist':
        netG = GeneratorMNIST().to(device)
        netG.apply(weights_init_mnist_model)
        netG = nn.DataParallel(netG)

        netD = DiscriminatorMNIST().to(device)
        netD.apply(weights_init_mnist_model)
        netD = nn.DataParallel(netD)
    elif data_name == 'stl10_48':
        netG = sngan.SNGANGenerator48().to(device)
        netG = nn.DataParallel(netG)

        netD = customSNGANDiscriminator48().to(device)
        netD = nn.DataParallel(netD)
    elif data_name in ["celeba", "lsun_bedroom_128", "imagenet_128"]:
        netG = sngan.SNGANGenerator128(ngf=cfg.MODEL.IMAGENET_WIDTH).to(device)
        netG = nn.DataParallel(netG)

        netD = customSNGANDiscriminator128(ndf=cfg.MODEL.IMAGENET_WIDTH).to(device)
        netD = nn.DataParallel(netD)
    else:
        raise ValueError()

    return netD, netG
