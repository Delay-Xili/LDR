import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.5):
        super(MaximalCodingRateReduction, self).__init__()
        # self.gam1 = gam1
        # self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = torch.logdet(I + scalar * Z @ Z.T)
        return logdet / 2.

    def compute_compress_loss(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = 0.
        for j in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * Z_ @ Z_.T)
            compress_loss += trPi / (2 * n) * log_det
        return compress_loss

    def forward(self, Z, Y, num_classes):

        Pi = F.one_hot(Y, num_classes).to(Z.device)
        discrimn_loss = self.compute_discrimn_loss(Z.T)
        compress_loss = self.compute_compress_loss(Z.T, Pi)
        total_loss = discrimn_loss - compress_loss
        return -total_loss, (discrimn_loss, compress_loss)


class MCRGANloss(nn.Module):

    def __init__(self, gam1=1., gam2=1., gam3=1., eps=0.5, numclasses=1000, mode=1):
        super(MCRGANloss, self).__init__()

        self.criterion = MaximalCodingRateReduction(eps=eps)
        self.num_class = numclasses
        self.train_mode = mode
        self.gam1 = gam1
        self.gam2 = gam2
        self.gam3 = gam3
        self.eps = eps

    def forward(self, Z, Z_bar, real_label):

        # t = time.time()
        # errD, empi = self.old_version(Z, Z_bar, real_label)
        # print("old version time: ", time.time() - t)

        # t = time.time()
        errD, empi = self.fast_version(Z, Z_bar, real_label)
        # print("new version time: ", time.time() - t)

        # print("errDf = ", errDf)
        # print("errD =", errD)
        #
        # print("empi :", empi)
        # print("empif :", empif)

        # self.debug(Z, Z_bar, real_label)

        return errD, empi

    def old_version(self, Z, Z_bar, real_label):

        if self.train_mode == 1:
            loss_z, _ = self.criterion(Z, real_label, self.num_class)
            loss_h, _ = self.criterion(Z_bar, real_label, self.num_class)
            errD = self.gam1 * loss_z + self.gam2 * loss_h
            empi = [loss_z, loss_h]
            term3 = 0.
            for i in range(self.num_class):
                new_Z = torch.cat((Z[real_label == i], Z_bar[real_label == i]), 0)
                new_label = torch.cat(
                    (torch.zeros_like(real_label[real_label == i]),
                     torch.ones_like(real_label[real_label == i]))
                )
                loss, em = self.criterion(new_Z, new_label, 2)
                term3 += loss
            empi = empi + [term3]
            errD += self.gam3 * term3
        elif self.train_mode == 0:
            new_Z = torch.cat((Z, Z_bar), 0)
            new_label = torch.cat((torch.zeros_like(real_label), torch.ones_like(real_label)))
            errD, empi = self.criterion(new_Z, new_label, 2)
        else:
            raise ValueError()

        return errD, empi

    def fast_version(self, Z, Z_bar, real_label):

        """ decrease the times of calculate log-det  from 52 to 32"""

        if self.train_mode == 1:
            z_total, (z_discrimn_item, z_compress_item, z_compress_losses, z_scalars) = self.deltaR(Z, real_label, self.num_class)
            zbar_total, (zbar_discrimn_item, zbar_compress_item, zbar_compress_losses, zbar_scalars) = self.deltaR(Z_bar, real_label, self.num_class)
            empi = [z_total, zbar_total]

            itemRzjzjbar = 0.
            for j in range(self.num_class):
                new_z = torch.cat((Z[real_label == j], Z_bar[real_label == j]), 0)
                R_zjzjbar = self.compute_discrimn_loss(new_z.T)
                itemRzjzjbar += R_zjzjbar

            errD_ = self.gam1 * (z_discrimn_item - z_compress_item) + \
                    self.gam2 * (zbar_discrimn_item - zbar_compress_item) + \
                    self.gam3 * (itemRzjzjbar - 0.25 * sum(z_compress_losses) - 0.25 * sum(zbar_compress_losses))
            errD = -errD_

            empi = empi + [itemRzjzjbar - 0.25 * sum(z_compress_losses) - 0.25 * sum(zbar_compress_losses)]

        elif self.train_mode == 0:
            new_Z = torch.cat((Z, Z_bar), 0)
            new_label = torch.cat((torch.zeros_like(real_label), torch.ones_like(real_label)))
            errD, empi = self.criterion(new_Z, new_label, 2)
        else:
            raise ValueError()

        return errD, empi

    def debug(self, Z, Z_bar, real_label):

        print("===========================")

        Pi = F.one_hot(real_label, self.num_class).to(Z.device)
        z_compress, z_scalars = self.compute_compress_loss(Z.T, Pi)
        z_bar_compress, z_bar_scalars = self.compute_compress_loss(Z_bar.T, Pi)

        print("z compress", z_compress)
        print("z_bar compress", z_bar_compress)

        for i in range(self.num_class):

            new_Z = torch.cat(
                (Z[real_label == i], Z_bar[real_label == i])
                , 0)
            new_label = torch.cat(
                (torch.zeros_like(real_label[real_label == i]),
                 torch.ones_like(real_label[real_label == i]))
            )
            Pi_ = F.one_hot(new_label, 2).to(Z.device)

            zzhat, zzhat_scalars = self.compute_compress_loss(new_Z.T, Pi_)
            print("z and z_bar compress", zzhat)

        print("===========================")

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = torch.logdet(I + scalar * Z @ Z.T)
        return logdet / 2.

    def compute_compress_loss(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = []
        scalars = []
        for j in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * Z_ @ Z_.T)
            compress_loss.append(log_det)
            scalars.append(trPi / (2 * n))
        return compress_loss, scalars

    def deltaR(self, Z, Y, num_classes):

        Pi = F.one_hot(Y, num_classes).to(Z.device)
        discrimn_loss = self.compute_discrimn_loss(Z.T)
        compress_loss, scalars = self.compute_compress_loss(Z.T, Pi)

        compress_term = 0.
        for z, s in zip(compress_loss, scalars):
            compress_term += s * z
        total_loss = discrimn_loss - compress_term

        return -total_loss, (discrimn_loss, compress_term, compress_loss, scalars)
