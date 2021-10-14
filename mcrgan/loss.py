import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
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
            Z_ = Z[:, Pi[:, j]]
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
        return -total_loss, (discrimn_loss.item(), compress_loss.item())

    # def compute_discrimn_loss_empirical(self, W):
    #     """Empirical Discriminative Loss."""
    #     p, m = W.shape
    #     I = torch.eye(p).cuda()
    #     scalar = p / (m * self.eps)
    #     logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
    #     return logdet / 2.
    #
    # def compute_compress_loss_empirical(self, W, Pi):
    #     """Empirical Compressive Loss."""
    #     p, m = W.shape
    #     k, _, _ = Pi.shape
    #     I = torch.eye(p).cuda()
    #     compress_loss = 0.
    #     for j in range(k):
    #         trPi = torch.trace(Pi[j]) + 1e-8
    #         scalar = p / (trPi * self.eps)
    #         log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
    #         compress_loss += log_det * trPi / m
    #     return compress_loss / 2.
    #
    # def forward(self, X, Y, num_classes=None):
    #     if num_classes is None:
    #         num_classes = Y.max() + 1
    #     W = X.T
    #     Pi = label_to_membership(Y.numpy(), num_classes)
    #     Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
    #
    #     discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
    #     compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
    #
    #     total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
    #     return (total_loss_empi,
    #             [discrimn_loss_empi, compress_loss_empi])


class MCRGANloss(nn.Module):

    def __init__(self, gam1=1., gam2=1., eps=0.5, numclasses=1000, mode=1):
        super(MCRGANloss, self).__init__()

        self.criterion = MaximalCodingRateReduction(gam1=gam1, gam2=gam2, eps=eps)
        self.num_class = numclasses
        self.train_mode = mode

    def forward(self, Z, Z_bar, real_label):

        if self.train_mode == 1:
            loss_z, _ = self.criterion(Z, real_label, self.num_class)
            loss_h, _ = self.criterion(Z_bar, real_label, self.num_class)
            errD = loss_z + loss_h
            empi = [loss_z, loss_h]
            for i in np.arange(self.num_class):
                new_Z = torch.cat((Z[real_label == i], Z_bar[real_label == i]), 0)
                new_label = torch.cat(
                    (torch.zeros_like(real_label[real_label == i]),
                     torch.ones_like(real_label[real_label == i]))
                )
                loss, _ = self.criterion(new_Z, new_label, 2)
                errD += loss
        elif self.train_mode == 0:
            new_Z = torch.cat((Z, Z_bar), 0)
            new_label = torch.cat((torch.zeros_like(real_label), torch.ones_like(real_label)))
            errD, empi = self.criterion(new_Z, new_label, 2)
        else:
            raise ValueError()

        return errD, empi

