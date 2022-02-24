import torch
import torch.nn as nn
import torch.nn.functional as F


class MCRGANloss(nn.Module):

    def __init__(self, gam1=1., gam2=1., gam3=1., eps=0.5, numclasses=1000, mode=1, rho=None):
        super(MCRGANloss, self).__init__()

        self.num_class = numclasses
        self.train_mode = mode
        self.faster_logdet = False
        self.gam1 = gam1
        self.gam2 = gam2
        self.gam3 = gam3
        self.eps = eps

    def forward(self, Z, Z_bar, real_label, ith_inner_loop, num_inner_loop):

        # t = time.time()
        # errD, empi = self.old_version(Z, Z_bar, real_label, ith_inner_loop, num_inner_loop)
        errD, empi = self.fast_version(Z, Z_bar, real_label, ith_inner_loop, num_inner_loop)
        # print("faster version time: ", time.time() - t)
        # print("faster errD", errD)

        return errD, empi

    def old_version(self, Z, Z_bar, real_label, ith_inner_loop, num_inner_loop):

        """ original version, need to calculate 52 times log-det"""
        if self.train_mode == 2:
            loss_z, _ = self.deltaR(Z, real_label, self.num_class)
            assert num_inner_loop >= 2
            if (ith_inner_loop + 1) % num_inner_loop != 0:
                return loss_z, None

            loss_h, _ = self.deltaR(Z_bar, real_label, self.num_class)
            errD = self.gam1 * loss_z + self.gam2 * loss_h
            empi = [loss_z, loss_h]
            term3 = 0.

            for i in range(self.num_class):
                new_Z = torch.cat((Z[real_label == i], Z_bar[real_label == i]), 0)
                new_label = torch.cat(
                    (torch.zeros_like(real_label[real_label == i]),
                     torch.ones_like(real_label[real_label == i]))
                )
                loss, _ = self.deltaR(new_Z, new_label, 2)
                term3 += loss
            empi = empi + [term3]
            errD += self.gam3 * term3

        elif self.train_mode == 1:

            loss_z, _ = self.deltaR(Z, real_label, self.num_class)
            loss_h, _ = self.deltaR(Z_bar, real_label, self.num_class)
            errD = self.gam1 * loss_z + self.gam2 * loss_h
            empi = [loss_z, loss_h]
            term3 = 0.

            for i in range(self.num_class):
                new_Z = torch.cat((Z[real_label == i], Z_bar[real_label == i]), 0)
                new_label = torch.cat(
                    (torch.zeros_like(real_label[real_label == i]),
                     torch.ones_like(real_label[real_label == i]))
                )
                loss, _ = self.deltaR(new_Z, new_label, 2)
                term3 += loss
            empi = empi + [term3]
            errD += self.gam3 * term3
        elif self.train_mode == 0:
            new_Z = torch.cat((Z, Z_bar), 0)
            new_label = torch.cat((torch.zeros_like(real_label), torch.ones_like(real_label)))
            errD, em = self.deltaR(new_Z, new_label, 2)
            empi = (em[0], em[1])
        else:
            raise ValueError()

        return errD, empi

    def fast_version(self, Z, Z_bar, real_label, ith_inner_loop, num_inner_loop):

        """ decrease the times of calculate log-det  from 52 to 32"""

        if self.train_mode == 2:
            z_total, (z_discrimn_item, z_compress_item, z_compress_losses, z_scalars) = self.deltaR(Z, real_label,
                                                                                                    self.num_class)
            assert num_inner_loop >= 2
            if (ith_inner_loop + 1) % num_inner_loop != 0:
                # print(f"{ith_inner_loop + 1}/{num_inner_loop}")
                # print("calculate delta R(z)")
                return z_total, None

            zbar_total, (zbar_discrimn_item, zbar_compress_item, zbar_compress_losses, zbar_scalars) = self.deltaR(
                Z_bar, real_label, self.num_class)
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

            empi = empi + [-itemRzjzjbar + 0.25 * sum(z_compress_losses) + 0.25 * sum(zbar_compress_losses)]
            # print("calculate multi")

        elif self.train_mode == 1:
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

            empi = empi + [-itemRzjzjbar + 0.25 * sum(z_compress_losses) + 0.25 * sum(zbar_compress_losses)]

        elif self.train_mode == 0:
            new_Z = torch.cat((Z, Z_bar), 0)
            new_label = torch.cat((torch.zeros_like(real_label), torch.ones_like(real_label)))
            errD, extra = self.deltaR(new_Z, new_label, 2)
            empi = (extra[0], extra[1])

        elif self.train_mode == 10:
            errD, empi = self.double_loop(Z, Z_bar, real_label, ith_inner_loop, num_inner_loop)
        else:
            raise ValueError()

        return errD, empi

    def logdet(self, X):

        if self.faster_logdet:
            return 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(X, upper=True))))
        else:
            return torch.logdet(X)

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = self.logdet(I + scalar * Z @ Z.T)
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
            log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar * Z_ @ Z_.T)
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
