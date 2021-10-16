import torch
import torch_mimicry as mmc
import torch.nn as nn
import os
from torch_mimicry.training import scheduler, logger, metric_log
import time
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms.functional as FF
import torchvision.utils as vutils
from .datasets import infiniteloop
from .loss import MCRGANloss
from .default import _C as cfg


class MUltiGPUTrainer(mmc.training.Trainer):

    def __init__(self,
                 netD,
                 netG,
                 optD,
                 optG,
                 dataloader,
                 num_steps,
                 log_dir='./logs',
                 n_dis=1,
                 lr_decay=None,
                 device=None,
                 netG_ckpt_file=None,
                 netD_ckpt_file=None,
                 print_steps=1,
                 vis_steps=500,
                 log_steps=50,
                 save_steps=5000,
                 flush_secs=30,
                 amp=False):

        # Input values checks
        ints_to_check = {
            'num_steps': num_steps,
            'n_dis': n_dis,
            'print_steps': print_steps,
            'vis_steps': vis_steps,
            'log_steps': log_steps,
            'save_steps': save_steps,
            'flush_secs': flush_secs
        }
        for name, var in ints_to_check.items():
            if var < 1:
                raise ValueError('{} must be at least 1 but got {}.'.format(
                    name, var))

        self.netD = netD
        self.netG = netG
        self.optD = optD
        self.optG = optG
        self.n_dis = n_dis
        self.lr_decay = lr_decay
        self.dataloader = dataloader
        self.num_steps = num_steps
        self.device = device
        self.log_dir = log_dir
        self.netG_ckpt_file = netG_ckpt_file
        self.netD_ckpt_file = netD_ckpt_file
        self.print_steps = print_steps
        self.vis_steps = vis_steps
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.amp = amp
        self.parallel = isinstance(self.netG, nn.DataParallel)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Training helper objects
        self.logger = logger.Logger(log_dir=self.log_dir,
                                    num_steps=self.num_steps,
                                    dataset_size=len(self.dataloader),
                                    flush_secs=flush_secs,
                                    device=self.device)

        self.scheduler = scheduler.LRScheduler(lr_decay=self.lr_decay,
                                               optD=self.optD,
                                               optG=self.optG,
                                               num_steps=self.num_steps)

        # Obtain custom or latest checkpoint files
        if self.netG_ckpt_file:
            self.netG_ckpt_dir = os.path.dirname(netG_ckpt_file)
            self.netG_ckpt_file = netG_ckpt_file
        else:
            self.netG_ckpt_dir = os.path.join(self.log_dir, 'checkpoints',
                                              'netG')
            self.netG_ckpt_file = self._get_latest_checkpoint(
                self.netG_ckpt_dir)  # can be None

        if self.netD_ckpt_file:
            self.netD_ckpt_dir = os.path.dirname(netD_ckpt_file)
            self.netD_ckpt_file = netD_ckpt_file
        else:
            self.netD_ckpt_dir = os.path.join(self.log_dir, 'checkpoints',
                                              'netD')
            self.netD_ckpt_file = self._get_latest_checkpoint(
                self.netD_ckpt_dir)

        # Log hyperparameters for experiments
        self.params = {
            'log_dir': self.log_dir,
            'num_steps': self.num_steps,
            'batch_size': self.dataloader.batch_size,
            'n_dis': self.n_dis,
            'lr_decay': self.lr_decay,
            'optD': optD.__repr__(),
            'optG': optG.__repr__(),
            'save_steps': self.save_steps,
        }
        self._log_params(self.params)

        # Device for hosting model and data
        if not self.device:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else "cpu")

    def _save_model_checkpoints(self, global_step):
        """
        Saves both discriminator and generator checkpoints.
        """
        if not self.parallel:
            self.netG.save_checkpoint(directory=self.netG_ckpt_dir,
                                      global_step=global_step,
                                      optimizer=self.optG)

            self.netD.save_checkpoint(directory=self.netD_ckpt_dir,
                                      global_step=global_step,
                                      optimizer=self.optD)
        else:
            self.netG.module.save_checkpoint(directory=self.netG_ckpt_dir,
                                             global_step=global_step,
                                             optimizer=self.optG)

            self.netD.module.save_checkpoint(directory=self.netD_ckpt_dir,
                                             global_step=global_step,
                                             optimizer=self.optD)

    def _restore_models_and_step(self):
        """
        Restores model and optimizer checkpoints and ensures global step is in sync.
        """
        global_step_D = global_step_G = 0

        if self.netD_ckpt_file and os.path.exists(self.netD_ckpt_file):
            print("INFO: Restoring checkpoint for D...")
            global_step_D = self.netD.module.restore_checkpoint(
                ckpt_file=self.netD_ckpt_file, optimizer=self.optD)

        if self.netG_ckpt_file and os.path.exists(self.netG_ckpt_file):
            print("INFO: Restoring checkpoint for G...")
            global_step_G = self.netG.module.restore_checkpoint(
                ckpt_file=self.netG_ckpt_file, optimizer=self.optG)

        if global_step_G != global_step_D:
            raise ValueError('G and D Networks are out of sync.')
        else:
            global_step = global_step_G  # Restores global step

        return global_step

    def train(self):
        """
        Runs the training pipeline with all given parameters in Trainer.
        """
        # Restore models
        global_step = self._restore_models_and_step()
        print("INFO: Starting training from global step {}...".format(
            global_step))

        try:
            start_time = time.time()

            # Iterate through data
            iter_dataloader = iter(self.dataloader)
            while global_step < self.num_steps:
                log_data = metric_log.MetricLog()  # log data for tensorboard

                # -------------------------
                #   One Training Step
                # -------------------------
                # Update n_dis times for D
                for i in range(self.n_dis):
                    iter_dataloader, real_batch = self._fetch_data(
                        iter_dataloader=iter_dataloader)

                    # ------------------------
                    #   Update D Network
                    # -----------------------

                    self.netD.module.zero_grad()
                    real_images, real_labels = real_batch
                    batch_size = real_images.shape[0]  # Match batch sizes for last iter

                    # Produce logits for real images
                    output_real = self.netD(real_images)

                    # Produce fake images
                    noise = torch.randn((batch_size, self.netG.module.nz), device=self.device)
                    fake_images = self.netG(noise).detach()

                    # Produce logits for fake images
                    output_fake = self.netD(fake_images)

                    # Compute loss for D
                    errD = self.netD.module.compute_gan_loss(output_real=output_real,
                                                             output_fake=output_fake)

                    # Backprop and update gradients
                    errD.backward()
                    self.optD.step()

                    # Compute probabilities
                    D_x, D_Gz = self.netD.module.compute_probs(output_real=output_real,
                                                               output_fake=output_fake)

                    # Log statistics for D once out of loop
                    log_data.add_metric('errD', errD.item(), group='loss')
                    log_data.add_metric('D(x)', D_x, group='prob')
                    log_data.add_metric('D(G(z))', D_Gz, group='prob')

                    # -----------------------
                    #   Update G Network
                    # -----------------------
                    # Update G, but only once.
                    if i == (self.n_dis - 1):

                        self.netG.module.zero_grad()

                        # Get only batch size from real batch
                        batch_size = real_batch[0].shape[0]

                        # Produce fake images
                        noise = torch.randn((batch_size, self.netG.module.nz), device=self.device)
                        fake_images = self.netG(noise)

                        # Compute output logit of D thinking image real
                        output = self.netD(fake_images)

                        # Compute loss
                        errG = self.netG.module.compute_gan_loss(output=output)

                        # Backprop and update gradients
                        errG.backward()
                        self.optG.step()

                        # Log statistics
                        log_data.add_metric('errG', errG, group='loss')

                # --------------------------------
                #   Update Training Variables
                # -------------------------------
                log_data = self.scheduler.step(log_data=log_data,
                                               global_step=global_step)

                # -------------------------
                #   Logging and Metrics
                # -------------------------
                if global_step % self.log_steps == 0:
                    self.logger.write_summaries(log_data=log_data,
                                                global_step=global_step)

                if global_step % self.print_steps == 0:
                    curr_time = time.time()
                    self.logger.print_log(global_step=global_step,
                                          log_data=log_data,
                                          time_taken=(curr_time - start_time) /
                                          self.print_steps)
                    start_time = curr_time

                if global_step % self.vis_steps == 0:
                    self.logger.vis_images(netG=self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG,
                                           global_step=global_step)

                if global_step % self.save_steps == 0:
                    print("INFO: Saving checkpoints...")
                    self._save_model_checkpoints(global_step)

                global_step += 1

            print("INFO: Saving final checkpoints...")
            self._save_model_checkpoints(global_step)

        except KeyboardInterrupt:
            print("INFO: Saving checkpoints from keyboard interrupt...")
            self._save_model_checkpoints(global_step)

        finally:
            self.logger.close_writers()

        print("INFO: Training Ended.")


class MCRTrainer(MUltiGPUTrainer):

    def __init__(self,
                 netD,
                 netG,
                 optD,
                 optG,
                 dataloader,
                 num_steps,
                 log_dir='./log',
                 n_dis=1,
                 lr_decay=None,
                 device=None,
                 netG_ckpt_file=None,
                 netD_ckpt_file=None,
                 print_steps=1,
                 vis_steps=500,
                 log_steps=50,
                 save_steps=5000,
                 flush_secs=30,
                 num_class=1000,
                 mode=0):

        super(MCRTrainer, self).__init__(netD, netG, optD, optG, dataloader, num_steps, log_dir, n_dis, lr_decay,
                                         device, netG_ckpt_file, netD_ckpt_file, print_steps, vis_steps, log_steps,
                                         save_steps, flush_secs)

        self.mcr_gan_loss = MCRGANloss(gam1=cfg.LOSS.GAM1, gam2=cfg.LOSS.GAM2, gam3=cfg.LOSS.GAM3, eps=cfg.LOSS.EPS, numclasses=num_class, mode=mode)

    def show(self, imgs, epoch, name):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = FF.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if not os.path.exists(f"{self.log_dir}/images"):
            os.makedirs(f"{self.log_dir}/images")
        plt.savefig(f"{self.log_dir}/images/{epoch:07d}_{name}.png", bbox_inches="tight")
        plt.close()

    def train(self):
        """
                Runs the training pipeline with all given parameters in Trainer.
                """
        # Restore models

        self.parallel = isinstance(self.netG, nn.DataParallel)

        try:
            global_step = self._restore_models_and_step()
            print("INFO: Starting training from global step {}...".format(
                global_step))

            iter_dataloader = infiniteloop(self.dataloader)
            nz = self.netD.module.nz

            start_time = time.time()
            while global_step < self.num_steps:
                log_data = metric_log.MetricLog()  # log data for tensorboard

                data_time = time.time()
                data, label = next(iter_dataloader)
                data_time = time.time() - data_time

                for i in range(self.n_dis):

                    # Update Discriminator
                    self.netD.zero_grad()
                    self.optD.zero_grad()

                    # Format batch and label
                    real_cpu = data.to(self.device)
                    real_label = label.clone().detach()

                    # Forward pass real batch through D(X->Z)
                    Z = self.netD(real_cpu)

                    # Generate batch of latent vectors (Z->X')
                    X_bar = self.netG(torch.reshape(Z, (len(Z), nz)))

                    # Forward pass fake batch through D(X'->Z')
                    Z_bar = self.netD(X_bar.detach())

                    # Optimize Delta R(Z)+deltaR(Z')+sum(delta(R(Z,Z'))) by alternating G/D
                    errD, errD_EC = self.mcr_gan_loss(Z, Z_bar, real_label)

                    errD.backward()
                    self.optD.step()

                # Update Discriminator
                self.netG.zero_grad()
                self.optG.zero_grad()

                # Format batch and label
                real_cpu = data.to(self.device)
                real_label = label.clone().detach()

                # Repeat (X->Z->X'->Z')
                Z = self.netD(real_cpu)
                X_bar = self.netG(torch.reshape(Z, (len(Z), nz)))
                Z_bar = self.netD(X_bar)

                errG, errG_EC = self.mcr_gan_loss(Z, Z_bar, real_label)

                errG = (-1) * errG
                errG.backward()
                self.optG.step()

                log_data.add_metric('errD', -errD.item(), group='discriminator loss')
                log_data.add_metric('errG', -errG.item(), group='generator loss')

                if self.mcr_gan_loss.train_mode == 0:
                    log_data.add_metric('errD_E', -errD_EC[0].item(), group='discriminator loss')
                    log_data.add_metric('errD_C', -errD_EC[1].item(), group='discriminator loss')

                    log_data.add_metric('errG_E', -errG_EC[0].item(), group='generator loss')
                    log_data.add_metric('errG_C', -errG_EC[1].item(), group='generator loss')

                elif self.mcr_gan_loss.train_mode == 1:
                    log_data.add_metric('errD_iterm1', -errD_EC[0].item(), group='discriminator loss')
                    log_data.add_metric('errD_iterm2', -errD_EC[1].item(), group='discriminator loss')
                    log_data.add_metric('errD_iterm3', -errD_EC[2].item(), group='discriminator loss')

                    log_data.add_metric('errG_iterm1', -errG_EC[0].item(), group='generator loss')
                    log_data.add_metric('errG_iterm2', -errG_EC[1].item(), group='generator loss')
                    log_data.add_metric('errG_iterm3', -errG_EC[2].item(), group='generator loss')

                else:
                    raise ValueError()

                log_data = self.scheduler.step(log_data=log_data,
                                               global_step=global_step)

                if global_step % self.print_steps == 0:
                    with torch.no_grad():
                        real = self.netG(torch.reshape(Z[:32], (32, nz))).detach().cpu()
                        self.show(vutils.make_grid(real, padding=2, normalize=True), global_step, "transcript")
                        self.show(vutils.make_grid(real_cpu[:32], padding=2, normalize=True), global_step, "input")

                # -------------------------
                #   Logging and Metrics
                # -------------------------
                if global_step % self.log_steps == 0:
                    self.logger.write_summaries(log_data=log_data,
                                                global_step=global_step)

                if global_step % self.print_steps == 0:
                    curr_time = time.time()
                    self.logger.print_log(global_step=global_step,
                                          log_data=log_data,
                                          time_taken=(curr_time - start_time) /
                                                     self.print_steps)
                    print("data load time: ", data_time)
                    print(f"[{global_step % len(self.dataloader)}/{len(self.dataloader)}]")
                    start_time = curr_time

                if global_step % self.vis_steps == 0:
                    self.logger.vis_images(netG=self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG,
                                           global_step=global_step)

                if global_step % self.save_steps == 0:
                    print("INFO: Saving checkpoints...")
                    self._save_model_checkpoints(global_step)

                global_step += 1

            print("INFO: Saving final checkpoints...")
            self._save_model_checkpoints(global_step)

        except KeyboardInterrupt:
            print("INFO: Saving checkpoints from keyboard interrupt...")
            self._save_model_checkpoints(global_step)

        finally:
            self.logger.close_writers()

        print("INFO: Training Ended.")