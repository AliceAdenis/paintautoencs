import logging
log = logging.getLogger(__name__)

import os
import numpy as np

import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# OPTIMIZER
class Optimizer():
    """Pytoch optimizer for autoencoder.
    """

    def __init__(self, autoenc, loss_func, optimizer, path_model=None):
        """Instantiate the optimizer.
        """
        log.info('instantiate optimization')
        self.autoenc = autoenc
        self.loss_func = loss_func
        self.optim = optimizer
        self.path_model = path_model

        self.epoch = 0
        if self.path_model:
            log.info('checking previous epoch')
            list_models = [int(f.split('.')[0]) for f in os.listdir(self.path_model) \
                           if f.split('.')[-1] == 'pth']
            if len(list_models) > 0:
                self.epoch = np.max(list_models)
                self._get_state_dict(self.epoch)

        log.info('starting at epoch %s', self.epoch)


    def _get_state_dict(self, epoch):
        self.autoenc.load_state_dict(
            torch.load(os.path.join(self.path_model, str(epoch)+'.pth')))
        self.autoenc.eval()


    def _epoch(self, dataset, batch_size=10):
        """Run one epoch.
        """
        loss_train = []
        for T_train in dataset.load_batches(batch_size=batch_size):

            self.optim.zero_grad()
            decoded = self.autoenc(T_train)
            loss_train_subset = self.loss_func(T_train, decoded)
            loss_train_subset.backward()
            self.optim.step()

            loss_train.append(loss_train_subset)

        loss = np.sum(loss_train)

        log.info('epoch %d - loss train: %.6f' % (self.epoch, loss))
        self.epoch += 1


    def _save_epoch(self):
        log.info('saving epoch %s', self.epoch)
        torch.save(
            self.autoenc.state_dict(),
            os.path.join(self.path_model, str(self.epoch)+'.pth'))



    def optimize(self,
                 dataset,
                 n_epoch=100,
                 batch_size=10,
                 n_save=10,
                 save_fig=None):
        """Run the optimization.
        """
        log.info('start training autoencoder')

        for i in range(n_epoch):
            self._epoch(dataset, batch_size)

            if self.epoch % n_save == (n_save - 1):
                self._save_epoch()
                last_saved = self.epoch

                if save_fig:
                    self.plot_sample(dataset, save_fig, 1)

        if last_saved != self.epoch:
            self._save_epoch()


    def plot_sample(self, dataset, path_result, n_sample=2, shuffle=True):
        log.info("saving sample")
        with torch.no_grad():
            plt.figure(figsize=(6, 3*n_sample))

            for i, img1 in enumerate(
                    dataset.load_batches(batch_size=1, shuffle=shuffle)):
                img2 = self.autoenc(img1)
                img1_tmp = torch.squeeze(img1)
                img2_tmp = torch.squeeze(img2)

                plt.subplot(n_sample, 2, 2 * i + 1)
                plt.imshow(transforms.ToPILImage()(img1_tmp),
                           interpolation="bicubic")

                plt.subplot(n_sample, 2, 2 * i + 2)
                plt.imshow(transforms.ToPILImage()(img2_tmp),
                           interpolation="bicubic")

                if i == n_sample - 1:
                    break

            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(path_result, 'sample'+str(self.epoch)+'.png'))


    def plot_loss(self, dataset, path_result, batch_size=10):
        log.info("calculating loss")
        list_models = sorted([int(f.split('.')[0]) for f in os.listdir(self.path_model) \
                     if f.split('.')[-1] == 'pth'])

        list_loss = []
        for epoch in list_models:
            log.info('loss on epoch %s', epoch)
            self._get_state_dict(epoch)
            with torch.no_grad():
                loss_train = []
                for T_train in dataset.load_batches(batch_size=batch_size):
                    decoded = self.autoenc(T_train)
                    loss_train.append(self.loss_func(T_train, decoded))
                list_loss.append(np.sum(loss_train))

        plt.figure(figsize=(9, 3))
        plt.plot(list_models, list_loss, color='royalblue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(path_result, 'loss.png'))
