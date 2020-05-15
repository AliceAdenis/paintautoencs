import logging
log = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MIXIN AUTOENCODER (to mix with nn.Module)
class MixinAE():
    """Mixin class for autoencoders.
    """

    def _info(self, name, X, get_size=True):
        """Log the shape information about the current tensor.

        name: str
            Id of the step.
        X: torch.Tensor
            The tensor to analyse.
        get_size: bool
            If `True` the total number of parameters will be computed.
        """
        if get_size:
            log.info('%s: %s - %s', name, X.shape, self._size(X))
        else:
            log.info('%s: %s', name, X.shape)


    def _size(self, X):
        """Returns the total number of parameters in the given tensor.

        Parameters
        ----------
        X: torch.Tensor
            The tensor to analyse.

        Returns
        -------
        int
            The total number of parameters in the tensor.
        """
        s = 1
        for v in X.shape[1:]:
            s = s*v
        return s

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# V0
class AEPaintingsV0(nn.Module, MixinAE):
    """V0 Autoencoder designed for my paintings.
    """

    def __init__(self, image_size, encoded_size=100):
        super().__init__()
        log.info(
            "initialize autoenc with image size %s and encoded size %s",
            image_size,
            encoded_size)

        self.s0 = (1, 1, 3, image_size, image_size)
        X = torch.rand(self.s0)
        self._info('init', X)

        # Encode
        self.conv1 = nn.Conv3d(1, 3, (3, 5, 5), stride=(1, 3, 3), padding=1)
        X = self.conv1(X)
        self._info('conv1', X)

        self.conv2 = nn.Conv3d(3, 6, (3, 5, 5), stride=(1, 3, 3), padding=1)
        X = self.conv2(X)
        self._info('conv2', X)

        self.conv3 = nn.Conv3d(6, 9, 3, stride=2)
        X = self.conv3(X)
        self._info('conv3', X)

        self.s3_0 = X.shape
        s3_1 = self._size(X)
        X = X.view((X.shape[0], -1))

        s_4 = int((s3_1 + encoded_size) / 3)
        self.elin4 = nn.Linear(s3_1, s_4)
        X = self.elin4(X)
        self._info('elin4', X, False)

        self.elin5 = nn.Linear(s_4, encoded_size)
        X = self.elin5(X)
        self._info('elin5', X, False)

        # Decode
        self.dlin5 = nn.Linear(encoded_size, s_4)
        X = self.dlin5(X)
        self._info('dlin5', X, False)

        self.dlin4 = nn.Linear(s_4, s3_1)
        X = self.dlin4(X)
        self._info('dlin4', X, False)

        X = X.view(self.s3_0)

        self.deconv3 = nn.ConvTranspose3d(9, 6, 3, stride=2)
        X = self.deconv3(X)
        self._info('deconv3', X)

        self.deconv2 = nn.ConvTranspose3d(6, 3, (3, 5, 5), stride=(1, 3, 3), padding=1)
        X = self.deconv2(X)
        self._info('deconv2', X)

        self.deconv1 = nn.ConvTranspose3d(3, 1, (3, 5, 5), stride=(1, 3, 3), padding=1)
        X = self.deconv1(X)
        self._info('deconv1', X)

        X = X[:self.s0[0], :self.s0[1], :self.s0[2], :self.s0[3], :self.s0[4]]
        self._info('final', X)


    def encode(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = X.view((X.shape[0], -1))
        X = F.relu(self.elin4(X))
        X = F.relu(self.elin5(X))
        return X


    def decode(self, X):
        X = F.relu(self.dlin5(X))
        X = F.relu(self.dlin4(X))
        X = X.view(X.shape[0], self.s3_0[1], self.s3_0[2], self.s3_0[3], self.s3_0[4])
        X = F.relu(self.deconv3(X))
        X = F.relu(self.deconv2(X))
        X = torch.sigmoid(self.deconv1(X))
        X = X[:, :self.s0[1], :self.s0[2], :self.s0[3], :self.s0[4]]
        return X


    def forward(self, X):
        """Forward step for each autoencoder.
        """
        X = self.encode(X)
        X = self.decode(X)
        return X

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# V1
class AEPortraitV1(nn.Module, MixinAE):
    """V0 Autoencoder designed for my paintings.
    """

    def __init__(self, image_size, encoded_size=100):
        super().__init__()
        log.info(
            "initialize autoenc with image size %s and encoded size %s",
            image_size,
            encoded_size)

        self.s0 = (1, 1, 3, image_size, image_size)
        X = torch.rand(self.s0)
        self._info('init', X)

        # Encode
        self.conv1 = nn.Conv3d(1, 3, (3, 5, 5), stride=1)
        X = self.conv1(X)
        X = torch.squeeze(X, 2)
        self._info('conv1', X)

        self.conv2 = nn.Conv2d(3, 6, 3, stride=2)
        X = self.conv2(X)
        self._info('conv2', X)

        self.conv2_5 = nn.MaxPool2d(2, return_indices=True)
        X, self.ind_pool = self.conv2_5(X)
        self._info('conv2_5', X)

        self.conv3 = nn.Conv2d(6, 9, 3, stride=2)
        X = self.conv3(X)
        self._info('conv3', X)

        self.s3_0 = X.shape
        s3_1 = self._size(X)
        X = X.view((X.shape[0], -1))

        s_4 = int((s3_1 + encoded_size) / 3)
        self.elin4 = nn.Linear(s3_1, s_4)
        X = self.elin4(X)
        self._info('elin4', X, False)

        self.elin5 = nn.Linear(s_4, encoded_size)
        X = self.elin5(X)
        self._info('elin5', X, False)

        # Decode
        self.dlin5 = nn.Linear(encoded_size, s_4)
        X = self.dlin5(X)
        self._info('dlin5', X, False)

        self.dlin4 = nn.Linear(s_4, s3_1)
        X = self.dlin4(X)
        self._info('dlin4', X, False)

        X = X.view(self.s3_0)

        self.deconv3 = nn.ConvTranspose2d(9, 6, 3, stride=2)
        X = self.deconv3(X)
        self._info('deconv3', X)

        self.unpool2_5 = nn.MaxUnpool2d(2)
        X = self.unpool2_5(X, self.ind_pool)
        self._info('unpool2_5', X)

        self.deconv2 = nn.ConvTranspose2d(6, 3, 3, stride=2)
        X = self.deconv2(X)
        X = torch.unsqueeze(X, 2)
        self._info('deconv2', X)

        self.deconv1 = nn.ConvTranspose3d(3, 1, (3, 5, 5), stride=1)
        X = self.deconv1(X)
        self._info('deconv1', X)

        X = X[:self.s0[0], :self.s0[1], :self.s0[2], :self.s0[3], :self.s0[4]]
        self._info('final', X)


    def encode(self, X):
        X = F.relu(self.conv1(X))
        X = torch.squeeze(X, 2)
        X = F.relu(self.conv2(X))
        X, self.ind_pool = self.conv2_5(X)
        X = F.relu(self.conv3(X))
        X = X.view((X.shape[0], -1))
        X = F.relu(self.elin4(X))
        X = F.relu(self.elin5(X))
        return X


    def decode(self, X):
        X = F.relu(self.dlin5(X))
        X = F.relu(self.dlin4(X))
        X = X.view(X.shape[0], self.s3_0[1], self.s3_0[2], self.s3_0[3])
        X = F.relu(self.deconv3(X))
        X = self.unpool2_5(X, self.ind_pool)
        X = F.relu(self.deconv2(X))
        X = torch.unsqueeze(X, 2)
        X = torch.sigmoid(self.deconv1(X))
        X = X[:, :self.s0[1], :self.s0[2], :self.s0[3], :self.s0[4]]
        return X


    def forward(self, X):
        """Forward step for each autoencoder.
        """
        X = self.encode(X)
        X = self.decode(X)
        return X

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# V2
class AEPortraitV2(nn.Module, MixinAE):
    """V0 Autoencoder designed for my paintings.
    """

    def __init__(self, image_size, encoded_size=100, level=None):
        super().__init__()
        log.info(
            "initialize autoenc with image size %s and encoded size %s",
            image_size,
            encoded_size)

        self.level = level
        if not level:
            self.level = 1
        self.max_level = 5

        self.s0 = (1, 1, 3, image_size, image_size)
        X = torch.rand(self.s0)
        self._info('init', X)

        # Encode
        self.conv1 = nn.Conv3d(1, 3, (3, 5, 5), stride=1, padding=1)
        X = self.conv1(X)
        self._info('conv1', X)

        self.conv2 = nn.Conv3d(3, 6, 3, stride=(1, 2, 2), padding=1)
        X = self.conv2(X)
        self._info('conv2', X)

        self.conv2_5 = nn.MaxPool3d((1, 2, 2), return_indices=True)
        X, self.ind_pool = self.conv2_5(X)
        self._info('pool2_5', X)

        self.conv3 = nn.Conv3d(6, 9, 3, stride=(1, 2, 2), padding=1)
        X = self.conv3(X)
        self._info('conv3', X)

        self.s3_0 = X.shape
        s3_1 = self._size(X)
        X = X.view((X.shape[0], -1))

        s_4 = int((s3_1 + encoded_size) / 3)
        self.elin4 = nn.Linear(s3_1, s_4)
        X = self.elin4(X)
        self._info('elin4', X, False)

        self.elin5 = nn.Linear(s_4, encoded_size)
        X = self.elin5(X)
        self._info('elin5', X, False)

        # Decode
        self.dlin5 = nn.Linear(encoded_size, s_4)
        X = self.dlin5(X)
        self._info('dlin5', X, False)

        self.dlin4 = nn.Linear(s_4, s3_1)
        X = self.dlin4(X)
        self._info('dlin4', X, False)

        X = X.view(self.s3_0)

        self.deconv3 = nn.ConvTranspose3d(9, 6, 3, stride=(1, 2, 2), padding=1)
        X = self.deconv3(X)
        self._info('deconv3', X)

        self.unpool2_5 = nn.MaxUnpool3d((1, 2, 2))
        X = self.unpool2_5(X, self.ind_pool)
        self._info('unpool2_5', X)

        self.deconv2 = nn.ConvTranspose3d(6, 3, 3, stride=(1, 2, 2), padding=1)
        X = self.deconv2(X)
        self._info('deconv2', X)

        self.deconv1 = nn.ConvTranspose3d(3, 1, (3, 5, 5), stride=1, padding=1)
        X = self.deconv1(X)
        self._info('deconv1', X)

        X = X[:self.s0[0], :self.s0[1], :self.s0[2], :self.s0[3], :self.s0[4]]
        self._info('final', X)

    def add_level(self):
        if self.level < self.max_level:
            self.level += 1

    def encode(self, X):
        X = F.relu(self.conv1(X))

        if self.level > 1:
            X = F.relu(self.conv2(X))
            X, self.ind_pool = self.conv2_5(X)

        if self.level > 2:
            X = F.relu(self.conv3(X))
            X = X.view((X.shape[0], -1))

        if self.level > 3:
            X = F.relu(self.elin4(X))

        if self.level > 4:
            X = F.relu(self.elin5(X))

        return X


    def decode(self, X):

        if self.level > 4:
            X = F.relu(self.dlin5(X))

        if self.level > 3:
            X = F.relu(self.dlin4(X))

        if self.level > 2:
            X = X.view(X.shape[0], self.s3_0[1], self.s3_0[2], self.s3_0[3])
            X = F.relu(self.deconv3(X))

        if self.level > 1:
            X = self.unpool2_5(X, self.ind_pool)
            X = F.relu(self.deconv2(X))

        X = torch.sigmoid(self.deconv1(X))
        X = X[:, :self.s0[1], :self.s0[2], :self.s0[3], :self.s0[4]]
        return X


    def forward(self, X):
        """Forward step for each autoencoder.
        """
        X = self.encode(X)
        X = self.decode(X)
        return X
