import logging
log = logging.getLogger(__name__)

from abc import ABC, abstractmethod

import torch.nn as nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MIXIN AUTOENCODER (to mix with nn.Module)
class MixinAE(ABC):
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


    @abstractmethod
    def encode(self, X):
        """Absract method for encoding.
        """
        pass


    @abstractmethod
    def decode(self, X):
        """Absract method for decoding.
        """
        pass


    def forward(self, X):
        """Forward step for each autoencoder.
        """
        X = self.encode(X)
        X = self.decode(X)
        return X



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# V0
