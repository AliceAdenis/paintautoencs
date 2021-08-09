from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader

from PIL import Image


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# BASE DATASET
class BaseDataset(ABC, Dataset):
    """Base class for datasets.
    """

    @abstractmethod
    def __init__(self):
        """Abstract method to initialize the dataset.
        """
        pass

    @abstractmethod
    def __len__(self):
        """Abstract method to return length of dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, index):
        """Abstract method to return data from index.

        Parameters
        ----------
        index: int
            Index of the data to return.
        """
        pass

    def load_batches(self, batch_size=10, **kwargs):
        """Iterator to load batches.

        Parameters
        ----------
        batch_size: int
            Size of the batches.
        **kwargs
            Other arguments to pass to `torch.utils.data.DataLoader`.
        """
        return DataLoader(self, batch_size=batch_size, **kwargs)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# DATASET PAINTINGS
class DatasetPaintings(BaseDataset):
    """Data Loader for paintings (or for any image).
    """

    def __init__(self, path_list, transform=None):
        """Initialisation of the class.

        Parameters
        ----------
        path_list: list of str
            List that contain the list of images path to be loaded.
        transform: callable (optional)
            Optional transform to be applied on a sample.
        """
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        """Returns the length of the dataset.

        Return
        ------
        int
            The length of the dataset.
        """
        return len(self.path_list)

    def __getitem__(self, index):
        """Load the data corresponding to an index, and apply the
        transforms requested in the initialisation.

        Parameters
        ----------
        index: int
            The index of the requested dataset.

        Return
        ------
        dict
            The data requested by the index. For example, for image classifier:
            ``{'image': image, 'category': category}``
        """

        image = Image.open(self.path_list[index])
        if self.transform:
            image = self.transform(image)

        return image
