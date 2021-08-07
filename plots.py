import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def save_fig(fig_path=None):
    if fig_path is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()


def plot_data(image, fig_path=None):
    """Plot the image at the requested path.

    Parameters
    ----------
    image: torch.Tensor
        The tensor containing the image to plot.
    fig_path: str
        The path to the image to save.
    """
    img_tmp = torch.squeeze(image)
    plt.imshow(transforms.ToPILImage()(img_tmp), interpolation="bicubic")

    save_fig(fig_path)


def plot_compare_data(img1, img2, fig_path=None):
    """Plot the comparison between two images.

    Parameters
    ----------
    img1: torch.Tensor
        The tensor containing the first image to plot.
    img2: torch.Tensor
        The tensor containing the second image to plot.
    fig_path: str
        The path to the image to save.
    """
    img1_tmp = torch.squeeze(img1)
    img2_tmp = torch.squeeze(img2)

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(transforms.ToPILImage()(img1_tmp), interpolation="bicubic")

    plt.subplot(1, 2, 2)
    plt.imshow(transforms.ToPILImage()(img2_tmp), interpolation="bicubic")

    save_fig(fig_path)


def plot_loss(list_loss_train, list_loss_val):
    """"""
    plt.figure(figsize=(12, 4))
    plt.plot(list_loss_train, color='royalblue', label='Train')
    plt.plot(list_loss_val, color='red', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
