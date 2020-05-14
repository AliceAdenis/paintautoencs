import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def plot_data(image):
    img_tmp = torch.squeeze(image)
    plt.imshow(transforms.ToPILImage()(img_tmp), interpolation="bicubic")
    plt.show()


def plot_compare_data(img1, img2):
    img1_tmp = torch.squeeze(img1)
    img2_tmp = torch.squeeze(img2)

    plt.figure(figsize=(6,3))

    plt.subplot(1, 2, 1)
    plt.imshow(transforms.ToPILImage()(img1_tmp), interpolation="bicubic")

    plt.subplot(1, 2, 2)
    plt.imshow(transforms.ToPILImage()(img2_tmp), interpolation="bicubic")

    plt.show()


def plot_loss(list_loss_train, list_loss_val):
    plt.figure(figsize=(12, 4))
    plt.plot(list_loss_train, color='royalblue', label='Train')
    plt.plot(list_loss_val, color='red', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


