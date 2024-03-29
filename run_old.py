import logging
log = logging.getLogger(__name__)  # noqa: E402

import os

import torchvision.transforms as transforms
import torch.nn.functional as func
from torch import optim

from AutoEncoders import AEPaintingsV0, AEPortraitV1, AEPortraitV2
from Datasets import DatasetPaintings
from Optimizers import Optimizer


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PARAMETERS
raw_path = '/srv/data/data_peinture/data/processed/portraits'
autoencoders_path = '/srv/data/data_peinture/models/autoenc_paintings_V2'
results_path = '/srv/data/data_peinture/data/results/portraits_V2'

image_size = 149
autoenc_key = "portraitv2"

dict_autoencs = {
    "paintingsv0": AEPaintingsV0,
    "portraitv1": AEPortraitV1,
    "portraitv2": AEPortraitV2,
}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    log.info('training autoencoder')

    # Dataset
    log.info('loading data')
    path_list = [os.path.join(raw_path, f) for f in os.listdir(raw_path)
                 if f.split('.')[-1] == 'jpg']

    transf = transforms.Compose([
        transforms.CenterCrop(size=image_size),
        # transforms.RandomCrop(size=image_size),
        # transforms.RandomResizedCrop(size=image_size),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda tens: tens.view([1, 3, image_size, image_size]))
    ])

    dataset = DatasetPaintings(path_list, transform=transf)

    # Autoencoder
    autoenc = dict_autoencs[autoenc_key](image_size)

    # Optimization
    optim = Optimizer(
        autoenc=autoenc,
        loss_func=func.mse_loss,
        optimizer=optim.Adam(autoenc.parameters(), lr=1e-3),
        path_model=autoencoders_path,
        level_treshold=0.10
    )

    optim.optimize(dataset, n_epoch=400, n_save=20, save_fig=results_path)

    # optim.plot_loss(dataset, path_results)
    # optim.plot_sample(dataset, path_results, n_sample=2)
