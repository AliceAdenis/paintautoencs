import logging
log = logging.getLogger(__name__)

import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from AutoEncoders import AEPaintingsV0, AEPortraitV1
from Datasets import DatasetPaintings
from Optimizers import Optimizer


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PARAMETERS
path_raw = '/srv/data/data_peinture/data/processed/portraits'
path_autoencoders = '/srv/data/data_peinture/models/autoenc_paintings_V1'
path_results = '/srv/data/data_peinture/data/results/portraits'

image_size = 193
autoenc_key = "portraitv1"

dict_autoencs = {
    "paintingsv0": AEPaintingsV0,
    "portraitv1": AEPortraitV1
}



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    log.info('training autoencoder')

    # Dataset
    log.info('loading data')
    list_path = [os.path.join(path_raw, f) for f in os.listdir(path_raw) \
                 if f.split('.')[-1] == 'jpg']

    transf = transforms.Compose([
        transforms.RandomCrop(size=image_size),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda tens: tens.view([1, 3, image_size, image_size]))
    ])

    dataset = DatasetPaintings(list_path, transform=transf)

    # Autoencoder
    autoenc = dict_autoencs[autoenc_key](image_size)

    # Optimization
    optim = Optimizer(
        autoenc=autoenc,
        loss_func=F.mse_loss,
        optimizer=optim.Adam(autoenc.parameters(), lr=2e-3),
        path_model=path_autoencoders
    )

    optim.optimize(dataset, n_epoch=1000, save_fig=path_results)

    #optim.plot_loss(dataset, path_results)
    #optim.plot_sample(dataset, path_results, n_sample=2)

