import logging
log = logging.getLogger(__name__)  # noqa: E402

import os
import shutil
import configparser

from torch import manual_seed as torch_manual_seed
import torchvision.transforms as transforms
import torch.nn.functional as func
from torch import optim

from Datasets import DatasetPaintings
from AutoEncoders import AEPortraitV2
from Optimizers import Optimizer

from plots import plot_data
import utl


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PARAMETERS
config = configparser.ConfigParser()
config.read('config.ini')

data_path = config['DEFAULT']['data_path']
raw_data_path = os.path.join(data_path, config['DEFAULT']['raw_data_path'])
report_path = os.path.join(data_path, config['DEFAULT']['report_path'])
autoencoders_path = os.path.join(data_path, config['DEFAULT']['autoencoders_path'])

image_size = int(config['DEFAULT']['image_size'])
torch_seed = int(config['DEFAULT']['torch_seed'])

torch_manual_seed(torch_seed)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MAIN
if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    # Path
    if os.path.isdir(report_path):
        shutil.rmtree(report_path)
    os.mkdir(report_path)

    # Dataset
    log.info('loading data from %s', raw_data_path)
    path_list = [os.path.join(raw_data_path, f) for f in os.listdir(raw_data_path)
                 if os.path.splitext(f)[1] == '.jpg']
    log.debug('list of images: %s', path_list)

    transf = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        # transforms.CenterCrop(size=(image_size, image_size)),
        # transforms.RandomCrop(size=(image_size, image_size)),
        # transforms.RandomResizedCrop(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda tens: tens.view([1, 3, image_size, image_size]))
    ])

    dataset = DatasetPaintings(path_list, transform=transf)

    # Plot input samples
    for T in dataset.load_batches():
        log.info('batch: %s', T.shape)
        input_sample_path = os.path.join(report_path, 'input_samples')
        utl.folder(input_sample_path)
        for i, img in enumerate(T):
            plot_data(img, os.path.join(input_sample_path, 'figure_' + str(i) + '.jpg'))
        break

    # Autoencoder
    autoenc = AEPortraitV2(image_size)

    # Optimization
    utl.folder(autoencoders_path)
    optim = Optimizer(
        autoenc=autoenc,
        loss_func=func.mse_loss,
        optimizer=optim.Adam(autoenc.parameters(), lr=1e-3),
        path_model=autoencoders_path,
        level_treshold=0.10
    )

    results_path = os.path.join(report_path, 'results')
    utl.folder(results_path)
    optim.optimize(dataset, n_epoch=400, n_save=20, save_fig=results_path)
