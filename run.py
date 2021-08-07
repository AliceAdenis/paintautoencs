import logging
log = logging.getLogger(__name__)  # noqa: E402

import os
import configparser


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PARAMETERS
config = configparser.ConfigParser()
config.read('config.ini')
data_path = config['DEFAULT']['data_path']


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MAIN
if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    log.info('data_path: %s', data_path)


