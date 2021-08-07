import logging
log = logging.getLogger(__name__)  # noqa: E402

import os

def folder(folder_path):
    """Create folder if it does not exist already.

    Parameters
    ----------
    folder_path: str
        The path to the folder to create.
    """
    if not os.path.isdir(folder_path):
        if not os.path.isdir(os.path.split(folder_path)[0]):
            folder(os.path.split(folder_path)[0])
        os.mkdir(folder_path)
