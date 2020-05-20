import json

def read_json(path):
    """ Read json file

    Parameters
    ----------
    path: str
        The path where is the file.

    Returns
    -------
    dict
        The json data under a dictionary file.
    """

    with open(path, "r") as read_file:
        data = json.load(read_file)

    return data


def dump_json(data, path):
    """ Dump the data into a json file.

    Parameters
    ----------
    data: dict
        A dictionary.
    path: str
        The paths where the file is dumped.
    """

    with open(path, "w") as write_file:
        json.dump(data, write_file)
