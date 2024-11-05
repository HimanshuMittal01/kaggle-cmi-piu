"""
Contains utility functions to handle file information
"""

from pathlib import Path

import logging
from tabulate import tabulate
from tqdm.contrib.logging import tqdm_logging_redirect

def show_progress():
    tqdm_logger = logging.getLogger('tqdm')
    tqdm_logger.setLevel(logging.INFO)
    return tqdm_logging_redirect()


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def display_dir_items(path):
    filedata = []
    for filepath in path.glob('*'):
        pathtype = "File" if filepath.is_file() else "Folder"
        filesize = filepath.stat().st_size
        filedata.append([pathtype, filepath, sizeof_fmt(filesize)])

    print(tabulate(filedata, headers=['Type', 'Path', 'Size']))
