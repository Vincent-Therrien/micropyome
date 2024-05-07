"""
    File manipulation module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import os
import tarfile
import requests
import shutil

from micropyome.utils import log


def clean_dir_path(directory: str) -> str:
    """Return the correct filepath to `directory`.

    Args:
        directory (str): Directory name to validate.

    Raises:
        RuntimeError: If the directory does not exist.

    Returns (str): Cleaned directory path.
    """
    if not directory.endswith("/"):
        directory += "/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


def download(url: str, dst: str, verbosity: int, name: str = "") -> None:
    """Download a file through HTTPS.

    Args:
        url (str): Location of the file to download.
        dst (str): File path of the downloaded content.
        verbosity (int): Verbosity level of the function. `0` silences
            the function. `1` prints a loading bar.
        name (str): Name of the downloaded file - used for logging.
    """
    if verbosity > 1:
        print(f"Downloading the file `{url}` in `{dst}`.")
    with open(dst, "wb") as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')
        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                if verbosity:
                    suffix = f"Downloading   {name} "
                    log.progress_bar(total_length, dl, suffix)
    if verbosity:
        print()


def decompress_tar(
        filename: str,
        mode: str,
        dst: str,
        verbosity: int,
        name: str = ""
    ) -> None:
    """
    Decompress a TAR file.

    Args:
        filename (str): Name of the file to decompress.
        mode (str): Decompression mode (e.g. `r:gz`).
        dst (str): Output directory.
        verbosity (int): Verbosity level. `0` silences the function.
        name (str): Decompressed file name - used for logging.
    """
    tar = tarfile.open(filename, mode=mode)
    if verbosity:
        members = tar.getmembers()
        for i, member in enumerate(members):
            tar.extract(member, path=dst)
            if verbosity:
                log.progress_bar(len(members), i, f"Decompressing {name} ")
    else:
        tar.extractall(dst)
    tar.close()
    if verbosity:
        print()


def create_dir(filename: str) -> None:
    """Safe-create a directory."""
    dir = "".join([i + "/" for i in filename.split("/")[:-1]])
    if not os.path.exists(dir):
        os.makedirs(dir)


def safe_delete(filename: str) -> None:
    """Try to erase a file."""
    try:
        if os.path.isfile(filename) or os.path.islink(filename):
            os.remove(filename)
        elif os.path.isdir(filename):
            shutil.rmtree(filename)
    except:
        pass
