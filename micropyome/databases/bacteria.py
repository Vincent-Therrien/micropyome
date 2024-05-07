"""
    Download bacteria datasets.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import shutil
import os

from micropyome.utils import file_io, log


DATASETS = {
    "ramirez": {
        "publication": "https://doi.org/10.1038/s41564-017-0062-x",
        "date": "2017-11-20",
        "description": "Macroecological bacteria dataset assembled from 30 studies that total 1998 soil samples.",  # nopep8
        "approximate_memsize": "70MB",
        "downloads": {
            "summary.csv": {
                "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41564-017-0062-x/MediaObjects/41564_2017_62_MOESM5_ESM.csv",  # nopep8
                "type": "csv",
                "encoding": "iso-8859-1"
            },
            "sequence-matched.csv": {
                "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41564-017-0062-x/MediaObjects/41564_2017_62_MOESM6_ESM.zip",  # nopep8
                "type": "zip",
                "unpacked_names": [
                    "41564_2017_62_MOESM6_ESM.csv",
                    "Rameriz_etal_SeqMatched.csv"
                ]
            },
            "name-matched.csv": {
                "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41564-017-0062-x/MediaObjects/41564_2017_62_MOESM7_ESM.zip",  # nopep8
                "type": "zip",
                "unpacked_names": [
                    "41564_2017_62_MOESM7_ESM.csv",
                    "Rameriz_etal_NameMatched.csv"
                ]
            }
        }
    }
}


def download(name: str, dst: str) -> None:
    """Download a bacteria dataset.

    Each dataset can comprise one or several files.

    Args:
        name (str): Name of the dataset. Run
            `micropyome.datasets.bacteria.get_names()` to see possible
            values.
        dst (str): Name of the **directory** in which data are
            downloaded.
    """
    # TODO: Check if already downloaded.
    # Validate arguments.
    if not name in DATASETS.keys():
        raise ValueError(
            f"The bacteria dataset `{name}` is unknown. "
            + "Run `micropyome.datasets.bacteria.get_names()` to obtain "
            + "the list of valid bacteria datasets."
        )
    # Create the result directory.
    if dst[-1] != '/':
        dst = dst + '/'
    file_io.create_dir(dst)
    # Download and unzip the files.
    for file_name, target in DATASETS[name]["downloads"].items():
        log.info(f"Installing `{file_name}` from bacteria dataset `{name}`.")
        complete_filename = dst + file_name
        file_io.download(target["url"], complete_filename, 1)
        # Clean a CSV file because they contain non UTF-8 characters.
        if target["type"] == "csv":
            with open(complete_filename, encoding=target["encoding"]) as file:
                content = file.read()
                cleaned_content = ''.join(
                    c for c in content if c.isprintable() or c == "\n"
                )
            os.remove(complete_filename)
            with open(complete_filename, "w") as output:
                output.write(cleaned_content)
        # Unzip zip files.
        elif target["type"] == "zip":
            for unpacking in target["unpacked_names"]:
                zip_name = complete_filename + ".zip"
                os.rename(complete_filename, zip_name)
                shutil.unpack_archive(zip_name, dst)
                os.remove(zip_name)
                shutil.copyfile(dst + unpacking, complete_filename)
                os.remove(dst + unpacking)
        else:
            pass
        log.info(f"Installed `{file_name}` from bacteria dataset `{name}`.")
