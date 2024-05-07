"""
    Message logging module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import sys
from datetime import datetime
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style


colorama_init()


def _print_with_datetime(message: str) -> None:
    """Prefix the argument with a timestamp and print it.

    Args:
        message (str): Message to print.
    """
    print(f"{datetime.now().isoformat()} {message}")


def info(message: str) -> None:
    """Print information about the execution of the program.

    Args:
        message (str): Message to display.
    """
    _print_with_datetime(f"{Fore.GREEN}> INFO{Style.RESET_ALL} {message}")


def title(message: str) -> None:
    """Print a highlighted message. Used at the beginning of scripts.

    Args:
        message (str): Message to display.
    """
    _print_with_datetime("")
    n = (len(message) + 1) * "-"
    print(f"{Fore.MAGENTA}  |   ----------------{n}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}/ | \    MICROPYOME   {Style.RESET_ALL}{message}")
    print(f"{Fore.MAGENTA}  |   ----------------{n}{Style.RESET_ALL}")


def trace(message: str) -> None:
    """Print a trace (i.e. pedantic) message.

    Args:
        message (str): Message to display.
    """
    _print_with_datetime(f"{Fore.GREEN}>    {Style.RESET_ALL} {message}")


def warning(message: str) -> None:
    """Print a warning message that may cause a failure.

    Args:
        message (str): Message to display.
    """
    _print_with_datetime(f"{Fore.YELLOW}> WARNING{Style.RESET_ALL} {message}")


def error(message: str) -> None:
    """Print an error message.

    Args:
        message (str): Message to display.
    """
    _print_with_datetime(f"{Fore.RED}> ERROR{Style.RESET_ALL} {message}")


def progress_bar(N: int, n: int, suffix: str = "") -> None:
    """
    Print a progress bar in the standard output.

    Args:
        N (int): Total number of elements to process.
        n (int): Number of elements that have been processed.
        suffix (str): A text to display after the progress bar.
    """
    if n == N - 1:
        done = 50
    else:
        done = int(50 * n / N)
    bar = f"[{'=' * done}{' ' * (50-done)}]"
    back = '\033[K\r'
    timestamp = f"{datetime.now().isoformat()} "
    dash = f'{Fore.GREEN}>{Style.RESET_ALL} '
    prefix_len = len(str(N)) * 2 + 3
    prefix = f"{n} / {N}"
    prefix = (" " * (prefix_len - len(prefix))) + prefix + " "
    sys.stdout.write(back + timestamp + dash + prefix + bar + suffix)
    sys.stdout.flush()
