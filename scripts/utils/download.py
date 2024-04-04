from sys import stdout

from wget import download as wget_download


def download(url: str, save_path: str):
    wget_download(url, save_path, bar=bar_progress)


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] Mb" % (
        current / total * 100, 
        current // 1000 ** 2, 
        total // 1000 ** 2
    )
    # Don't use print() as it will print in new line every time.
    stdout.write("\r" + progress_message)
    stdout.flush()