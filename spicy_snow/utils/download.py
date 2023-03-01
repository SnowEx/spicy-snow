"""
Utility functions for downloading files
"""
import sys
import time
from os.path import basename, exists
import gzip
from urllib.request import urlretrieve

import logging
log = logging.getLogger(__name__)

def reporthook(count, block_size, total_size):
    """
    Hook for urlib downloads to get progress readout.
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = min(int(count*block_size*100/total_size),100)
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def url_download(url, out_fp, overwrite = False, verbose = True):
    """
    Downloads url with a progress bar and overwrite check.
    """
    # check if file already exists
    if not exists(out_fp) or overwrite == True:
        # progress bar for your download?
        if verbose:
            log.info(f'Downloading {basename(out_fp)}.')
            urlretrieve(url, out_fp, reporthook)
            log.info('')
        # or not?
        else:
            urlretrieve(url, out_fp)
    # if already exists. skip download.
    else:
        if verbose:
            log.info(f'{basename(out_fp)} already exists. Skipping.')

def decompress(infile, tofile):
    """
    Decompress gzipped infile to outfile location.
    """

    log.debug(f"decompressing {infile} to {tofile}")

    with open(infile, 'rb') as inf, open(tofile, 'wb') as ouf:
        decom_str = gzip.decompress(inf.read())
        ouf.write(decom_str)

        return tofile