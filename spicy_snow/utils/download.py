"""
Utility functions for downloading files
"""
from pathlib import Path
import time
import gzip

import asf_search as asf
from asf_search import download_url

from checks import validate_urls

import logging
log = logging.getLogger(__name__)

def download_urls(urls, out_directory, reprocess=False, retries = 3):
    """
    Download a list of images from given URLs into the specified output directory.

    Args:
        urls (list): List of URLs (or nested iterables) pointing to images or files.
        out_directory (str or Path): Directory where files will be saved.
        reprocess (bool, optional): If True, re-download files even if they already exist. Defaults to False.

    Returns:
        list of Path: List of file paths to the downloaded files.
    """
    urls = validate_urls(urls)
    
    # Ensure output directory is a Path object
    out_directory = Path(out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)

    # ASF session for authenticated downloads
    session = asf.ASFSession()

    download_fps = []
    for url in urls:
        out_fp = out_directory.joinpath(Path(url).name)
        
        if out_fp.exists() and out_fp.stat().st_size != 0 and not reprocess:
            download_fps.append(out_fp)
            continue

        for attempt in range(retries):
            try:
                download_url(url, out_directory, session=session)
                break  # success, exit retry loop
            except Exception as e:
                if attempt == retries-1:  # last attempt
                    raise e
                # optionally print warning
                print(f"Retry {attempt+1} failed for {url}: {e}")
                time.sleep(2)  # wait before retrying

        download_fps.append(out_fp)
    
    return download_fps

def decompress(infile, tofile):
    """
    Decompress gzipped infile to outfile location.
    """

    log.debug(f"decompressing {infile} to {tofile}")

    with open(infile, 'rb') as inf, open(tofile, 'wb') as ouf:
        decom_str = gzip.decompress(inf.read())
        ouf.write(decom_str)

        return tofile
    