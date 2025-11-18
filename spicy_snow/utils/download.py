"""
Utility functions for downloading files
"""
from pathlib import Path
import time
import gzip
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm
import asf_search as asf
from asf_search import download_url

import sys
sys.path.append('/Users/zmhoppinen/Documents/spicy-snow')
from spicy_snow.utils.checks import validate_urls

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
    for url in tqdm(urls, desc = 'Downloading Urls'):
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

def download_urls_parallel(
    urls,
    out_directory,
    reprocess=False,
    retries=3,
    max_workers=10,
    timeout=20,
):
    """
    Faster parallel downloader using a shared requests.Session,
    keep-alive, connection pooling, larger thread pool,
    and streaming writes.
    """

    urls = validate_urls(urls)
    out_directory = Path(out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)

    # --- KEY SPEED-UP: pooled TCP connections
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers,
                                            pool_maxsize=max_workers,
                                            max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    def download_one(url):
        out_fp = out_directory / Path(url).name

        if out_fp.exists() and out_fp.stat().st_size != 0 and not reprocess:
            return out_fp

        for attempt in range(retries):
            try:
                r = session.get(url, stream=True, timeout=timeout)
                r.raise_for_status()

                # Fast streaming write
                with open(out_fp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
                return out_fp

            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(1.5)

        return out_fp

    # multi-threaded dispatch
    download_fps = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_one, url): url for url in urls}
        for fut in tqdm(as_completed(futures), total=len(urls), desc="Downloading"):
            download_fps.append(fut.result())

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

def download_proba_v(out_fp):
    # this is the url from Lievens et al. 2021 paper
    fcf_url = 'https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif'
    # download just forest cover fraction to out file
    fcf_fp = download_url(url = fcf_url, filename = out_fp)

    return fcf_fp
