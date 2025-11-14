from pathlib import Path
import time
import warnings
from urllib.parse import urlparse
from collections.abc import Iterable

import asf_search as asf
from asf_search import download_url

def validate_urls(urls, check_exists=False, timeout=5):
    """
    Flatten, validate, and optionally check existence of URLs.

    Args:
        urls (iterable): List/tuple/set of URLs or nested iterables.
        check_exists (bool): If True, perform a simple HTTP HEAD request to verify URL exists.
        timeout (int|float): Timeout for existence check in seconds.

    Returns:
        list: Flattened, validated URLs.

    Raises:
        ValueError: If any URL is invalid or if no URLs remain after validation.
    """

    def flatten(lst):
        """Recursively flatten nested iterables, skip strings as iterables."""
        for item in lst:
            if isinstance(item, str) or item is None:
                yield item
            elif isinstance(item, Iterable):
                yield from flatten(item)
            else:
                yield item

    clean_urls = []
    for u in flatten(urls):
        if u is None:
            continue

        if not isinstance(u, str):
            warnings.warn(f"Skipping non-string URL: {u}")
            continue

        u = u.strip()
        if not u:
            continue

        parsed = urlparse(u)
        if not parsed.scheme or not parsed.netloc:
            warnings.warn(f"URL does not look valid: {u}")
        
        clean_urls.append(u)

    if not clean_urls:
        raise ValueError("No valid URLs found in input.")

    return clean_urls

def download_images(urls, out_directory, reprocess=False, retries = 3):
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