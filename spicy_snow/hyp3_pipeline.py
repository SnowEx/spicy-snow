"""
Functions to process sentinel-1 data through the ASF HyP3 pipeline
"""

import sys
import pandas as pd
from tqdm.auto import tqdm
import hyp3_sdk as sdk
from hyp3_sdk.exceptions import AuthenticationError

from typing import Union

import sys
from os.path import expanduser
sys.path.append(expanduser('~/Documents/spicy-snow'))

import logging
log = logging.getLogger(__name__)



def hyp3_pipeline(search_results: pd.DataFrame, job_name, existing_job_name: Union[bool, str] = False) -> sdk.jobs.Batch:
    """
    Start and monitor Hyp3 pipeline for desired Sentinel-1 granules
    https://hyp3-docs.asf.alaska.edu/using/sdk_api/

    Args:
    search_results: Pandas Dataframe of asf_search search results.
    job_name: name to give hyp3 batch run
    existing_job_name: if you have an existing job that you want to find and reuse [default: False]

    Returns:
    rtc_jobs: Hyp3 batch object of completed jobs.
    """ 
    try:
        # .netrc
        hyp3 = sdk.HyP3()
    except AuthenticationError:
        # prompt for password
        hyp3 = sdk.HyP3(prompt = True)

    # if existing job name exists then don't submit and simply watch existing jobs.
    while existing_job_name:
        log.debug(f"existing name provided {existing_job_name}.")
        rtc_jobs = hyp3.find_jobs(name = existing_job_name)
        rtc_jobs = rtc_jobs.filter_jobs(succeeded = True, failed = False, \
            running = True, include_expired = False)
        log.debug(f"Found {len(rtc_jobs)} jobs under existing name. \
                  This is only succeeded and running jobs.")

        # if no jobs found go to original search with name.
        if len(rtc_jobs) == 0:
            break

        # if no running jobs then just return succeeded jobs
        if len(rtc_jobs.filter_jobs(succeeded = False, failed = False)) == 0:
            log.debug("No running jobs. Returning succeeded.")
            return rtc_jobs.filter_jobs(succeeded = True)

        # otherwise watch running jobs
        hyp3.watch(rtc_jobs)

        # refresh with new successes and failures
        rtc_jobs = hyp3.refresh(rtc_jobs)
        
        log.debug(f"Successful jobs returned after watching {len(rtc_jobs.filter_jobs(succeeded = True))}")
        # return successful jobs
        return rtc_jobs.filter_jobs(succeeded = True)
    
    # check if you have passed quota
    quota = hyp3.check_credits()
    if not quota or len(search_results) > quota:
        raise ValueError(f'More search results ({len(search_results)}) than quota ({quota}).')


    # gather granules to submit to the hyp3 pipeline
    granules = search_results['properties.sceneName']

    # create a new hyp3 batch to hold submitted jobs
    rtc_jobs = sdk.Batch()
    for g in tqdm(granules, desc = 'Submitting s1 jobs'):
        # submit rtc jobs and ask for incidence angle map, in dBs, @ 30 m resolution
        # https://hyp3-docs.asf.alaska.edu/using/sdk_api/#hyp3_sdk.hyp3.HyP3.submit_rtc_job

        # DEM Matching turned off based on
        # https://www.mdpi.com/2072-4292/15/21/5110
        rtc_jobs += hyp3.submit_rtc_job(g, name = job_name, include_inc_map = False,\
            scale = 'power', dem_matching = False, resolution = 30, radiometry = 'gamma0')

    # warn user this may take a few hours for big jobs
    log.info(f'Watching {len(rtc_jobs)} jobs. This may take a while...')

    # have hyp3 watch and update progress bar every 60 seconds
    hyp3.watch(rtc_jobs)

    # refresh jobs list with successes and failures
    rtc_jobs = hyp3.refresh(rtc_jobs)

    # filter out failed jobs
    failed_jobs = rtc_jobs.filter_jobs(succeeded=False, running=False, failed=True)
    if len(failed_jobs) > 0:
        log.info(f'{len(failed_jobs)} jobs failed.')
    
    # return only successful jobs
    return rtc_jobs.filter_jobs(succeeded = True)
