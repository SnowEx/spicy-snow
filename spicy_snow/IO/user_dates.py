"""
Functions to capture, check, and return dates.
"""

from typing import Union
import numpy as np
import pandas as pd
from datetime import datetime

def get_input_dates(date2: Union[str, pd.Timestamp, np.datetime64], 
                    date1: Union[None, str, pd.Timestamp, pd.Timedelta, np.datetime64] = None):
    """
    This helper function parses user inputs and returns two dates for use in retrieval.py

    Args:
    date2: The second date to end retrievals at.
    date1: The first date to start retrievals at. If missing will be assumed to
    be last August 1st before date2

    Returns:
    retrieval_dates: Tuple of two dates in YYYY-MM-DD format to pass to retrieval
    function.
    """

    # try and parse date inputs

    # if date2 is not none then we convert to pd datetime 
    if date2:

        date2 = pd.to_datetime(date2)

        assert date2.year > 2014, f"Please enter a date after 2016 for s1 coverage.\
            Current parsed date is {date2}"
        
        assert date2 < datetime.now(), f"Please enter a date in the past"
    
    # if date1 is not none then convert to pd datetime.

    if date1:

        date1 = pd.to_datetime(date1)

        assert date1 < date2, f"Provided date 1 is later than date2."

    else:

        if date2.month < 8:
            date1 = pd.to_datetime(f'{int(date2.year - 1)}-08-01')
        else:
            date1 = pd.to_datetime(f'{int(date2.year)}-08-01')

    return date1.strftime('%Y-%m-%d'), date2.strftime('%Y-%m-%d')

