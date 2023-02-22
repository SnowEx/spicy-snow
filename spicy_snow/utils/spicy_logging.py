"""
Functions to create and set loggers.
"""

import sys
import os
import logging
from os.path import join, basename
from datetime import datetime

def setup_logging(log_dir = './logs/', debug = False):

    os.makedirs(log_dir, exist_ok= True)
    log_prefix = 'spicy-snow'
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")

    logging.basicConfig(level=logging.INFO,
                        
        format=f"(spicy-snow {__name__} %(levelname)s) %(message)s",
        # saves out to log file and outputs to command line.

        handlers=[
            logging.FileHandler(join(log_dir, f'{log_prefix}_{timestr}.log')),
            logging.StreamHandler(sys.stdout)]
    )
    
    log = logging.getLogger()

    if debug:
        log.setLevel(logging.DEBUG) 