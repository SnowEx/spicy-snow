"""
Functions to create and set loggers.
"""

import sys
import os
import logging
from os.path import join, basename
from datetime import datetime
from contextlib import contextmanager

@contextmanager
def temp_silence_logger(name, level=logging.ERROR):
    logger = logging.getLogger(name)
    old_level = logger.level
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)

def setup_logging(log_dir = './logs/', debug = False):

    os.makedirs(log_dir, exist_ok= True)
    log_prefix = 'spicy-snow'
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")

    logging.basicConfig(level=logging.INFO,
                        
        format=f"(spicy-snow %(name)s %(levelname)s) %(message)s",
        # saves out to log file and outputs to command line.

        handlers=[
            logging.FileHandler(join(log_dir, f'{log_prefix}_{timestr}.log')),
            logging.StreamHandler(sys.stdout)]
    )
    
    log = logging.getLogger()

    if debug:
        log.setLevel(logging.DEBUG) 