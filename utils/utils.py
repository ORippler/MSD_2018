import pickle
import os
import collections

import tensorflow as tf
import logging
import numpy as np
from .sitk_utils import resample_to_spacing, calculate_origin_offset


def distribution_strategy(num_gpus):
    if num_gpus == 1:
        return tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
    elif num_gpus > 1:
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    else:
        return None


def start_logger(logfile, level=logging.INFO):
    """Start logger

    Parameters
    ----------
    logfile : string (optional)
        file to which the log is saved
    level : int, default: logging.INFO
        logging level as int or logging.DEBUG, logging.ERROR
    """
    f = '%(asctime)s %(name)-35s %(levelname)-8s %(message)s'
    logging.basicConfig(level=level,
                        format=f,
                        datefmt='%d.%m. %H:%M:%S',
                        filename=logfile,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(level)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s %(levelname)s] %(message)s')
    formatter.datefmt = '%d.%m. %H:%M:%S'
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
