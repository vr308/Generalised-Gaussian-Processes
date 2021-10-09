#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from six.moves import configparser
from utils.config import *
import os

# cfg = configparser.ConfigParser()
# dirs = [os.curdir, os.path.dirname(os.path.realpath(__file__)),
#         os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')]
# locations = map(os.path.abspath, dirs)

# for loc in locations:
#     if cfg.read(os.path.join(loc, 'config.ini')):
#         break

def expand_to_absolute(path):
    if './' == path[:2]:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), path[2:])
    else:
        return path

#DATA_PATH = expand_to_absolute(cfg['paths']['data_path'])
#BASE_SEED = int(cfg['seeds']['seed'])

DATA_PATH = DATASET_DIR
#RESULTS_DB_PATH = expand_to_absolute(cfg['paths']['results_path'])

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

#print(dirs)
#print(loc)
#print(__file__)