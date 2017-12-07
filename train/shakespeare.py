#!/usr/bin/env python3

import setup

import config
from models import lstm
import os
import itertools

path = os.path.join(config.DATA_DIR, 'shakespeare_input.txt')
model = lstm.LSTMModelTextFileAlphabetFile('shakespeare', path, skip_padding=True, tag=setup.tag)
model.train([path]*12, autosave=True, cont=True, count=True)
