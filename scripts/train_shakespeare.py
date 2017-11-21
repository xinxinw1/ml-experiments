#!/usr/bin/env python

import setup

import config
from models import lstm
import os
import itertools

model = lstm.LSTMModelString('shakespeare')
with open(os.path.join(config.DATA_DIR, 'shakespeare_input.txt'), 'r') as f:
    model.train(itertools.islice(f, 1000))
model.save_to_file()
