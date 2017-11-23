#!/usr/bin/env python

import setup

import config
from models import lstm
import os
import itertools

model = lstm.LSTMModelFile('shakespeare')
with open(os.path.join(config.DATA_DIR, 'shakespeare_input.txt'), 'r') as f:
    model.train([f])
model.save_to_file()
