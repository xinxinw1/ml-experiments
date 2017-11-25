#!/usr/bin/env python3

import setup

import config
from models import lstm
import os
import itertools

model = lstm.LSTMModelTextFile('shakespeare', skip_padding=True)
with open(os.path.join(config.DATA_DIR, 'shakespeare_input.txt'), 'r') as f:
    model.train([f], autosave=3000, cont=True)
