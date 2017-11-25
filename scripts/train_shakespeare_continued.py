#!/usr/bin/env python3

import setup

import config
from models import lstm
import os
import itertools

model = lstm.LSTMModelFromFile('shakespeare-continued-2')
with open(os.path.join(config.DATA_DIR, 'shakespeare_input.txt'), 'r') as f:
    model.train([f], autosave=3000)
model.save_to_file()
