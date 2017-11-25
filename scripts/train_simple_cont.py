#!/usr/bin/env python3

import setup

from models import lstm

model = lstm.LSTMModelFromFile('simple')
model.train([[1, 0]] * 10000)
model.save_to_file()
