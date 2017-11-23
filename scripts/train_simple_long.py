#!/usr/bin/env python3

import setup

from models import lstm

model = lstm.LSTMModel('simple-long', 2, use_long=True)
model.train([[1, 0] * 100000])
model.save_to_file()
