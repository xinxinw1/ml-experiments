#!/usr/bin/env python

import setup

from models import lstm

model = lstm.LSTMModel('simple', 2)
model.train([[1, 0]] * 30000)
model.save_to_file()
