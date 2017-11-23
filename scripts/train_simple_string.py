#!/usr/bin/env python3

import setup

from models import lstm

model = lstm.LSTMModelString('simple-string')
model.train(['ab'] * 3000)
model.save_to_file()
