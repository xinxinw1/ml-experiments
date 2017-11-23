#!/usr/bin/env python3

import setup

from models import lstm

model = lstm.LSTMModelEncoding('simple-string-alphabet', 'string-alphabet', 2)
model.train(['ab'] * 30000)
model.save_to_file()
