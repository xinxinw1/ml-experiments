#!/usr/bin/env python3

import setup

from models import lstm

model = lstm.LSTMModelEncoding('simple-string-alphabet', 'string-alphabet', 'ab')
model.train(['ab'] * 10000)
model.save_to_file()
