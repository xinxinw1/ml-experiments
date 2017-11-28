#!/usr/bin/env python3

import setup

from models import lstm

model = lstm.LSTMModelAlphabet('simple-alphabet', ['hey', 'yo'])
model.train([['hey', 'yo']] * 10000)
model.save_to_file()
