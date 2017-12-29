#!/usr/bin/env python3

import setup

from models import lstm
alphabet = encoding.make_alphabet_with_encoding(['hey', 'yo'])
model = lstm.LSTMModel('simple-alphabet', setup.tag, encoding=[('alphabet', alphabet)], decoding=[('alphabet', alphabet)])
model.train([['hey', 'yo']] * 10000)
model.save_to_file()
