#!/usr/bin/env python3

import setup

import config
from models import lstm
import os
import itertools

path = os.path.join(config.DATA_DIR, 'shakespeare_input.txt')
alphabet = encoding.make_alphabet_with_encoding(path, encoding=['path-to-chars'])
model = lstm.LSTMModel('shakespeare', setup.tag,
        encoding=['string-to-chars', ('alphabet-to-nums', alphabet)],
        train_encoding=['path-to-chars', ('alphabet-to-nums', alphabet)],
        decoding=[('nums-to-alphabet', alphabet), 'chars-to-string'],
        use_tokens=False, skip_padding=True)
model.train([path]*3, autosave=10000, cont=True, count=True)
