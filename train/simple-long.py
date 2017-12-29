#!/usr/bin/env python3

import setup

from models import lstm

# Use skip_padding to remove the last training example which is padded
# to prevent spikes in loss_max
model = lstm.LSTMModel('simple-long', setup.tag, encoding=[('numbers', 2)], decoding=[('numbers', 2)],
        use_tokens=False, skip_padding=True)
data = [[1, 0] * 100000]
model.train(data, autosave=300, cont=True)
