#!/usr/bin/env python

import setup

from models import lstm
import numpy as np

model = lstm.LSTMModel('count-1-0', 2)
model.train([(lambda n: [1]*n + [0]*n)(np.random.randint(0, 10)) for i in range(100000)])
model.save_to_file()
