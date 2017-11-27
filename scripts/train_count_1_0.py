#!/usr/bin/env python3

import setup

from models import lstm
import numpy as np

model = lstm.LSTMModel('count-1-0', 2)
data = [(lambda n: [1]*n + [0]*n)(np.random.randint(0, 10)) for i in range(100000)]
model.train(data, count=True)
model.save_to_file()
