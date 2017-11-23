import pytest

import os

from models import lstm
import config

def test_simple():
    model = lstm.LSTMModel('test-simple', 2)
    model.train([[1, 0]] * 1000)
    model.sample()
    model.analyze([1, 0])
    model.save_to_file()
    model.sample([1])

    model = lstm.LSTMModelFromFile('test-simple')
    model.sample()
    model.analyze([1, 0])
    model.train([[1, 0]] * 1000)
    model.sample()
    model.analyze([1, 0])
    model.save_to_file()

def test_simple_string():
    model = lstm.LSTMModelString('test-simple-string')
    model.train(['ab'] * 50)
    model.sample()
    model.analyze('ab')
    model.save_to_file()
    model.sample('a')

    model = lstm.LSTMModelFromFile('test-simple-string')
    model.sample()
    model.analyze('ab')
    model.train(['ab'] * 50)
    model.sample()
    model.analyze('ab')
    model.save_to_file()

def test_simple_long():
    model = lstm.LSTMModel('test-simple-long', 2, use_long=True)
    model.train([[1, 0] * 1000])
    model.sample([1])
    model.analyze([1, 0])
    model.save_to_file()
    model.sample([1])

    model = lstm.LSTMModelFromFile('test-simple-long')
    model.sample([1])
    model.analyze([1, 0])
    model.train([[1, 0] * 1000])
    model.sample([1])
    model.analyze([1, 0])
    model.save_to_file()

def test_simple_string_long():
    model = lstm.LSTMModelString('test-simple-string', use_long=True)
    model.train(['ab' * 50])
    model.sample('a')
    model.analyze('ab')
    model.save_to_file()
    model.sample('a')

    model = lstm.LSTMModelFromFile('test-simple-string')
    model.sample('a')
    model.analyze('ab')
    model.train(['ab' * 50])
    model.sample('a')
    model.analyze('ab')
    model.save_to_file()

def test_text_file():
    model = lstm.LSTMModelTextFile('test-text-file')
    with open(os.path.join(config.DATA_DIR, 'test.txt'), 'r') as f:
        model.train([f])
    model.sample('a')
    model.analyze('ab')
    model.save_to_file()
