import pytest

import os

from models import lstm
import config

@pytest.fixture
def saved_dirs_kwargs(tmpdir):
    return {
        'saved_models_dir': str(tmpdir.mkdir('models')),
        'saved_summaries_dir': str(tmpdir.mkdir('summaries'))
    }

def test_simple_short(saved_dirs_kwargs):
    model = lstm.LSTMModel('test-simple', 2, **saved_dirs_kwargs)
    model.train([[1, 0]] * 1000)
    model.sample()
    model.analyze([1, 0])
    model.save_to_file()
    model.sample([1])

    model = lstm.LSTMModelFromFile('test-simple', **saved_dirs_kwargs)
    model.sample()
    model.analyze([1, 0])
    model.train([[1, 0]] * 1000)
    model.sample()
    model.analyze([1, 0])
    model.save_to_file()

    model.train([[1, 0]] * 1000, autosave=30)

def test_simple_string_short(saved_dirs_kwargs):
    model = lstm.LSTMModelString('test-simple-string', **saved_dirs_kwargs)
    model.train(['ab'] * 50)
    model.sample()
    model.analyze('ab')
    model.save_to_file()
    model.sample('a')

    model = lstm.LSTMModelFromFile('test-simple-string', **saved_dirs_kwargs)
    model.sample()
    model.analyze('ab')
    model.train(['ab'] * 50)
    model.sample()
    model.analyze('ab')
    model.save_to_file()

def test_simple_long(saved_dirs_kwargs):
    model = lstm.LSTMModel('test-simple-long', 2, use_long=True, **saved_dirs_kwargs)
    model.train([[1, 0] * 1000])
    model.sample([1])
    model.analyze([1, 0])
    model.save_to_file()
    model.sample([1])

    model = lstm.LSTMModelFromFile('test-simple-long', **saved_dirs_kwargs)
    model.sample([1])
    model.analyze([1, 0])
    model.train([[1, 0] * 1000])
    model.sample([1])
    model.analyze([1, 0])
    model.save_to_file()

def test_simple_string_long(saved_dirs_kwargs):
    model = lstm.LSTMModelString('test-simple-string', use_long=True, **saved_dirs_kwargs)
    model.train(['ab' * 50])
    model.sample('a')
    model.analyze('ab')
    model.save_to_file()
    model.sample('a')

    model = lstm.LSTMModelFromFile('test-simple-string', **saved_dirs_kwargs)
    model.sample('a')
    model.analyze('ab')
    model.train(['ab' * 50])
    model.sample('a')
    model.analyze('ab')
    model.save_to_file()

def test_text_file(saved_dirs_kwargs):
    model = lstm.LSTMModelTextFile('test-text-file', **saved_dirs_kwargs)
    with open(os.path.join(config.ROOT_DIR, 'tests', 'small.txt'), 'r') as f:
        model.train([f])
    model.sample('a')
    model.analyze('ab')
    model.save_to_file()
