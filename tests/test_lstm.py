import pytest

import os

from models import lstm
import tools
import config

@pytest.fixture
def lstm_kwargs(tmpdir):
    return {
        'saved_models_dir': str(tmpdir.mkdir('models')),
        'saved_summaries_dir': str(tmpdir.mkdir('summaries'))
    }

@pytest.fixture
def lstm_kwargs_1(lstm_kwargs):
    return dict(lstm_kwargs, tag='test-1')

@pytest.fixture
def lstm_kwargs_2(lstm_kwargs):
    return dict(lstm_kwargs, tag='test-2')

def test_simple_short(lstm_kwargs_1, lstm_kwargs_2):
    model = lstm.LSTMModel('test-simple', 2, **lstm_kwargs_1)
    model.train([[1, 0]] * 1000)
    s = model.sample()
    assert isinstance(s, list)
    print(s)
    model.analyze([1, 0])
    model.save_to_file()
    model.sample([1])

    model = lstm.LSTMModelFromFile('test-simple', **lstm_kwargs_1)
    s = model.sample()
    assert isinstance(s, list)
    print(s)
    model.analyze([1, 0])
    model.train([[1, 0]] * 1000)
    model.sample()
    model.analyze([1, 0])
    model.save_to_file()
    # Double saving is ok
    model.save_to_file()

    model = lstm.LSTMModel('test-simple', 2, **lstm_kwargs_1)
    model.train([[1, 0]] * 1000, autosave=30)

    model = lstm.LSTMModel('test-simple', 2, **lstm_kwargs_2)
    model.train([[1, 0]] * 1000, autosave=30)

    model = lstm.LSTMModelFromFile('test-simple-copy', from_name='test-simple', training_steps=100, **lstm_kwargs_1)
    model.train([[1, 0]] * 1000, autosave=30)

def test_simple_alphabet_short(lstm_kwargs_1):
    model = lstm.LSTMModelAlphabet('test-simple-alphabet', ['hey', 'yo'], **lstm_kwargs_1)
    model.train([['hey', 'yo']] * 50)
    s = model.sample()
    assert isinstance(s, list)
    print(s)
    model.analyze(['hey', 'yo'])
    model.save_to_file()
    model.sample(['hey'])

    model = lstm.LSTMModelFromFile('test-simple-alphabet', **lstm_kwargs_1)
    s = model.sample()
    assert isinstance(s, list)
    print(s)
    model.analyze(['hey', 'yo'])
    model.train([['hey', 'yo']] * 50)
    model.sample()
    model.analyze(['hey', 'yo'])
    model.save_to_file()

def test_simple_string_short(lstm_kwargs_1):
    model = lstm.LSTMModelString('test-simple-string', **lstm_kwargs_1)
    model.train(['ab'] * 50)
    s = model.sample()
    assert tools.is_string_or_bytes(s)
    print(s)
    model.analyze('ab')
    model.save_to_file()
    model.sample('a')

    model = lstm.LSTMModelFromFile('test-simple-string', **lstm_kwargs_1)
    s = model.sample()
    assert tools.is_string_or_bytes(s)
    print(s)
    model.analyze('ab')
    model.train(['ab'] * 50)
    model.sample()
    model.analyze('ab')
    model.save_to_file()

def test_simple_string_alphabet_short(lstm_kwargs_1):
    model = lstm.LSTMModelStringAlphabet('test-simple-string-alphabet', 'ab', **lstm_kwargs_1)
    model.train(['ab'] * 50)
    s = model.sample()
    assert tools.is_string_or_bytes(s)
    print(s)
    model.analyze('ab')
    model.save_to_file()
    model.sample('a')

    model = lstm.LSTMModelFromFile('test-simple-string-alphabet', **lstm_kwargs_1)
    s = model.sample()
    assert tools.is_string_or_bytes(s)
    print(s)
    model.analyze('ab')
    model.train(['ab'] * 50)
    model.sample()
    model.analyze('ab')
    model.save_to_file()

def test_simple_long(lstm_kwargs_1):
    model = lstm.LSTMModel('test-simple-long', 2, use_long=True, **lstm_kwargs_1)
    model.train([[1, 0] * 1000])
    s = model.sample([1])
    assert isinstance(s, list)
    print(s)
    model.analyze([1, 0])
    model.save_to_file()
    model.sample([1])

    model = lstm.LSTMModelFromFile('test-simple-long', **lstm_kwargs_1)
    s = model.sample([1])
    assert isinstance(s, list)
    print(s)
    model.analyze([1, 0])
    model.train([[1, 0] * 1000])
    model.sample([1])
    model.analyze([1, 0])
    model.save_to_file()

    model.train([[1, 0]])

def test_simple_long_skip_padding(lstm_kwargs_1):
    model = lstm.LSTMModel('test-simple-long', 2, use_long=True, skip_padding=True, **lstm_kwargs_1)
    model.train([[1, 0] * 1000])
    model.train([[1, 0]])
    model.save_to_file()

    model = lstm.LSTMModelFromFile('test-simple-long', **lstm_kwargs_1)
    model.train([[1, 0] * 1000])
    model.train([[1, 0]])
    model.save_to_file()

def test_simple_string_long(lstm_kwargs_1):
    model = lstm.LSTMModelString('test-simple-string', use_long=True, **lstm_kwargs_1)
    model.train(['ab' * 50])
    s = model.sample('a')
    assert tools.is_string_or_bytes(s)
    print(s)
    model.analyze('ab')
    model.save_to_file()
    model.sample('a')

    model = lstm.LSTMModelFromFile('test-simple-string', **lstm_kwargs_1)
    s = model.sample('a')
    assert tools.is_string_or_bytes(s)
    print(s)
    model.analyze('ab')
    model.train(['ab' * 50])
    model.sample('a')
    model.analyze('ab')
    model.save_to_file()

def test_text_file(lstm_kwargs_1):
    model = lstm.LSTMModelTextFile('test-text-file', **lstm_kwargs_1)
    path = os.path.join(config.ROOT_DIR, 'tests', 'small.txt')
    model.train([path])
    s = model.sample('a')
    assert tools.is_string_or_bytes(s)
    print(s)
    model.analyze('ab')
    model.save_to_file()

def test_text_file_alphabet(lstm_kwargs_1):
    path = os.path.join(config.ROOT_DIR, 'tests', 'small.txt')
    model = lstm.LSTMModelTextFileAlphabet('test-text-file-alphabet', path, **lstm_kwargs_1)
    model.train([path])
    s = model.sample('a')
    assert tools.is_string_or_bytes(s)
    print(s)
    model.analyze('ab')
    model.save_to_file()
