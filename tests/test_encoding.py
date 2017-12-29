import pytest

import encoding

def assert_equals(a, b):
    assert a == b

def assert_iter_equals(it, exp, assert_fn=assert_equals):
    for item in exp:
        assert_fn(next(it), item)
    with pytest.raises(StopIteration):
        next(it)

@pytest.mark.parametrize('i', [(0), (1), (2)])
def test_alphabet(i):
    alphabet = encoding.Alphabet([1, 2, 3])
    assert alphabet.encode_ind(i+1) == i

def test_ordered_alphabet():
    alphabet = encoding.Alphabet([3, 1, 2])
    assert alphabet.encode_ind(1) == 0

def test_encoder_names_to_objects():
    alphabet = encoding.Alphabet({32, 97, 98})

    encoder_names = [('numbers', 5)]
    objects = encoding.encoder_names_to_objects(encoder_names)
    assert isinstance(objects, list)

    encoder_names = ['path-to-lines', 'string-to-bytes', ('alphabet-to-nums', alphabet)]
    encoding.encoder_names_to_objects(encoder_names)

    encoder_names = [('numbers', 5), 'string-to-chars', 'chars-to-string', 'string-to-bytes', 'bytes-to-string', 'path-to-chars', 'path-to-bytes', 'path-to-lines', ('alphabet-to-nums', alphabet), ('nums-to-alphabet', alphabet)]
    encoding.encoder_names_to_objects(encoder_names)

def test_encodings():
    alphabet = encoding.Alphabet({32, 97, 98})
    encoder = encoding.AlphabetToNums(alphabet)
    assert encoder.encode_individual(32) == 0
    assert_iter_equals(encoder.encode_single([32, 97, 98]), [0, 1, 2])
    assert_iter_equals(encoder.encode_multiple([[32, 97, 98], [98, 97, 32]]), [[0, 1, 2], [2, 1, 0]], assert_fn=assert_iter_equals)
    assert encoder.get_alphabet_size(None) == 3

    encoder = encoding.NumsToAlphabet(alphabet)
    assert encoder.encode_individual(0) == 32
    assert_iter_equals(encoder.encode_single([0, 1, 2]), [32, 97, 98])
    assert_iter_equals(encoder.encode_multiple([[0, 1, 2], [2, 1, 0]]), [[32, 97, 98], [98, 97, 32]], assert_fn=assert_iter_equals)

def test_encode():
    alphabet = encoding.Alphabet({32, 97, 98})
    encoder_names = [('nums-to-alphabet', alphabet), 'bytes-to-string']
    encoder = encoding.encoder_names_to_objects(encoder_names)
    assert encoding.encode_individual(0, encoder) == ' '
    assert encoding.encode_single([0, 2, 1], encoder) == ' ba'
    assert_iter_equals(encoding.encode_multiple([[0, 2, 1], [1, 2, 0]], encoder), [' ba', 'ab '])

def test_encode_2():
    alphabet = encoding.Alphabet({32, 97, 98})
    encoder_names = ['string-to-bytes', ('alphabet-to-nums', alphabet)]
    encoder = encoding.encoder_names_to_objects(encoder_names)
    with pytest.raises(NotImplementedError):
        encoding.encode_individual(' ', encoder)
    assert_iter_equals(encoding.encode_single(' ba', encoder), [0, 2, 1])
    assert_iter_equals(encoding.encode_multiple([' ba', 'ab '], encoder), [[0, 2, 1], [1, 2, 0]], assert_fn=assert_iter_equals)
    assert encoding.get_alphabet_size(encoder) == 3
    assert encoding.get_empty_single(encoder) == ''

def test_encode_3():
    encoder_names = ['string-to-bytes', 'bytes-to-string']
    encoder = encoding.encoder_names_to_objects(encoder_names)
    with pytest.raises(NotImplementedError):
        encoding.encode_individual(' ', encoder)
    assert encoding.encode_single(' ba', encoder) == ' ba'
    assert_iter_equals(encoding.encode_multiple([' ba', 'ab '], encoder), [' ba', 'ab '])

def test_encode_4():
    encoder_names = [('numbers', 3)]
    encoder = encoding.encoder_names_to_objects(encoder_names)
    assert encoding.encode_individual(0, encoder) == 0
    assert encoding.encode_individual(2, encoder) == 2
    with pytest.raises(ValueError):
        encoding.encode_individual(-1, encoder)
    with pytest.raises(ValueError):
        encoding.encode_individual(3, encoder)

def test_not_implemented():
    encoder = encoding.Encoding()
    with pytest.raises(NotImplementedError):
        encoder.encode_individual(0)
    with pytest.raises(NotImplementedError):
        encoder.get_empty_single()

def test_make_alphabet():
    encoder = ['string-to-chars']
    alphabet = encoding.make_alphabet_with_encoding('the quick brown fox jumps over the lazy dog', encoder)
    assert alphabet.alphabet_list == list(' abcdefghijklmnopqrstuvwxyz')

