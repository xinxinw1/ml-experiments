import pytest

import tools
import numpy as np
import itertools

def assert_equals(a, b):
    assert a == b

def assert_iter_equals(it, exp, assert_fn=assert_equals):
    for item in exp:
        assert_fn(next(it), item)
    with pytest.raises(StopIteration):
        next(it)

def test_group():
    assert list(tools.group([1, 2, 3], 2)) == [[1, 2], [3]]
    assert list(tools.group([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    assert list(tools.group([1, 2, 3, 4], 5)) == [[1, 2, 3, 4]]
    input_it = iter([1, 2, 3, 4])
    output_it = tools.group(input_it, 2)
    assert next(output_it) == [1, 2]
    assert next(output_it) == [3, 4]
    with pytest.raises(StopIteration):
        next(output_it)

def test_flatmap():
    assert list(tools.flatmap(lambda s: s, [[1, 2], [3]])) == [1, 2, 3]
    assert list(tools.flatmap(lambda s: [s, s+1], [1, 2])) == [1, 2, 2, 3]

def test_drop_last():
    inp = iter([1, 2, 3, 4, 5])
    out = tools.drop_last(inp)
    expected = [1, 2, 3, 4]
    assert_iter_equals(out, [1, 2, 3, 4])

def test_drop_last_infinite():
    inp = itertools.cycle([1, 2, 3, 4, 5])
    out = itertools.islice(tools.drop_last(inp), 7)
    assert list(out) == [1, 2, 3, 4, 5, 1, 2]

def test_split_max_len():
    assert list(tools.split_max_len([[1, 2, 3, 4, 5], [1, 2], [1, 2, 3], [1, 2, 3, 4]], 4)) == [[1, 2, 3, 4], [5], [1, 2], [1, 2, 3], [1, 2, 3, 4]]

def test_pad_to_same_len():
    assert list(tools.pad_to_same_len([[1, 2], [1], [1, 2, 3]], 0)) == [[1, 2, 0], [1, 0, 0], [1, 2, 3]]

def test_single_make_batches():
    batches = list(tools.single_make_batches([1, 2, 3, 4, 5], 2, 2, 0))
    expected = [
        [[1, 2], [4, 5]],
        [[3], [0]]
    ]
    for batch, expect in zip(batches, expected):
        assert (batch == expect).all()

def test_make_batches_with_start_end():
    inputs = iter([[1, 2, 3, 4], [1], [2, 3]])
    batches = tools.make_batches_with_start_end(inputs, 2, 0, 5)
    expected = [
        ([[0, 1, 2, 3, 4], [0, 1, 5, 5, 5]], [[1, 2, 3, 4, 0], [1, 0, 5, 5, 5]]),
        ([[0, 2, 3]], [[2, 3, 0]])
    ]
    assert_iter_equals(batches, expected)

def test_make_batches_with_start_end_infinite():
    inputs = itertools.cycle([[1, 2, 3, 4], [1], [2, 3]])
    batches = itertools.islice(tools.make_batches_with_start_end(inputs, 2, 0, 5), 2)
    expected = [
        ([[0, 1, 2, 3, 4], [0, 1, 5, 5, 5]], [[1, 2, 3, 4, 0], [1, 0, 5, 5, 5]]),
        ([[0, 2, 3, 5, 5], [0, 1, 2, 3, 4]], [[2, 3, 0, 5, 5], [1, 2, 3, 4, 0]])
    ]
    assert_iter_equals(batches, expected)

def test_make_batches_long():
    inputs = iter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    batches = tools.make_batches_long(inputs, 2, 3, 0)
    expected = [
        ([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]),
        ([[7, 8, 9], [10, 0, 0]], [[8, 9, 10], [11, 0, 0]])
    ]
    assert_iter_equals(batches, expected)

    inputs = iter([1, 2, 3, 4])
    batches = tools.make_batches_long(inputs, max_batch_size=1, max_batch_width=None, pad_item=0)
    expected = [
        ([[1, 2, 3]], [[2, 3, 4]]),
    ]
    assert_iter_equals(batches, expected)

    inputs = [1]
    batches = tools.make_batches_long(inputs, max_batch_size=1, max_batch_width=None, pad_item=0)
    expected = [
        ([[]], [[]]),
    ]
    assert_iter_equals(batches, expected)

def test_make_batches_long_infinite():
    inputs = itertools.cycle([1, 2, 3, 4, 5])
    batches = itertools.islice(tools.make_batches_long(inputs, 2, 3, 0), 2)
    expected = [
        ([[1, 2, 3], [4, 5, 1]], [[2, 3, 4], [5, 1, 2]]),
        ([[2, 3, 4], [5, 1, 2]], [[3, 4, 5], [1, 2, 3]])
    ]
    assert_iter_equals(batches, expected)

def test_LongBatchMaker():
    def assert_fn(a, b):
        for inner_a, inner_b in zip(a, b):
            assert (inner_a == inner_b).all()

    maker = tools.LongBatchMaker(2, 3, 0)

    inputs = iter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    batches = maker.make_batches_for_training(inputs)
    expected = [
        ([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]),
        ([[7, 8, 9]], [[8, 9, 10]])
    ]
    assert_iter_equals(batches, expected, assert_fn)

    inputs = iter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    batches = maker.make_batches_for_training(inputs)
    expected = [
        ([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]),
        ([[7, 8, 9], [10, 0, 0]], [[8, 9, 10], [11, 0, 0]])
    ]
    assert_iter_equals(batches, expected, assert_fn)

    inputs = iter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    batches = maker.make_batches_for_training(inputs)
    expected = [
        ([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]),
        ([[7, 8, 9], [10, 11, 12]], [[8, 9, 10], [11, 12, 13]])
    ]
    assert_iter_equals(batches, expected, assert_fn)

@pytest.mark.parametrize('use_iter', [
    (False), (True)])
def test_make_batches(use_iter):
    inputs = [[1, 2, 3, 4], [1]]
    if use_iter:
        inputs = iter(inputs)
    batches = list(tools.make_batches([[1, 2, 3, 4], [1]], 2, 3, 0, 0))
    expected = [
        ([[0, 1, 2], [4, 0, 1]], [[1, 2, 3], [0, 1, 0]]),
        ([[3], [0]], [[4], [0]])
    ] 
    for (inputs_batch, labels_batch), (expected_inputs, expected_labels) in zip(batches, expected):
        # print(inputs_batch, labels_batch, expected_inputs, expected_labels)
        assert np.array_equal(inputs_batch, expected_inputs)
        assert np.array_equal(labels_batch, expected_labels)

    inputs = [[1, 2, 3, 4]]
    if use_iter:
        inputs = iter(inputs)
    batches = tools.make_batches(inputs, batch_size=1, max_single_len=None, token_item=0, pad_item=0)
    expected_inputs = [[0, 1, 2, 3, 4]]
    expected_labels = [[1, 2, 3, 4, 0]]
    inputs_batch, labels_batch = next(batches)
    assert np.array_equal(inputs_batch, expected_inputs)
    assert np.array_equal(labels_batch, expected_labels)
    with pytest.raises(StopIteration):
        next(batches)

def test_get_latest_in_dir(tmpdir):
    with pytest.raises(FileNotFoundError):
        tools.get_latest_in_dir(str(tmpdir))
    tmpdir.mkdir('1')
    tmpdir.mkdir('2')
    tmpdir.mkdir('12')
    assert tools.get_latest_in_dir(str(tmpdir)) == '2'
    assert tools.get_latest_in_dir(str(tmpdir), key=int) == '12'
