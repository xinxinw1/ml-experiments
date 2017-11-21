import pytest

import tools
import numpy as np

def test_group():
    assert list(tools.group([1, 2, 3], 2)) == [[1, 2], [3]]

def test_flatmap():
    assert list(tools.flatmap(lambda s: s, [[1, 2], [3]])) == [1, 2, 3]
    assert list(tools.flatmap(lambda s: [s, s+1], [1, 2])) == [1, 2, 2, 3]

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
