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

def test_make_batches():
    batches = list(tools.make_batches([[1, 2, 3, 4], [1]], 2, 3, 0, 0))
    print(batches)
    expected = [
        ([[0, 1, 2], [4, 0, 1]], [[1, 2, 3], [0, 1, 0]]),
        ([[3], [0]], [[4], [0]])
    ] 
    for (inputs_batch, labels_batch), (expected_inputs, expected_labels) in zip(batches, expected):
        print(inputs_batch, labels_batch, expected_inputs, expected_labels)
        assert (inputs_batch == expected_inputs).all()
        assert (labels_batch == expected_labels).all()

    batches = tools.make_batches([[1, 2, 3, 4]], batch_size=1, max_single_len=None, token_item=0, pad_item=0)
    expected_inputs = [[0, 1, 2, 3, 4]]
    expected_labels = [[1, 2, 3, 4, 0]]
    inputs_batch, labels_batch = next(batches)
    assert (inputs_batch == expected_inputs).all()
    assert (labels_batch == expected_labels).all()
    with pytest.raises(StopIteration):
        next(batches)
