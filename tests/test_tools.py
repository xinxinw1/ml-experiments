import pytest

import tools

def test_group():
    assert list(tools.group([1, 2, 3], 2)) == [[1, 2], [3]]

def test_flatmap():
    assert list(tools.flatmap(lambda s: s, [[1, 2], [3]])) == [1, 2, 3]
    assert list(tools.flatmap(lambda s: [s, s+1], [1, 2])) == [1, 2, 2, 3]

def test_split_max_len():
    assert list(tools.split_max_len([[1, 2, 3, 4, 5], [1, 2], [1, 2, 3], [1, 2, 3, 4]], 4)) == [[1, 2, 3, 4], [5], [1, 2], [1, 2, 3], [1, 2, 3, 4]]

def test_pad_to_same_len():
    assert list(tools.pad_to_same_len([[1, 2], [1], [1, 2, 3]], 0)) == [[1, 2, 0], [1, 0, 0], [1, 2, 3]]

def test_make_batches():
    assert (list(tools.make_batches([[1, 2, 3, 4], [1]], 2, 3, 0, 0)) == [
        ([[0, 1, 2], [3, 4, 0]], [[1, 2, 3], [4, 0, 0]]),
        ([[0, 1]], [[1, 0]])
    ])
    batches = tools.make_batches([[1, 2, 3, 4]], batch_size=1, max_single_len=None, token_item=0, pad_item=0)
    assert next(batches) == ([[0, 1, 2, 3, 4]], [[1, 2, 3, 4, 0]])
    with pytest.raises(StopIteration):
        next(batches)
