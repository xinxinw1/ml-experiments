import shutil
import math
import numpy as np
import itertools

def group(lst, n):
    """
    Args:
        lst: A python iterable
    Returns:
        grps: A python iterator of python lists
    """
    currlst = []
    for item in lst:
        currlst.append(item)
        if len(currlst) == n:
            yield currlst
            currlst = []
    if currlst:
        yield currlst

def flat(lst_of_lsts):
    return (item for lst in lst_of_lsts for item in lst)

def flatmap(f, lst):
    """
    Args:
        lst: A python iterable
    Returns:
        new_lst: A python iterator
    """
    return flat(map(f, lst))

def drop_last(iterable):
    it = iter(iterable)
    curr = next(it)
    for item in it:
        yield curr
        curr = item

def split_max_len(lst_of_lsts, max_len):
    """
    Args:
        lst_of_lsts: A python iterable of python lists
    Returns:
        split_lst: A python list of python lists
    """
    return list(flatmap(lambda lst: group(lst, max_len), lst_of_lsts))

def pad_to_len(lst, max_len, pad_item):
    """
    Args:
        lst: A python list
    Returns:
        pad_lst: A python list
    """
    return lst + [pad_item] * max(0, max_len-len(lst))

def pad_to_same_len(lst_of_lsts, pad_item):
    """
    Args:
        lst_of_lsts: A python iterable of python lists
    Returns:
        pad_lst: A python list of python lists
    """
    max_len = max(map(len, lst_of_lsts))
    return list(map(lambda lst: pad_to_len(lst, max_len, pad_item), lst_of_lsts))

def make_batch_half_with_it_of_lists(it_of_lists, max_batch_size, pad_item):
    # it_of_lists is iterator of python lists of numbers
    batch_half_it = group(it_of_lists, max_batch_size)
    # batch_half_it is iterator of python lists where each list contains a python lists of numbers
    batch_half_it = map(lambda batch_half: pad_to_same_len(batch_half, pad_item), batch_half_it)
    # batch_half_it is iterator of batch_halfs where each batch_half contains python lists of numbers
    return batch_half_it

def make_batch_half_with_it_of_nums(it_of_nums, max_batch_size, max_batch_width, pad_item):
    # it_of_nums is iterator of numbers
    if max_batch_width is None:
        it_of_lists = iter([list(it_of_nums)])
    else:
        it_of_lists = group(it_of_nums, max_batch_width)
    # it_of_lists is iterator of python lists of numbers
    return make_batch_half_with_it_of_lists(it_of_lists, max_batch_size, pad_item)

def make_batches_with_start_end(inputs, max_batch_size, token_item, pad_item):
    """
    Args:
        inputs: A python iterable of python lists of numbers
    Returns:
        batches: A python iterator of batches where each batch is a
            tuple (inputs batch, labels batch) and each batch part
            is a python list of lists of numbers
    """
    inputs_it, labels_it = itertools.tee(inputs)
    inputs_it = map(lambda inp: [token_item] + inp, inputs_it)
    labels_it = map(lambda inp: inp + [token_item], labels_it)
    inputs_batches_it = make_batch_half_with_it_of_lists(inputs_it, max_batch_size, pad_item)
    labels_batches_it = make_batch_half_with_it_of_lists(labels_it, max_batch_size, pad_item)
    return zip(inputs_batches_it, labels_batches_it)

def make_batches_long(inputs, max_batch_size, max_batch_width, pad_item):
    """
    Args:
        inputs: A python iterable of numbers
    Returns:
        batches: A python iterator of batches where each batch is a
            tuple (inputs batch, labels batch) and each batch part
            is a python list of lists of numbers
    """
    inputs_it, labels_it = itertools.tee(inputs)
    inputs_it = drop_last(inputs_it)
    labels_it = itertools.islice(labels_it, 1, None)
    inputs_batches_it = make_batch_half_with_it_of_nums(inputs_it, max_batch_size, max_batch_width, pad_item)
    labels_batches_it = make_batch_half_with_it_of_nums(labels_it, max_batch_size, max_batch_width, pad_item)
    return zip(inputs_batches_it, labels_batches_it)

def make_batches(inputs, batch_size, max_single_len, token_item, pad_item):
    """
    Args:
        inputs: A python iterable of python lists
    Returns:
        batches: A python iterator of tuples of numpy arrays
    """
    inputs = list(inputs)
    inputs_batches = flatmap(lambda input: [token_item] + input, inputs)
    labels_batches = flatmap(lambda input: input + [token_item], inputs)
    inputs_batches = single_make_batches(list(inputs_batches), batch_size, max_single_len, pad_item)
    labels_batches = single_make_batches(list(labels_batches), batch_size, max_single_len, pad_item)

    return zip(inputs_batches, labels_batches)

def get_len_to_make_divisible(lst, num):
    return math.ceil(len(lst) / num) * num

def single_make_batches(lst, batch_size, max_single_len, pad_item):
    """
    Args:
        lst: A python list of numbers
    Returns:
        batches: A python iterator of numpy arrays
    """
    # goal_len is smallest number that is >= len(lst) and is divisible by batch_size
    goal_len = get_len_to_make_divisible(lst, batch_size)
    padded_lst = pad_to_len(lst, goal_len, pad_item)
    long_batch = np.array(padded_lst).reshape(batch_size, -1)

    if max_single_len is not None:
        for i in range(0, long_batch.shape[1], max_single_len):
            yield long_batch[:, i:i+max_single_len]
    else:
        yield long_batch

def rmdir_if_exists(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

def string_to_bytes(s):
    return list(s.encode('utf-8'))

def bytes_to_string(byte_list):
    try:
        return bytes(byte_list).decode('utf-8')
    except UnicodeDecodeError:
        return bytes(byte_list)

def file_to_chars(f):
    """
    Args:
        f: File object such as from open('file.txt')
    """
    while True:
        c = f.read(1)
        if c:
            yield c
        else:
            break

def file_to_bytes(f):
    return flatmap(string_to_bytes, file_to_chars(f))
