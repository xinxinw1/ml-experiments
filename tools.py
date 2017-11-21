import shutil
import math
import numpy as np

def group(lst, n):
    """
    Args:
        lst: A python list
    Returns:
        grps: A python iterator of python lists
    """
    return (lst[i:i+n] for i in range(0, len(lst), n))

def flatmap(f, lst):
    """
    Args:
        lst: A python iterable
    Returns:
        new_lst: A python iterator
    """
    return (inner_item for item in lst for inner_item in f(item))

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
