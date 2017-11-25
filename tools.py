import my_logging as logging

import shutil
import math
import numpy as np
import itertools
import signal
import os
import glob
from datetime import datetime

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

class LongBatchMaker(object):
    def __init__(self, max_batch_size, max_batch_width, pad_item):
        self.inputs_array = np.zeros((max_batch_size, max_batch_width), dtype=np.int32)
        self.labels_array = np.zeros((max_batch_size, max_batch_width), dtype=np.int32)
        self.max_batch_size = max_batch_size
        self.max_batch_width = max_batch_width
        self.pad_item = pad_item

    def make_batches_for_training(self, inputs):
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
        i = 0
        for input_num, label_num in zip(inputs_it, labels_it):
            batch_index = i // self.max_batch_width
            in_batch_index = i % self.max_batch_width
            self.inputs_array[batch_index, in_batch_index] = input_num
            self.labels_array[batch_index, in_batch_index] = label_num
            i += 1
            if i == self.max_batch_size * self.max_batch_width:
                yield (self.inputs_array, self.labels_array)
                i = 0
        if i != 0:
            batch_index = i // self.max_batch_width
            in_batch_index = i % self.max_batch_width
            if in_batch_index != 0:
                for in_batch_index in range(in_batch_index, self.max_batch_width):
                    self.inputs_array[batch_index, in_batch_index] = self.pad_item
                    self.labels_array[batch_index, in_batch_index] = self.pad_item
                batch_index += 1
                in_batch_index = 0
            yield (self.inputs_array[:batch_index], self.labels_array[:batch_index])

    def make_input_for_sample(self, inpt):
        """
        Args:
            inpt: A python list of integers.
        Returns:
            sample_inpt: A python list of integers.
        """
        return inpt

    def make_batch_for_analysis(self, inpt):
        """
        Args:
            inpt: A python list of integers.
        Returns:
            batch: A python tuple (inputs_batch, labels_batch) and each batch half is either
                a python or numpy array of integers.
        """
        inputs = [inpt[:-1]]
        labels = [inpt[1:]]
        return (inputs, labels)

class ShortBatchMaker(object):
    def __init__(self, max_batch_size, token_item, pad_item):
        self.max_batch_size = max_batch_size
        self.token_item = token_item
        self.pad_item = pad_item

    def make_batches_for_training(self, inputs):
        """
        Args:
            inputs: A python iterable of python lists of numbers
        Returns:
            batches: A python iterator of batches where each batch is a
                tuple (inputs batch, labels batch) and each batch part
                is a python list of lists of numbers
        """
        return make_batches_with_start_end(inputs, self.max_batch_size,
                self.token_item, self.pad_item)

    def make_input_for_sample(self, inpt):
        """
        Args:
            inpt: A python list of integers.
        Returns:
            sample_inpt: A python list of integers.
        """
        return [self.token_item] + inpt

    def make_batch_for_analysis(self, inpt):
        """
        Args:
            inpt: A python list of integers.
        Returns:
            batch: A python tuple (inputs_batch, labels_batch) and each batch half is either
                a python or numpy array of integers.
        """
        inputs = [[self.token_item] + inpt]
        labels = [inpt + [self.token_item]]
        return (inputs, labels)


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

def date_str():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_tensor_by_name_if_exists(graph, name):
    try:
        return graph.get_tensor_by_name(name)
    except KeyError:
        return None

class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.info('KeyboardInterrupt received.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

def get_latest_in_dir(path, glb='*', key=None):
    paths = glob.glob(os.path.join(path, glb))
    if not paths:
        raise ValueError('Path %s is empty' % path)
    paths = map(os.path.basename, paths)
    if key is not None:
        return max(paths, key=key)
    else:
        return max(paths)

def remove_from_end(s, suffix):
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return s

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    "Sort the given list in the way that humans expect."
    return sorted(l, key=alphanum_key)

def print_glob(glob_str):
    paths = sort_nicely(glob.glob(glob_str))
    for path in paths:
        print(path)

# todo: make alphabet
