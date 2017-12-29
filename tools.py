import my_logging as logging

import shutil
import math
import numpy as np
import itertools
import signal
import os
import sys
import glob
import uuid
from datetime import datetime, timedelta

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

def debug(item):
    print(item)
    return item

def debug_it(it):
    lst = list(it)
    print(lst)
    return iter(lst)

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

def len_iter(it):
    return sum(1 for _ in it)

class LongBatchMaker(object):
    def __init__(self, max_batch_size, max_batch_width, pad_item, skip_padding=False):
        logging.info('Using skip padding %s' % skip_padding)
        self.inputs_array = np.zeros((max_batch_size, max_batch_width), dtype=np.int32)
        self.labels_array = np.zeros((max_batch_size, max_batch_width), dtype=np.int32)
        self.max_batch_size = max_batch_size
        self.max_batch_width = max_batch_width
        self.pad_item = pad_item
        self.skip_padding = skip_padding

    def make_batches_for_training(self, inputs):
        """
        Args:
            inputs: A python iterable of numbers
        Returns:
            batches: A python iterator of batches where each batch is a
                tuple (inputs batch, labels batch) and each batch part
                is a numpy array of numbers
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
            if batch_index == 0:
                # Use :1 to keep it a 2d array
                yield (self.inputs_array[:1, :in_batch_index], self.labels_array[:1, :in_batch_index])
            else:
                if in_batch_index != 0 and not self.skip_padding:
                    for in_batch_index in range(in_batch_index, self.max_batch_width):
                        self.inputs_array[batch_index, in_batch_index] = self.pad_item
                        self.labels_array[batch_index, in_batch_index] = self.pad_item
                    batch_index += 1
                    in_batch_index = 0
                yield (self.inputs_array[:batch_index], self.labels_array[:batch_index])

    def count_batches_for_training(self, inputs):
        """
        Args:
            inputs: A python iterable of numbers. inputs cannot be an iterator!
        Returns:
            count: An integer that is the number of batches that will be returned
        """
        return math.ceil(len_iter(inputs)/(self.max_batch_size*self.max_batch_width))

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

class ArrayManager(object):
    """
    Array manager strategy:
        For every inpt item:
            If pad_item is None and curr line len + 1 > prev line length:
                send previous lines and clear
            If curr line len + 1 > max_width:
                move to next line
            If curr line len + 1 > curr_width:
                expand array
            Add inpt item horizontally
        When an end_of_inpt is received,
            If pad_item is None
                If curr line length < prev line length,
                    send previous lines and shift curr line to top
                else
                    move to next line
            else
                pad until curr_width
                move to next line
        When an end_of_inputs is received,
            Send previous lines and clear
    """

    def __init__(self, max_height, max_width=None, pad_item=None):
        if max_width is None:
            max_width = math.inf
        self.max_height = max_height
        self.max_width = max_width
        self.curr_width = max_width if max_width != math.inf else 2
        self.array = np.zeros((max_height, self.curr_width), dtype=np.int32)
        self.max_prev_hori_len = None
        self.curr_hori_len = 0
        # Number of completed vertical lines
        self.curr_vert_len = 0
        self.pad_item = pad_item
        self.uuid = str(uuid.uuid4())
        logging.debug(self.uuid + ' Construct')

    def move_to_next_line(self):
        logging.debug(self.uuid + ' Move to next line')
        self.curr_vert_len += 1
        if self.max_prev_hori_len is None:
            self.max_prev_hori_len = self.curr_hori_len
        else:
            self.max_prev_hori_len = max(self.max_prev_hori_len, self.curr_hori_len)
        logging.debug(self.uuid + ' max prev hori len %s' % self.max_prev_hori_len)
        self.curr_hori_len = 0

    def clear(self):
        logging.debug(self.uuid + ' Clearing')
        self.curr_vert_len = 0
        self.max_prev_hori_len = None
        self.curr_hori_len = 0

    def expand_array(self):
        logging.debug(self.uuid + ' Expanding array')
        new_width = self.curr_width*2
        if self.pad_item is not None:
            new_array = np.full((self.max_height, new_width), self.pad_item, dtype=np.int32)
        else:
            new_array = np.zeros((self.max_height, new_width), dtype=np.int32)
        new_array[:self.array.shape[0], :self.array.shape[1]] = self.array
        self.array = new_array
        self.curr_width = new_width
        logging.debug(self.uuid + ' New width %s' % new_width)

    def add_ind(self, ind):
        logging.debug(self.uuid + ' add ind %s curr vert %s curr hori %s' % (ind, self.curr_vert_len, self.curr_hori_len))
        if self.curr_hori_len + 1 > self.max_width:
            logging.debug(self.uuid + ' passed max width')
            # If curr line is at max_width, move to next line
            self.move_to_next_line()
            if self.curr_vert_len == self.max_height:
                logging.debug(self.uuid + ' yielding curr vert %s prev hori %s' % (self.curr_vert_len, self.max_prev_hori_len))
                yield self.array[:self.curr_vert_len, :self.max_prev_hori_len]
                self.clear()
        if self.pad_item is None and self.max_prev_hori_len is not None and self.curr_hori_len + 1 > self.max_prev_hori_len:
            logging.debug(self.uuid + ' no pad item and line too long')
            # If there is no padding and this line is too long,
            # send previous lines first
            curr_line_idx = self.curr_vert_len
            curr_hori_len = self.curr_hori_len
            logging.debug(self.uuid + ' yielding curr vert %s prev hori %s' % (self.curr_vert_len, self.max_prev_hori_len))
            yield self.array[:self.curr_vert_len, :self.max_prev_hori_len]
            self.clear()
            self.array[0] = self.array[curr_line_idx]
            self.curr_hori_len = curr_hori_len
        if self.curr_hori_len + 1 > self.curr_width:
            self.expand_array()
        logging.debug(self.uuid + ' added at curr vert %s curr hori %s' % (self.curr_vert_len, self.curr_hori_len))
        self.array[self.curr_vert_len, self.curr_hori_len] = ind
        self.curr_hori_len += 1

    def end_of_inpt(self):
        logging.debug(self.uuid + ' end of inpt curr vert %s curr hori %s' % (self.curr_vert_len, self.curr_hori_len))
        if self.pad_item is None:
            if self.max_prev_hori_len is not None and self.curr_hori_len < self.max_prev_hori_len:
                # If there is no padding and this line is shorter than previous ones,
                # send previous lines first
                curr_line_idx = self.curr_vert_len
                curr_hori_len = self.curr_hori_len
                logging.debug(self.uuid + ' yielding curr vert %s prev hori %s' % (self.curr_vert_len, self.max_prev_hori_len))
                yield self.array[:self.curr_vert_len, :self.max_prev_hori_len]
                self.clear()
                self.array[0] = self.array[curr_line_idx]
                self.curr_hori_len = curr_hori_len
        else:
            # If there is padding, pad the end of the line
            self.array[self.curr_vert_len, self.curr_hori_len:] = self.pad_item
        self.move_to_next_line()
        if self.curr_vert_len == self.max_height:
            logging.debug(self.uuid + ' yielding curr vert %s prev hori %s' % (self.curr_vert_len, self.max_prev_hori_len))
            yield self.array[:self.curr_vert_len, :self.max_prev_hori_len]
            self.clear()

    def end_of_inputs(self):
        logging.debug(self.uuid + ' end of inputs curr vert %s curr hori %s' % (self.curr_vert_len, self.curr_hori_len))
        # Assume end_of_inpt was already called
        assert self.curr_hori_len == 0
        if self.curr_vert_len > 0:
            logging.debug(self.uuid + ' yielding curr vert %s prev hori %s' % (self.curr_vert_len, self.max_prev_hori_len))
            yield self.array[:self.curr_vert_len, :self.max_prev_hori_len]
            self.clear()

class BatchMaker(object):
    def __init__(self, max_batch_size, max_batch_width=None, token_item=None, pad_item=None):
        self.inputs_array = ArrayManager(max_batch_size, max_batch_width, pad_item)
        self.labels_array = ArrayManager(max_batch_size, max_batch_width, pad_item)
        self.token_item = token_item

    def make_batches_for_training(self, inputs):
        """
        Args:
            inputs: A python iterable of python iterables of numbers
        Returns:
            batches: A python iterator of batches where each batch is a
                tuple (inputs batch, labels batch) and each batch part
                is a numpy array of numbers
        """
        for inpt in inputs:
            if self.token_item is not None:
                inpt = itertools.chain([self.token_item], inpt, [self.token_item])
            inpt, labl = itertools.tee(inpt)
            inpt = drop_last(inpt)
            labl = itertools.islice(labl, 1, None)
            for ind, lab in zip(inpt, labl):
                logging.debug('ind %s lab %s' % (ind, lab))
                # Use zip_longest to ensure that iterators are exhausted even when one is empty
                # Using list() doesn't work because it exhausts the iterators too quickly
                for batch_tuple in itertools.zip_longest(self.inputs_array.add_ind(ind), self.labels_array.add_ind(lab)):
                    yield batch_tuple
            for batch_tuple in itertools.zip_longest(self.inputs_array.end_of_inpt(), self.labels_array.end_of_inpt()):
                yield batch_tuple
        for batch_tuple in itertools.zip_longest(self.inputs_array.end_of_inputs(), self.labels_array.end_of_inputs()):
            yield batch_tuple

    def count_batches_for_training(self, inputs):
        """
        Args:
            inputs: A python iterable of python lists of numbers
        Returns:
            count: An integer that is the number of batches that will be returned
        """
        return math.ceil(len_iter(inputs)/self.max_batch_size)

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

def is_string_or_bytes(s):
    return isinstance(s, str) or isinstance(s, bytes)

def file_to_chars(path):
    """
    Args:
        path: A file path
    """
    with open(path, 'r') as f:
        while True:
            c = f.read(1)
            if c:
                yield c
            else:
                break

def file_to_bytes(path):
    return flatmap(string_to_bytes, file_to_chars(path))

def file_to_alphabet(path):
    lst = list(set(file_to_chars(path)))
    lst.sort()
    return lst

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
        raise FileNotFoundError('Path %s is empty' % path)
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

def remove_microsecs(delta):
    return delta - timedelta(microseconds=delta.microseconds)

class TimeRemaining(object):
    def __init__(self, total_steps, start_steps=0):
        self.start_time = datetime.now()
        self.total_steps = total_steps
        self.start_steps = start_steps

    def get_str(self, curr_steps):
        curr_time = datetime.now()
        time_passed = curr_time - self.start_time
        steps_passed = curr_steps - self.start_steps
        steps_remaining = self.total_steps - curr_steps
        if steps_passed == 0:
            return '?'
        else:
            time_per_step = time_passed / steps_passed
            time_remaining = time_per_step * steps_remaining
            time_remaining = remove_microsecs(time_remaining)
            return str(time_remaining)

def exit(code):
    sys.exit(code)

def get_or_set(data, name, value):
    # Set the value like this so when the model is saved, the value actually used
    # is restored
    if name not in data:
        data[name] = value
    return data[name]
