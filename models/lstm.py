import my_logging as logging

logging.info('Importing tensorflow...')
import tensorflow as tf
logging.info('Imported tensorflow.')
tf.get_default_graph().finalize()

import numpy as np
from datetime import datetime
import uuid
import os
import json
import pickle
import itertools
from abc import ABC, abstractmethod


import config

import tools

"""
Directory structure:
    saved_models/
        shakespeare/
            baseline/
                3000/
                11433/
    saved_summaries/
        shakespeare/
            baseline/
"""

class Encoding(ABC):
    @abstractmethod
    def encode_input_for_training(self, inpt):
        """
        Returns:
            encoded_inpt: A python iterable (if long) or list (if short) of numbers
        """
        pass

    @abstractmethod
    def encode_input_for_testing(self, inpt):
        """
        Returns:
            encoded_inpt: A python list of integers.
        """
        pass

    @abstractmethod
    def make_batches_for_training(self, encoded_inputs):
        """
        Args:
            encoded_inputs: A python iterable of python iterables (if long) or lists (if short) of numbers
        Returns:
            batches_it: A python iterable of batches where each batch is a tuple
                (inputs_batch, labels_batch) and each batch half is either
                a python or numpy array of integers.
        """
        pass

    def count_batches_for_training(self, encoded_inputs):
        """
        Args:
            encoded_inputs: A python iterable of python iterables (if long) or lists (if short) of numbers
        Returns:
            count: An integer that is the number of batches
        """
        pass

    @abstractmethod
    def make_input_for_sample(self, encoded_inpt):
        """
        Args:
            encoded_inpt: A python list of integers.
        Returns:
            sample_inpt: A python list of integers.
        """
        pass

    @abstractmethod
    def make_batch_for_analysis(self, encoded_inpt):
        """
        Args:
            encoded_inpt: A python list of integers.
        Returns:
            batch: A python tuple (inputs_batch, labels_batch) and each batch half is either
                a python or numpy array of integers.
        """
        pass

    @abstractmethod
    def decode_output(self, outpt):
        """
        Args:
            outpt: A python iterable of integers.
        """
        pass

    @abstractmethod
    def decode_num(self, num):
        """
        Args:
            num: A python integer.
        """
        pass

    @abstractmethod
    def empty(self):
        """
        Returns:
            empty_single: An empty starting object to use in sample
        """
        pass

    @abstractmethod
    def get_load_params(self):
        """
        Returns:
            params: Either None or a tuple (encoding_name, encoding_args, encoding_kwargs)
                to be used when loading the encoding again.
        """
        pass

class BasicEncoding(Encoding):
    encoding_name = 'basic'

    def __init__(self, alphabet_size, use_long=False, **kwargs):
        # Note: If it seems like some of these options aren't being applied,
        # it's probably because there is already a saved file and
        # the options are being read from it
        self.alphabet_size = alphabet_size
        self.token_item = self.alphabet_size
        self.pad_item = self.alphabet_size+1
        self.effective_alphabet_size = self.alphabet_size+2
        self.use_long = use_long
        if self.use_long:
            max_batch_size = kwargs.get('max_batch_size', 20)
            max_batch_width = kwargs.get('max_batch_width', 200)
            skip_padding = kwargs.get('skip_padding', False)
            self.batch_maker = tools.LongBatchMaker(max_batch_size, max_batch_width, self.pad_item, skip_padding)
        else:
            max_batch_size = kwargs.get('max_batch_size', 10)
            self.batch_maker = tools.ShortBatchMaker(max_batch_size, self.token_item, self.pad_item)

    def encode_input(self, inpt):
        """
        Returns:
            encoded_inpt: A python list of integers.
        """
        return inpt

    def encode_input_for_training(self, inpt):
        """
        Returns:
            encoded_inpt: A python iterable (if long) or list (if short) of numbers
        """
        return self.encode_input(inpt)

    def encode_input_for_testing(self, inpt):
        """
        Returns:
            encoded_inpt: A python list of integers.
        """
        return self.encode_input(inpt)

    def make_batches_for_training(self, encoded_inputs):
        """
        Args:
            encoded_inputs: A python iterable of python iterables (if long) or lists (if short) of numbers
        Returns:
            batches_it: A python iterable of batches where each batch is a tuple
                (inputs_batch, labels_batch) and each batch half is either
                a python or numpy array of integers.
        """
        if self.use_long:
            encoded_inputs = tools.flat(encoded_inputs)
        batches = self.batch_maker.make_batches_for_training(encoded_inputs)
        return batches

    def count_batches_for_training(self, encoded_inputs):
        """
        Args:
            encoded_inputs: A python iterable of python iterables (if long) or lists (if short) of numbers
        Returns:
            count: An integer that is the number of batches
        """
        if self.use_long:
            encoded_inputs = tools.flat(encoded_inputs)
        count = self.batch_maker.count_batches_for_training(encoded_inputs)
        return count

    def make_input_for_sample(self, encoded_inpt):
        """
        Args:
            encoded_inpt: A python list of integers.
        Returns:
            sample_inpt: A python list of integers.
        """
        return self.batch_maker.make_input_for_sample(encoded_inpt)

    def make_batch_for_analysis(self, encoded_inpt):
        """
        Args:
            encoded_inpt: A python list of integers.
        Returns:
            batch: A python tuple (inputs_batch, labels_batch) and each batch half is either
                a python or numpy array of integers.
        """
        return self.batch_maker.make_batch_for_analysis(encoded_inpt)

    def decode_output(self, outpt):
        return outpt

    def decode_num(self, num):
        return num

    def empty(self):
        return []

    def get_load_params(self):
        return None

class AlphabetEncoding(BasicEncoding):
    encoding_name = 'alphabet'

    def __init__(self, alphabet, **kwargs):
        self.alphabet = alphabet
        self.entry_to_num = {}
        self.num_to_entry = []
        for num, entry in enumerate(self.alphabet):
            self.entry_to_num[entry] = num
            self.num_to_entry.append(entry)
        alphabet_size = len(self.num_to_entry)
        BasicEncoding.__init__(self, alphabet_size, **kwargs)

    def encode_entry(self, entry):
        return self.entry_to_num[entry]

    def decode_num(self, num):
        return self.num_to_entry[num]

    def encode_input(self, inpt):
        return list(map(self.encode_entry, inpt))

    def decode_output(self, outpt):
        return list(map(self.decode_num, outpt))

class StringEncoding(BasicEncoding):
    encoding_name = 'string'

    def __init__(self, **kwargs):
        BasicEncoding.__init__(self, 256, **kwargs)

    def encode_input(self, inpt):
        return tools.string_to_bytes(inpt)

    def decode_output(self, outpt):
        return tools.bytes_to_string(outpt)

    def decode_num(self, num):
        return chr(num)

    def empty(self):
        return ''

class StringAlphabetEncoding(AlphabetEncoding, StringEncoding):
    encoding_name = 'string-alphabet'

    def __init__(self, alphabet, **kwargs):
        AlphabetEncoding.__init__(self, alphabet, **kwargs)

    def decode_output(self, outpt):
        return ''.join(map(self.decode_num, outpt))

class TextFileEncoding(StringEncoding):
    encoding_name = 'text-file'

    def __init__(self, **kwargs):
        StringEncoding.__init__(self, use_long=True, **kwargs)

    def encode_input_for_training(self, inpt):
        """
        Args:
            inpt: A file path
        Returns:
            encoded_inpt: A python iterable of numbers
        """
        return tools.file_to_bytes(inpt)

class TextFileAlphabetEncoding(StringAlphabetEncoding, TextFileEncoding):
    encoding_name = 'text-file-alphabet'

    def __init__(self, alphabet, **kwargs):
        StringAlphabetEncoding.__init__(self, alphabet, use_long=True, **kwargs)

    def encode_input_for_training(self, inpt):
        """
        Args:
            inpt: A file path
        Returns:
            encoded_inpt: A python iterable of numbers
        """
        return map(self.encode_entry, tools.file_to_chars(inpt))

class TextFileAlphabetFileEncoding(TextFileAlphabetEncoding):
    encoding_name = 'text-file-alphabet-file'

    def __init__(self, path_for_alphabet, **kwargs):
        alphabet = tools.file_to_alphabet(path_for_alphabet)
        self.encoding_args = [alphabet]
        self.encoding_kwargs = kwargs
        TextFileAlphabetEncoding.__init__(self, *self.encoding_args, **self.encoding_kwargs)

    def get_load_params(self):
        return (TextFileAlphabetEncoding.encoding_name, self.encoding_args, self.encoding_kwargs)

encodings = {}
encoding_classes = [
    BasicEncoding,
    AlphabetEncoding,
    StringEncoding,
    StringAlphabetEncoding,
    TextFileEncoding,
    TextFileAlphabetEncoding,
    TextFileAlphabetFileEncoding
]
for encoding_class in encoding_classes:
    encodings[encoding_class.encoding_name] = encoding_class

class LSTMModelBase(object):
    def __init__(self, name, tag=None, saved_models_dir=None, saved_summaries_dir=None):
        self.uuid = uuid.uuid4()
        logging.info('Running __init__ %s' % self.uuid)
        self.name = name
        if tag is None:
            tag = config.TAG
        logging.info('Using tag %s and name %s' % (tag, name))
        if saved_models_dir is None:
            saved_models_dir = config.SAVED_MODELS_DIR
        if saved_summaries_dir is None:
            saved_summaries_dir = config.SAVED_SUMMARIES_DIR
        self.tag = tag
        self.saved_models_dir = saved_models_dir
        self.saved_summaries_dir = saved_summaries_dir
        self.training_steps = 0
        self.round_steps = 0
        self.sess = None

    def __del__(self):
        logging.info('Running __del__ %s' % self.uuid)
        self._close_sess_if_open()

    def _close_sess_if_open(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None
            logging.info('Closed session %s' % self.uuid)

    def _setup_encoding(self, encoding_name, *args, **kwargs):
        self.encoding = encodings[encoding_name](*args, **kwargs)
        load_params = self.encoding.get_load_params()
        if load_params is None:
            self.encoding_name = encoding_name
            self.encoding_args = args
            self.encoding_kwargs = kwargs
        else:
            self.encoding_name, self.encoding_args, self.encoding_kwargs = load_params
        self.alphabet_size = self.encoding.alphabet_size
        self.effective_alphabet_size = self.encoding.effective_alphabet_size

    def init_from_encoding(self, encoding_name, *args, **kwargs):
        logging.info('Initializing from encoding...')
        self._setup_encoding(encoding_name, *args, **kwargs)

        self._close_sess_if_open()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.int32, shape=(None, None), name='inputs')
            self.labels = tf.placeholder(tf.int32, shape=(None, None), name='labels')
            # inputs is list of batches where each batch is a list of integers in [0, alphabet_size]
            # inputs is tensor with dim batch_size x timesteps

            # inputs_padded and labels_padded are tensors with dim batch_size x (timesteps+1)

            self.inputs_one_hot = tf.one_hot(self.inputs, self.effective_alphabet_size, name='inputs_one_hot')
            # inputs_one_hot is a list of batches where each batch is a list of one hot encoded lists
            # inputs_one_hot is a tensor with dim batch_size x (timesteps+1) x (alphabet_size+1)

            self.state_sizes = [768] * 2

            # def make_init_state(i, state_size):
            #     return tf.placeholder(tf.float32, [2, None, state_size], name='init_state_' + str(i))

            # def make_lstm_state_tuple(init_state):
            #     c, h = tf.unstack(init_state)
            #     return tf.contrib.rnn.LSTMStateTuple(c, h)

            # self.init_states = tuple(make_init_state(i, state_size) for i, state_size in enumerate(self.state_sizes))
            # rnn_init_state = tuple(make_lstm_state_tuple(init_state) for init_state in self.init_states)

            lstm_cells = list(map(tf.contrib.rnn.BasicLSTMCell, self.state_sizes))
            lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
            rnn_output, rnn_states = tf.nn.dynamic_rnn(lstm, self.inputs_one_hot, dtype=tf.float32)
            tf.summary.histogram('rnn_output', rnn_output)
            # rnn_outputs is a tensor with dim batch_size x (timesteps+1) x (alphabet_size+1)
            # state is a list of tensors with dim batch_size x state_size

            # def make_output_state(i, state_tuple):
            #     c = state_tuple.c
            #     h = state_tuple.h
            #     return tf.identity(tf.stack([c, h]), name='output_state_' + str(i))

            # self.output_states = tuple(make_output_state(i, state) for i, state in enumerate(rnn_states))

            logits = tf.contrib.layers.fully_connected(rnn_output, self.effective_alphabet_size, activation_fn=None)
            self.logits = tf.identity(logits, name='logits')
            tf.summary.histogram('logits', self.logits)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels,
                    logits=self.logits)
            self.losses = tf.identity(losses, name='losses')
            tf.summary.histogram('losses', self.losses)
            # losses is a tensor with dim batch_size x (timestamps+1)

            self.loss_mean = tf.reduce_mean(self.losses, name='loss_mean')
            tf.summary.scalar('loss_mean', self.loss_mean)
            self.loss_max = tf.reduce_max(self.losses, name='loss_max')
            tf.summary.scalar('loss_max', self.loss_max)
            self.loss_min = tf.reduce_min(self.losses, name='loss_min')
            tf.summary.scalar('loss_min', self.loss_min)
            self.loss_sum = tf.reduce_sum(self.losses, name='loss_sum')
            tf.summary.scalar('loss_sum', self.loss_sum)
            # loss_* is a tensor with a single float

            self.optimize = tf.train.AdamOptimizer().minimize(self.loss_mean, name='optimize')
            # self.optimize = tf.train.AdadeltaOptimizer(0.03).minimize(self.loss_mean, name='optimize')

            self.probabilities = tf.identity(tf.nn.softmax(self.logits), name='probabilities')
            tf.summary.histogram('probabilities', self.probabilities)

            self.next_probabilities = tf.identity(self.probabilities[:, -1], name='next_probabailities')

            self.saver = tf.train.Saver()

            self.summary = tf.identity(tf.summary.merge_all(), name='summary')

            self.glob_var_init = tf.global_variables_initializer()

            self.graph.finalize()

            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.glob_var_init)

        logging.info('Initialized from encoding.')

    def init_from_file(self, from_name=None, training_steps=None):
        if from_name is None:
            from_name = self.name
        training_steps_dir = os.path.join(self.saved_models_dir, from_name, self.tag)
        if training_steps is None:
            training_steps = tools.get_latest_in_dir(training_steps_dir, key=int)
        save_dir = os.path.join(training_steps_dir, str(training_steps))
        file_path = os.path.join(save_dir, 'data')

        logging.info('Loading model from file %s' % file_path)

        extra_file_path = file_path + '.pkl'
        with open(extra_file_path, 'rb') as f:
            extra = pickle.load(f)

        encoding_name = extra['encoding_name']
        encoding_args = extra['encoding_args']
        encoding_kwargs = extra['encoding_kwargs']
        self._setup_encoding(encoding_name, *encoding_args, **encoding_kwargs)

        self.state_sizes = extra['state_sizes']
        self.training_steps = extra.get('training_steps', int(training_steps))
        self.round_steps = extra.get('round_steps', 0)

        self._close_sess_if_open()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)

            meta_file_path = file_path + '.meta'
            self.saver = tf.train.import_meta_graph(meta_file_path)

            self.saver.restore(self.sess, file_path)

            self.inputs = self.graph.get_tensor_by_name('inputs:0')
            self.labels = self.graph.get_tensor_by_name('labels:0')
            self.logits = self.graph.get_tensor_by_name('logits:0')
            self.losses = self.graph.get_tensor_by_name('losses:0')
            self.loss_mean = self.graph.get_tensor_by_name('loss_mean:0')
            self.loss_max = self.graph.get_tensor_by_name('loss_max:0')
            self.loss_min = self.graph.get_tensor_by_name('loss_min:0')
            self.loss_sum = self.graph.get_tensor_by_name('loss_sum:0')
            self.optimize = self.graph.get_operation_by_name('optimize')
            self.probabilities = self.graph.get_tensor_by_name('probabilities:0')
            self.next_probabilities = tools.get_tensor_by_name_if_exists(self.graph, 'next_probabilities:0')

            # self.init_states = tuple(self.graph.get_tensor_by_name('init_state_' + str(i) + ':0')
            #         for i in range(len(self.state_sizes)))
            # self.output_states = tuple(self.graph.get_tensor_by_name('output_state_' + str(i) + ':0')
            #         for i in range(len(self.state_sizes)))

            self.summary = self.graph.get_tensor_by_name('summary:0')

            self.graph.finalize()

        logging.info('Loaded model from file %s' % file_path)

    def init_from_file_or_encoding(self, *args, **kwargs):
        logging.info('Initializing %s from file or encoding...' % self.name)
        try:
            self.init_from_file()
        except FileNotFoundError as e:
            logging.info('File not found for %s, initializing from encoding...' % self.name)
            self.init_from_encoding(*args, **kwargs)

    def save_to_file(self):
        save_dir = os.path.join(self.saved_models_dir, self.name, self.tag, str(self.training_steps))
        # Allow overwriting an existing save because we don't lose any data here
        tools.rmdir_if_exists(save_dir)
        os.makedirs(save_dir)

        file_path = os.path.join(save_dir, 'data')

        logging.info('Saving model to file %s' % file_path)

        extra = {
            'encoding_name': self.encoding_name,
            'encoding_args': self.encoding_args,
            'encoding_kwargs': self.encoding_kwargs,
            'state_sizes': self.state_sizes,
            'training_steps': self.training_steps,
            'round_steps': self.round_steps
        }
        extra_file_path = file_path + '.pkl'
        with open(extra_file_path, 'xb') as f:
            pickle.dump(extra, f)

        self.saver.save(self.sess, file_path)

        logging.info('Saved model to file %s' % file_path)

    # def _run_batch_with_state(self, tensors, batch, curr_states=None):
    #     """
    #     batch: tuple (inputs_batch, labels_batch)
    #     Returns:
    #         A list of outputs [new_states] + tensor_outputs
    #     """
    #     return self._run_batch(tensors + [self.output_states], batch, curr_states)

    def _run_batch(self, tensors, batch, curr_states=None):
        """
        batch: tuple (inputs_batch, labels_batch)
        Returns:
            A list of outputs tensor_outputs
        """
        inputs, labels = batch
        return self._run(tensors, inputs, labels, curr_states)

    def _run(self, tensors, inputs, labels=None, curr_states=None):
        # print(inputs, labels)

        if len(inputs) == 0 or len(inputs[0]) == 0 or (labels is not None and (len(labels) == 0 or len(labels[0]) == 0)):
            raise ValueError('Inputs and labels cannot be empty.')

        # if curr_states is None:
        #     batch_size = len(inputs)
        #     def make_current_state(state_size):
        #         return np.zeros((2, batch_size, state_size))
        #     curr_states = tuple(map(make_current_state, self.state_sizes))

        feed_dict = {self.inputs: inputs}
        if labels is not None:
            feed_dict[self.labels] = labels

        with self.graph.as_default():
            # logging.info('Start run...')
            results = self.sess.run(tensors, feed_dict=feed_dict)
            # logging.info('End run.')
            return results

    def _decode_output(self, outpt):
        return self.encoding.decode_output(outpt)

    def _decode_if_ok(self, num):
        return self.encoding.decode_num(num) if num < self.alphabet_size else num

    def _decode_output_to_list(self, outpt):
        return list(map(self._decode_if_ok, outpt))

    def encode_and_make_batches_for_training(self, inputs):
        inputs = map(self.encoding.encode_input_for_training, inputs)
        batches = self.encoding.make_batches_for_training(inputs)
        return batches

    def encode_and_count_batches_for_training(self, inputs):
        inputs = map(self.encoding.encode_input_for_training, inputs)
        count = self.encoding.count_batches_for_training(inputs)
        return count

    def train(self, inputs, autosave=None, cont=False, count=False, save_graph=True):
        """
        Args:
            inputs: A python iterable of things that encode to python iterables (if long) or lists (if short) of numbers
        """
        if count:
            logging.info('Counting enabled! This should only be used if inputs can be iterated over multiple times.')
            logging.info('Counting number of steps...')
            # Only use count if inputs and the items in it can be iterated over multiple times
            total_round_steps = self.encode_and_count_batches_for_training(inputs)
            logging.info('Number of steps is %s' % total_round_steps)

        batches = self.encode_and_make_batches_for_training(inputs)
        if cont:
            logging.info('Continuing enabled! Starting from round %s' % self.round_steps)
            # Note: round_steps is the number of rounds completed, so
            # is also the index of the next round to be done
            batches = itertools.islice(batches, self.round_steps, None)
        else:
            self.round_steps = 0

        save_dir = os.path.join(self.saved_summaries_dir, self.name, self.tag)
        os.makedirs(save_dir, exist_ok=True)

        summaries_dir = save_dir
        logging.info('Using summaries dir %s' % summaries_dir)
        if save_graph and self.training_steps == 0:
            logging.info('Saving graph...')
            # Only write graph during the first training attempt
            summary_writer = tf.summary.FileWriter(summaries_dir, self.graph)
        else:
            summary_writer = tf.summary.FileWriter(summaries_dir)

        logging.info('Starting training...')
        curr_states = None
        starting_i = self.training_steps
        starting_round = self.round_steps
        # Need this i set for if there ends up being no batches due to continuing
        i = starting_i
        try:
            for batch in batches:
                with tools.DelayedKeyboardInterrupt():
                    # Use key instead of i or self.training_steps
                    # to prevent the unlikely race condition
                    # of hitting the interrupt after
                    # the key is updated but before entering the
                    # DelayedKeyboardInterrupt section
                    self.training_steps += 1
                    self.round_steps += 1
                    i = self.training_steps
                    # i, training_steps will be steps completed after the run_batch
                    if i-1 == starting_i:
                        loss_max, loss_mean, loss_min = self._run_batch(
                                [self.loss_max, self.loss_mean, self.loss_min], batch, curr_states)
                        logging.info('Starting values: steps: %s round: %s loss max: %s mean: %s min: %s'
                                % (starting_i, starting_round, loss_max, loss_mean, loss_min))
                        if count:
                            time_rem = tools.TimeRemaining(total_round_steps, starting_round)
                    if i % 10 != 0:
                        _ = self._run_batch([self.optimize], batch, curr_states)
                    else:
                        _, summary, loss_max, loss_mean, loss_min = self._run_batch(
                                [self.optimize, self.summary, self.loss_max, self.loss_mean, self.loss_min], batch, curr_states)
                        summary_writer.add_summary(summary, i)
                        # Note: The printed numbers are the numbers from before the optimization update happens
                        # Note: Step numbers start from 1
                        if count:
                            time_rem_str = time_rem.get_str(self.round_steps)
                            logging.info('Step %s round %s/%s: loss max: %s mean: %s min: %s time rem: %s'
                                    % (i, self.round_steps, total_round_steps, loss_max, loss_mean, loss_min, time_rem_str))
                        else:
                            logging.info('Step %s round %s: loss max: %s mean: %s min: %s'
                                    % (i, self.round_steps, loss_max, loss_mean, loss_min))
                        #losses, probabilities = self.sess.run([self.losses, self.probabilities], feed_dict={self.inputs: batch})
                        #logging.info('Step %s: losses: %s probs: %s' % (i, losses, probabilities))
                    # curr_states = new_states
                    if autosave is not None and (autosave is not True and i % autosave == 0):
                        self.save_to_file()
        except KeyboardInterrupt:
            logging.info('Cancelling training...')
            logging.info('Saved summaries to %s' % summaries_dir)
            if autosave is not None and (autosave is True or i % autosave != 0):
                # Save the last one only if it hasn't already been saved
                self.save_to_file()
            tools.exit(22)
        else:
            self.round_steps = 0
            logging.info('Saved summaries to %s' % summaries_dir)
            if autosave is not None:
                # Always save the last one because round_steps has changed
                self.save_to_file()
            logging.info('Done training.')

    def _make_probs_dec(self, probs):
        probs_dec = [(self._decode_if_ok(i), prob) for i, prob in enumerate(probs)]
        probs_dec.sort(key=lambda s: s[1], reverse=True)
        probs_dec = probs_dec[:7]
        return probs_dec

    def sample(self, starting=None, max_num=50):
        """
        Args:
            starting: A thing that encodes to a python iterable of numbers
        """
        if starting is None:
            starting = self.encoding.empty()
        curr = list(self.encoding.encode_input_for_testing(starting))
        curr_output = list(curr)
        try:
            for i in range(max_num):
                with tools.DelayedKeyboardInterrupt():
                    encoded_inpt = self.encoding.make_input_for_sample(curr)
                    if self.next_probabilities is not None:
                        probs_batch = self._run(self.next_probabilities, [encoded_inpt])
                        probs = probs_batch[0]
                    else:
                        probs_batch = self._run(self.probabilities, [encoded_inpt])
                        probs = probs_batch[0][-1]
                    next_int = np.random.choice(self.effective_alphabet_size, 1, p=probs).item()
                    curr_dec = self._decode_output_to_list(curr[-5:])
                    next_dec = self._decode_if_ok(next_int)
                    probs_dec = self._make_probs_dec(probs)
                    logging.info('Step %s: curr: %s next: %s probs: %s' % (i, curr_dec, repr(next_dec), probs_dec))
                    if next_int == self.encoding.token_item:
                        break
                    if next_int < self.alphabet_size:
                        curr_output.append(next_int)
                    curr.append(next_int)
        except KeyboardInterrupt:
            logging.info('Cancelling sample...')
        return self.encoding.decode_output(curr_output)

    def analyze(self, inpt):
        """
        Args:
            starting: A thing that encodes to a python iterable of numbers
        """
        lst = list(self.encoding.encode_input_for_testing(inpt))
        batch = self.encoding.make_batch_for_analysis(lst)
        labels_batch, losses_batch, probs_batch = self._run_batch(
                [self.labels, self.losses, self.probabilities], batch)
        labels = labels_batch[0].tolist()
        losses = losses_batch[0].tolist()
        probs_list = map(self._make_probs_dec, probs_batch[0])
        lst_dec = self._decode_output_to_list(labels)
        for item, loss, probs in zip(lst_dec, losses, probs_list):
            print('%-5s %-11.8f %s' % (repr(item), loss, probs))

class LSTMModelEncoding(LSTMModelBase):
    def __init__(self, name, encoding_name, *args, tag=None,
            saved_models_dir=None, saved_summaries_dir=None, **kwargs):
        super(LSTMModelEncoding, self).__init__(name, tag=tag,
                saved_models_dir=saved_models_dir, saved_summaries_dir=saved_summaries_dir)
        self.init_from_file_or_encoding(encoding_name, *args, **kwargs)

class LSTMModel(LSTMModelEncoding):
    def __init__(self, name, *args, **kwargs):
        LSTMModelEncoding.__init__(self, name, 'basic', *args, **kwargs)

class LSTMModelAlphabet(LSTMModelEncoding):
    def __init__(self, name, *args, **kwargs):
        LSTMModelEncoding.__init__(self, name, 'alphabet', *args, **kwargs)

class LSTMModelString(LSTMModelEncoding):
    def __init__(self, name, *args, **kwargs):
        LSTMModelEncoding.__init__(self, name, 'string', *args, **kwargs)

class LSTMModelStringAlphabet(LSTMModelEncoding):
    def __init__(self, name, *args, **kwargs):
        LSTMModelEncoding.__init__(self, name, 'string-alphabet', *args, **kwargs)

class LSTMModelTextFile(LSTMModelEncoding):
    def __init__(self, name, *args, **kwargs):
        LSTMModelEncoding.__init__(self, name, 'text-file', *args, **kwargs)

class LSTMModelTextFileAlphabet(LSTMModelEncoding):
    def __init__(self, name, *args, **kwargs):
        LSTMModelEncoding.__init__(self, name, 'text-file-alphabet', *args, **kwargs)

class LSTMModelTextFileAlphabetFile(LSTMModelEncoding):
    def __init__(self, name, *args, **kwargs):
        LSTMModelEncoding.__init__(self, name, 'text-file-alphabet-file', *args, **kwargs)

class LSTMModelFromFile(LSTMModelBase):
    def __init__(self, name, tag=None, saved_models_dir=None, saved_summaries_dir=None, **kwargs):
        super(LSTMModelFromFile, self).__init__(name, tag=tag,
                saved_models_dir=saved_models_dir, saved_summaries_dir=saved_summaries_dir)
        self.init_from_file(**kwargs)

def list_all(tag=config.TAG):
    # saved_models/shakespeare/baseline/3000
    glob_str = os.path.join(config.SAVED_MODELS_DIR, '*', tag, '*')
    tools.print_glob(glob_str)

def list_names():
    glob_str = os.path.join(config.SAVED_MODELS_DIR, '*')
    tools.print_glob(glob_str)

def list_instances(name, tag=config.TAG):
    glob_str = os.path.join(config.SAVED_MODELS_DIR, name, tag, '*')
    tools.print_glob(glob_str)
