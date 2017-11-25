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
from abc import ABC, abstractmethod


import config

import tools

class Encoding(ABC):
    @abstractmethod
    def __init__(self):
        pass

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

class BasicEncoding(Encoding):
    def __init__(self, alphabet_size, use_long=False, **kwargs):
        self.alphabet_size = alphabet_size
        self.token_item = self.alphabet_size
        self.pad_item = self.alphabet_size+1
        self.effective_alphabet_size = self.alphabet_size+2
        self.use_long = use_long
        if self.use_long:
            max_batch_size = kwargs.get('max_batch_size', 2)
            max_batch_width = kwargs.get('max_batch_width', 200)
            self.batch_maker = tools.LongBatchMaker(max_batch_size, max_batch_width, self.pad_item)
        else:
            max_batch_size = kwargs.get('max_batch_size', 10)
            self.batch_maker = tools.ShortBatchMaker(max_batch_size, self.token_item, self.pad_item)

    def encode_input(self, inpt):
        """
        Returns:
            encoded_inpt: A python iterable of integers.
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

class StringEncoding(BasicEncoding):
    def __init__(self, **kwargs):
        super(StringEncoding, self).__init__(256, **kwargs)

    def encode_input(self, inpt):
        return tools.string_to_bytes(inpt)

    def decode_output(self, outpt):
        return tools.bytes_to_string(outpt)

    def decode_num(self, num):
        return chr(num)

    def empty(self):
        return ''

class StringAlphabetEncoding(BasicEncoding):
    def __init__(self, alphabet_size=26, **kwargs):
        assert alphabet_size <= 26
        super(StringAlphabetEncoding, self).__init__(alphabet_size, **kwargs)

    def encode_char(self, c):
        o = ord(c) - ord('a')
        if not 0 <= o < self.alphabet_size:
            raise ValueError('Expected char %s to be in alphabet size %s' % (c, self.alphabet-size))
        return o

    def encode_input(self, inpt):
        return list(map(self.encode_char, inpt))

    def decode_output(self, outpt):
        return ''.join(map(self.decode_num, outpt))

    def decode_num(self, num):
        return chr(num+ord('a'))

    def empty(self):
        return ''

class TextFileEncoding(StringEncoding):
    def __init__(self, **kwargs):
        super(TextFileEncoding, self).__init__(use_long=True, **kwargs)

    def encode_input_for_training(self, inpt):
        """
        Args:
            inpt: A file handle
        """
        return tools.file_to_bytes(inpt)

encodings = {
    'basic': BasicEncoding,
    'string': StringEncoding,
    'string-alphabet': StringAlphabetEncoding,
    'text-file': TextFileEncoding
}

class LSTMModelBase(object):
    def __init__(self, name):
        self.uuid = uuid.uuid4()
        logging.info('Running __init__ %s' % self.uuid)
        self.name = name
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
        self.encoding_name = encoding_name
        self.encoding_args = args
        self.encoding_kwargs = kwargs
        self.encoding = encodings[encoding_name](*args, **kwargs)
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

            self.state_sizes = [2*self.effective_alphabet_size] * 1

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

    def init_from_file(self):
        file_path = os.path.join(config.SAVED_MODELS_DIR, self.name, config.TAG)
        extra_file_path = file_path + '.pkl'
        with open(extra_file_path, 'rb') as f:
            extra = pickle.load(f)

        encoding_name = extra['encoding_name']
        encoding_args = extra['encoding_args']
        encoding_kwargs = extra['encoding_kwargs']
        self._setup_encoding(encoding_name, *encoding_args, **encoding_kwargs)

        self.state_sizes = extra['state_sizes']

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

        logging.info('Loaded from file %s' % file_path)

    def save_to_file(self, name=None):
        if name is None:
            name = self.name
        save_dir = os.path.join(config.SAVED_MODELS_DIR, name)
        tools.rmdir_if_exists(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, config.TAG)
        self.saver.save(self.sess, file_path)

        extra = {
            'encoding_name': self.encoding_name,
            'encoding_args': self.encoding_args,
            'encoding_kwargs': self.encoding_kwargs,
            'state_sizes': self.state_sizes
        }
        extra_file_path = file_path + '.pkl'
        with open(extra_file_path, 'wb') as f:
            pickle.dump(extra, f)

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

    def train(self, inputs, autosave=None):
        """
        Args:
            inputs: A python iterable of things that encode to python iterables (if long) or lists (if short) of numbers
        """
        inputs = map(self.encoding.encode_input_for_training, inputs)
        batches = self.encoding.make_batches_for_training(inputs)

        save_dir = os.path.join(config.SAVED_SUMMARIES_DIR, self.name, config.TAG)
        tools.rmdir_if_exists(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        summaries_dir = save_dir
        logging.info('Using summaries dir %s' % summaries_dir)
        summary_writer = tf.summary.FileWriter(summaries_dir, self.graph)

        curr_states = None
        try:
            for i, batch in enumerate(batches):
                if i % 10 != 0:
                    _ = self._run_batch([self.optimize], batch, curr_states)
                else:
                    _, summary, loss_max, loss_mean, loss_min = self._run_batch(
                            [self.optimize, self.summary, self.loss_max, self.loss_mean, self.loss_min], batch, curr_states)
                    summary_writer.add_summary(summary, i)
                    logging.info('Step %s: loss_max: %s loss_mean: %s loss_min: %s' % (i, loss_max, loss_mean, loss_min))
                    #losses, probabilities = self.sess.run([self.losses, self.probabilities], feed_dict={self.inputs: batch})
                    #logging.info('Step %s: losses: %s probs: %s' % (i, losses, probabilities))
                # curr_states = new_states
                if autosave is not None and i % autosave == 0 and i != 0:
                    name = self.name + '_' + str(i)
                    self.save_to_file(name)
        except KeyboardInterrupt:
            logging.info('Cancelling training...')
        logging.info('Saved summaries to %s' % summaries_dir)
        if autosave is not None and i % autosave != 0:
            # Save the last one only if it hasn't already been saved
            name = self.name + '_' + str(i)
            self.save_to_file(name)

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
    def __init__(self, name, encoding_name, *args, **kwargs):
        super(LSTMModelEncoding, self).__init__(name)
        self.init_from_encoding(encoding_name, *args, **kwargs)

class LSTMModel(LSTMModelEncoding):
    def __init__(self, name, alphabet_size, **kwargs):
        super(LSTMModel, self).__init__(name, 'basic', alphabet_size, **kwargs)

class LSTMModelString(LSTMModelEncoding):
    def __init__(self, name, **kwargs):
        super(LSTMModelString, self).__init__(name, 'string', **kwargs)

class LSTMModelTextFile(LSTMModelEncoding):
    def __init__(self, name, **kwargs):
        super(LSTMModelTextFile, self).__init__(name, 'text-file', **kwargs)

class LSTMModelFromFile(LSTMModelBase):
    def __init__(self, name):
        super(LSTMModelFromFile, self).__init__(name)
        self.init_from_file()
