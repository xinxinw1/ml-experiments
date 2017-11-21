import tensorflow as tf
import numpy as np
from datetime import datetime
import uuid
import os
import json
import pickle
from abc import ABC, abstractmethod

import my_logging as logging

import config

import tools

class Encoding(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def encode_single(self, inpt):
        pass

    @abstractmethod
    def decode_single(self, outpt):
        pass

    @abstractmethod
    def decode_num(self, num):
        pass

    @abstractmethod
    def empty(self):
        pass

class BasicEncoding(Encoding):
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size

    def encode_single(self, inpt):
        return inpt

    def decode_single(self, outpt):
        return outpt

    def decode_num(self, num):
        return num

    def empty(self):
        return []

class StringEncoding(Encoding):
    def __init__(self):
        self.alphabet_size = 256

    def encode_single(self, inpt):
        return tools.string_to_bytes(inpt)

    def decode_single(self, outpt):
        return tools.bytes_to_string(outpt)

    def decode_num(self, num):
        return chr(num)

    def empty(self):
        return ''

encodings = {
    'basic': BasicEncoding,
    'string': StringEncoding
}

class LSTMModelBase(object):
    def __init__(self, name):
        self.uuid = uuid.uuid4()
        logging.info('Running __init__ %s' % self.uuid)
        self.name = name
        self.sess = None

    def __del__(self):
        logging.info('Running __del__ %s' % self.uuid)
        self.close_sess_if_open()

    def close_sess_if_open(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None
            logging.info('Closed session %s' % self.uuid)

    def setup_encoding(self, encoding_name, *args):
        self.encoding_name = encoding_name
        self.encoding = encodings[encoding_name](*args)
        self.alphabet_size = self.encoding.alphabet_size

    def init_from_encoding(self, encoding_name, *args):
        self.setup_encoding(encoding_name, *args)
        self.init_after_encoding()

    def init_after_encoding(self):
        self.close_sess_if_open()

        self.effective_alphabet_size = self.alphabet_size + 2
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

            def make_init_state(i, state_size):
                return tf.placeholder(tf.float32, [2, None, state_size], name='init_state_' + str(i))

            def make_lstm_state_tuple(init_state):
                c, h = tf.unstack(init_state)
                return tf.contrib.rnn.LSTMStateTuple(c, h)

            self.state_sizes = [2*self.effective_alphabet_size] * 1
            self.init_states = tuple(make_init_state(i, state_size) for i, state_size in enumerate(self.state_sizes))
            rnn_init_state = tuple(make_lstm_state_tuple(init_state) for init_state in self.init_states)

            lstm_cells = list(map(tf.contrib.rnn.BasicLSTMCell, self.state_sizes))
            lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
            rnn_output, rnn_states = tf.nn.dynamic_rnn(lstm, self.inputs_one_hot, initial_state=rnn_init_state, dtype=tf.float32)
            # rnn_outputs is a tensor with dim batch_size x (timesteps+1) x (alphabet_size+1)
            # state is a list of tensors with dim batch_size x state_size

            def make_output_state(i, state_tuple):
                c = state_tuple.c
                h = state_tuple.h
                return tf.identity(tf.stack([c, h]), name='output_state_' + str(i))

            self.output_states = tuple(make_output_state(i, state) for i, state in enumerate(rnn_states))

            logits = tf.contrib.layers.fully_connected(rnn_output, self.effective_alphabet_size, activation_fn=None)
            self.logits = tf.identity(logits, name='logits')

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels,
                    logits=self.logits)
            self.losses = tf.identity(losses, name='losses')
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

            self.probabilities = tf.nn.softmax(self.logits, name='probabilities')

            self.saver = tf.train.Saver()

            self.summary = tf.identity(tf.summary.merge_all(), name='summary')

            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())


    def init_from_file(self):
        self.close_sess_if_open()

        file_path = os.path.join(config.SAVED_MODELS_DIR, self.name, config.TAG)
        extra_file_path = file_path + '.pkl'
        with open(extra_file_path, 'rb') as f:
            extra = pickle.load(f)

        self.encoding_name = extra['encoding_name']
        self.encoding = extra['encoding']
        self.alphabet_size = extra['alphabet_size']
        self.effective_alphabet_size = extra['effective_alphabet_size']
        self.state_sizes = extra['state_sizes']

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

            self.init_states = tuple(self.graph.get_tensor_by_name('init_state_' + str(i) + ':0')
                    for i in range(len(self.state_sizes)))
            self.output_states = tuple(self.graph.get_tensor_by_name('output_state_' + str(i) + ':0')
                    for i in range(len(self.state_sizes)))

            self.summary = self.graph.get_tensor_by_name('summary:0')

        logging.info('Loaded from file %s' % file_path)

    def save_to_file(self):
        save_dir = os.path.join(config.SAVED_MODELS_DIR, self.name)
        tools.rmdir_if_exists(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, config.TAG)
        self.saver.save(self.sess, file_path)

        extra = {
            'encoding_name': self.encoding_name,
            'encoding': self.encoding,
            'alphabet_size': self.alphabet_size,
            'effective_alphabet_size': self.effective_alphabet_size,
            'state_sizes': self.state_sizes
        }
        extra_file_path = file_path + '.pkl'
        with open(extra_file_path, 'wb') as f:
            pickle.dump(extra, f)

        logging.info('Saved model to file %s' % file_path)

    def run_batch_with_state(self, tensors, batch, curr_states=None):
        """
        batch: tuple (inputs_batch, labels_batch)
        Returns:
            A list of outputs [new_states] + tensor_outputs
        """
        return self.run_batch(tensors + [self.output_states], batch, curr_states)

    def run_batch(self, tensors, batch, curr_states=None):
        """
        batch: tuple (inputs_batch, labels_batch)
        Returns:
            A list of outputs tensor_outputs
        """
        inputs, labels = batch
        # print(inputs, labels)
        if curr_states is None:
            batch_size = inputs.shape[0]
            def make_current_state(state_size):
                return np.zeros((2, batch_size, state_size))

            curr_states = tuple(map(make_current_state, self.state_sizes))

        feed_dict = {self.inputs: inputs, self.labels: labels, self.init_states: curr_states}
        return self.sess.run(tensors, feed_dict=feed_dict)

    def run_single(self, tensors, lst):
        batches = tools.make_batches([lst], batch_size=1, max_single_len=None,
                token_item=self.alphabet_size, pad_item=self.alphabet_size)
        return self.run_batch(tensors, next(batches))

    def encode_single(self, inpt):
        return self.encoding.encode_single(inpt)

    def decode_single(self, outpt):
        return self.encoding.decode_single(outpt)

    def decode_num(self, num):
        return self.encoding.decode_num(num)

    def empty(self):
        return self.encoding.empty()

    def encode_iter(self, inputs):
        return map(self.encode_single, inputs)

    def decode_if_ok(self, num):
        return self.decode_num(num) if num < self.alphabet_size else num

    def decode_single_to_list(self, outpt):
        return list(map(self.decode_if_ok, outpt))

    def train(self, inputs):
        """
        Args:
            inputs: A python iterable of python lists
        """
        save_dir = os.path.join(config.SAVED_SUMMARIES_DIR, self.name, config.TAG)
        tools.rmdir_if_exists(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        summaries_dir = save_dir
        logging.info('Using summaries dir %s' % summaries_dir)
        summary_writer = tf.summary.FileWriter(summaries_dir)

        batch_size = 10
        max_single_len = 3

        inputs = self.encode_iter(inputs)

        batches = tools.make_batches(inputs, batch_size, max_single_len,
                token_item=self.alphabet_size, pad_item=self.alphabet_size+1)

        curr_states = None
        for i, batch in enumerate(batches):
            summary, _, new_states = self.run_batch_with_state([self.summary, self.optimize], batch, curr_states)
            summary_writer.add_summary(summary, i)
            if i % 100 == 0:
                loss_max, loss_mean, loss_min = self.run_batch([self.loss_max, self.loss_mean, self.loss_min], batch, curr_states)
                logging.info('Step %s: loss_max: %s loss_mean: %s loss_min: %s' % (i, loss_max, loss_mean, loss_min))
                #losses, probabilities = self.sess.run([self.losses, self.probabilities], feed_dict={self.inputs: batch})
                #logging.info('Step %s: losses: %s probs: %s' % (i, losses, probabilities))
            curr_states = new_states
        logging.info('Saved summaries to %s' % summaries_dir)

    def make_probs_dec(self, probs):
        probs_dec = [(self.decode_if_ok(i), prob) for i, prob in enumerate(probs)]
        probs_dec.sort(key=lambda s: s[1], reverse=True)
        probs_dec = probs_dec[:7]
        return probs_dec

    def sample(self, starting=None, max_num=1000):
        if starting is None:
            starting = self.empty()
        starting = self.encode_single(starting)
        # starting: A python iterable of numbers
        curr = list(starting)
        curr_output = list(curr)
        for i in range(max_num):
            probs_batch = self.run_single(self.probabilities, curr)
            probs = probs_batch[0][-1]
            next_int = np.random.choice(self.effective_alphabet_size, 1, p=probs).item()
            curr_dec = self.decode_single_to_list(curr)
            next_dec = self.decode_if_ok(next_int)
            probs_dec = self.make_probs_dec(probs)
            logging.info('Step %s: curr: %s next: %s probs: %s' % (i, curr_dec, repr(next_dec), probs_dec))
            if next_int == self.alphabet_size:
                break
            if next_int < self.alphabet_size:
                curr_output.append(next_int)
            curr.append(next_int)
        return self.decode_single(curr)

    def analyze(self, inpt):
        """
        Args:
            inpt: A python list
        """
        lst = self.encode_single(inpt)
        losses_batch, probs_batch = self.run_single([self.losses, self.probabilities], lst)
        losses = losses_batch[0].tolist()
        probs_list = map(self.make_probs_dec, probs_batch[0])
        lst_dec = self.decode_single_to_list(lst)
        for item, loss, probs in zip(lst_dec + [self.alphabet_size], losses, probs_list):
            print('%-5s %-11.8f %s' % (repr(item), loss, probs))

class LSTMModel(LSTMModelBase):
    def __init__(self, name, alphabet_size):
        super(LSTMModel, self).__init__(name)
        self.init_from_encoding('basic', alphabet_size)

class LSTMModelString(LSTMModelBase):
    def __init__(self, name):
        super(LSTMModelString, self).__init__(name)
        self.init_from_encoding('string')

class LSTMModelFile(LSTMModelBase):
    def __init__(self, name):
        super(LSTMModelFile, self).__init__(name)
        self.init_from_file()
