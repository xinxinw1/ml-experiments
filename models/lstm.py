"""
Interesting examples:

model = lstm.LSTMModel(2)
model.train([(lambda n: [1]*n + [0]*n)(np.random.randint(0, 10)) for i in range(20000)])
"""

import tensorflow as tf
import numpy as np
from datetime import datetime
import uuid
import os

import my_logging as logging

import config

def group(lst, n):
    return (lst[i:i+n] for i in range(0, len(lst), n))

def pad_to_len(lst, max_len, pad_item):
    return lst + [pad_item] * max(0, max_len-len(lst))

def pad_to_same_len(lst_of_lsts, pad_item):
    max_len = max(map(len, lst_of_lsts))
    return list(map(lambda lst: pad_to_len(lst, max_len, pad_item), lst_of_lsts))

class LSTMModel(object):
    def __init__(self, alphabet_size):
        self.uuid = uuid.uuid4()
        logging.info('Running __init__ %s' % self.uuid)

        self.alphabet_size = alphabet_size
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.int32, shape=(None, None), name='inputs')
            # inputs is list of batches where each batch is a list of integers in [0, alphabet_size)
            # inputs is tensor with dim batch_size x timesteps

            self.inputs_padded = tf.pad(self.inputs, [[0, 0], [1, 0]], constant_values=self.alphabet_size, name='inputs_padded')
            self.labels_padded = tf.pad(self.inputs, [[0, 0], [0, 1]], constant_values=self.alphabet_size, name='labels_padded')
            # inputs_padded and labels_padded are tensors with dim batch_size x (timesteps+1)

            self.inputs_one_hot = tf.one_hot(self.inputs_padded, self.alphabet_size+1, name='inputs_one_hot')
            # inputs_one_hot is a list of batches where each batch is a list of one hot encoded lists
            # inputs_one_hot is a tensor with dim batch_size x (timesteps+1) x (alphabet_size+1)

            state_sizes = [self.alphabet_size+1] * 1
            lstm_cells = list(map(tf.contrib.rnn.BasicLSTMCell, state_sizes))
            lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
            rnn_output, state = tf.nn.dynamic_rnn(lstm, self.inputs_one_hot, dtype=tf.float32)
            # rnn_outputs is a tensor with dim batch_size x (timesteps+1) x (alphabet_size+1)
            logits = tf.contrib.layers.fully_connected(rnn_output, self.alphabet_size+1, activation_fn=None)
            self.logits = tf.identity(logits, name='logits')

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels_padded,
                    logits=self.logits)
            self.losses = tf.identity(losses, name='losses')
            # losses is a tensor with dim batch_size x (timestamps+1)

            self.loss_mean = tf.reduce_mean(self.losses, name='loss_mean')
            tf.summary.scalar('loss_mean', self.loss_mean)
            self.loss_max = tf.reduce_max(self.losses, name='loss_max')
            tf.summary.scalar('loss_max', self.loss_max)
            self.loss_sum = tf.reduce_sum(self.losses, name='loss_sum')
            tf.summary.scalar('loss_sum', self.loss_sum)
            # loss_* is a tensor with a single float

            self.adam_optimize = tf.train.AdamOptimizer().minimize(self.loss_mean, name='adam_optimize')
            self.adadelta_optimize = tf.train.AdadeltaOptimizer(0.03).minimize(self.loss_mean, name='adadelta_optimize')

            self.probabilities = tf.nn.softmax(self.logits, name='probabilities')

            self.saver = tf.train.Saver()

            self.summary = tf.summary.merge_all()

            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        logging.info('Running __del__ %s' % self.uuid)
        self.sess.close()
        logging.info('Closed session %s' % self.uuid)

    def save_to_file(self, name=None):
        if name is None:
            name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_path = os.path.join(config.SAVED_MODELS_DIR, name)
        self.saver.save(self.sess, file_path)
        logging.info('Saved to file %s' % file_path)

    def load_from_file(self, name):
        file_path = os.path.join(config.SAVED_MODELS_DIR, name)
        self.saver.restore(self.sess, file_path)
        logging.info('Loaded from file %s' % file_path)

    def train(self, inputs, optimize=None):
        if optimize is None:
            optimize = self.adam_optimize
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        summaries_dir = os.path.join(config.SAVED_SUMMARIES_DIR, date)
        logging.info('Using summaries dir %s' % summaries_dir)
        summary_writer = tf.summary.FileWriter(summaries_dir)
        batches = group(inputs, 10)
        for i, batch in enumerate(batches):
            batch = pad_to_same_len(batch, self.alphabet_size)
            summary, _ = self.sess.run([self.summary, optimize], feed_dict={self.inputs: batch})
            summary_writer.add_summary(summary, i)
            if i % 100 == 0:
                loss_max, loss_mean = self.sess.run([self.loss_max, self.loss_mean], feed_dict={self.inputs: batch})
                logging.info('Step %s: loss_max: %s loss_mean: %s' % (i, loss_max, loss_mean))
                #losses, loss_mean, probabilities = self.sess.run([self.losses, self.loss_mean, self.probabilities], feed_dict={self.inputs: batch})
                #logging.info('Step %s: loss: %s losses: %s probs: %s' % (i, loss_mean, losses, probabilities))
        logging.info('Saved summaries to %s' % summaries_dir)

    def sample(self, starting=[], max_num=1000):
        curr = list(starting)
        for i in range(max_num):
            probs_batch = self.sess.run(self.probabilities, feed_dict={self.inputs: [curr]})
            probs = probs_batch[0][-1]
            next_int = np.random.choice(self.alphabet_size+1, 1, p=probs).item()
            logging.info('Step %s: curr: %s next_int: %s probs: %s' % (i, curr, next_int, probs))
            if next_int == self.alphabet_size:
                break
            curr.append(next_int)
        return curr

