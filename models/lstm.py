import tensorflow as tf
import numpy as np
import uuid

class LSTMModel(object):
    def __init__(self):
        self.uuid = uuid.uuid4()
        print('Running __init__ %s' % self.uuid)
        self.sess = None

    def __del__(self):
        print('Running __del__ %s' % self.uuid)
        if self.sess is not None:
            self.sess.close()
            print('Closed session %s' % self.uuid)

    def init_from_scratch(self, alphabet_size):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.int32, shape=(None, None), name='inputs')
            # inputs is list of batches where each batch is a list of integers in [0, alphabet_size)
            # inputs is tensor with dim batch_size x timesteps

            self.inputs_padded = tf.pad(self.inputs, [[0, 0], [1, 0]], constant_values=alphabet_size, name='inputs_padded')
            self.labels_padded = tf.pad(self.inputs, [[0, 0], [0, 1]], constant_values=alphabet_size, name='labels_padded')
            # inputs_padded and labels_padded are tensors with dim batch_size x (timesteps+1)

            self.inputs_one_hot = tf.one_hot(self.inputs_padded, alphabet_size+1, name='inputs_one_hot')
            # inputs_one_hot is a list of batches where each batch is a list of one hot encoded lists
            # inputs_one_hot is a tensor with dim batch_size x (timesteps+1) x (alphabet_size+1)

            state_sizes = [alphabet_size+1] * 5
            lstm_cells = list(map(tf.contrib.rnn.BasicLSTMCell, state_sizes))
            lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
            logits, state = tf.nn.dynamic_rnn(lstm, self.inputs_one_hot, dtype=tf.float32)
            # rnn_outputs is a tensor with dim batch_size x (timesteps+1) x (alphabet_size+1)
            self.logits = tf.identity(logits, name='logits')

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels_padded,
                    logits=self.logits)
            self.losses = tf.identity(losses, name='losses')
            # losses is a tensor with dim batch_size x (timestamps+1)

            self.loss_mean = tf.reduce_mean(self.losses, name='loss_mean')
            self.loss_max = tf.reduce_max(self.losses, name='loss_max')
            self.loss_sum = tf.reduce_sum(self.losses, name='loss_sum')
            # loss_* is a tensor with a single float

            self.optimize = tf.train.AdamOptimizer().minimize(self.loss_mean, name='optimize')

            self.probabilities = tf.nn.softmax(self.logits, name='probabilities')

            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())

    def train(self, inputs):
        for i, input in enumerate(inputs):
            batch = [input]
            self.sess.run(self.optimize, feed_dict={self.inputs: batch})
            loss_mean = self.sess.run(self.loss_mean, feed_dict={self.inputs: batch})
            print('Step %s: loss: %s' % (i, loss_mean))

    def export_to_str(self):
        return self.graph.as_graph_def().SerializeToString()

    def init_from_str(self, graph_def_str):
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_def_str)
            tf.import_graph_def(graph_def, name='')

            self.inputs = self.graph.get_tensor_by_name('inputs:0')
            self.logits = self.graph.get_tensor_by_name('logits:0')
            self.losses = self.graph.get_tensor_by_name('losses:0')
            self.loss_mean = self.graph.get_tensor_by_name('loss_mean:0')
            self.loss_max = self.graph.get_tensor_by_name('loss_max:0')
            self.loss_sum = self.graph.get_tensor_by_name('loss_sum:0')
            self.optimize = self.graph.get_operation_by_name('optimize')
            self.probabilities = self.graph.get_tensor_by_name('probabilities:0')

            self.sess = tf.Session(graph=self.graph)

