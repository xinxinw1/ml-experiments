from abc import ABC, abstractmethod
import tools
from functools import reduce

"""
The final output that goes to the lstm batch creator should be an
iterable of iterables of integers. For example:
    [[5, 2, 3, 0], [2, 3, 5, 2, 4, 5]]
"""

class Alphabet(object):
    def __init__(self, it):
        """
        Args:
            it: An iterable of hashable items
        """
        self.alphabet_list = list(set(it))
        # Use a list and sort it so that the order
        # is consistent every time
        self.alphabet_list.sort()
        self.ind_to_num = {}
        self.num_to_ind = []
        for num, ind in enumerate(self.alphabet_list):
            self.ind_to_num[ind] = num
            self.num_to_ind.append(ind)
        self.alphabet_size = len(self.alphabet_list)

    def encode_ind(self, ind):
        return self.ind_to_num[ind]

    def decode_num(self, num):
        return self.num_to_ind[num]

def get_encoder(encoder_name):
    return encoders[encoder_name]

def encoder_name_to_object(encoder_name):
    print('encoder name', encoder_name)
    if isinstance(encoder_name, (list, tuple)):
        encoder_name, *encoder_args = encoder_name
    else:
        encoder_args = []
    encoder_class = get_encoder(encoder_name)
    encoder_object = encoder_class(*encoder_args)
    return encoder_object

def encoder_names_to_objects(encoder_names):
    return list(map(encoder_name_to_object, encoder_names))

def encode_multiple(inputs, encoder_objects):
    return reduce(lambda inputs, encoder_object: encoder_object.encode_multiple(inputs), encoder_objects, inputs)

def encode_single(inpt, encoder_objects):
    return reduce(lambda inpt, encoder_object: encoder_object.encode_single(inpt), encoder_objects, inpt)

def encode_individual(ind, encoder_objects):
    return reduce(lambda ind, encoder_object: encoder_object.encode_individual(ind), encoder_objects, ind)

def get_alphabet_size(encoder_objects):
    return reduce(lambda prev_alphabet_size, encoder_object: encoder_object.get_alphabet_size(prev_alphabet_size),
            encoder_objects, None)

def get_empty_single(encoder_objects):
    return encoder_objects[0].get_empty_single()

def make_alphabet_with_encoding(inpt, encoding=[]):
    encoder_objects = encoder_names_to_objects(encoding)
    encoded = encode_single(inpt, encoder_objects)
    return Alphabet(encoded)

class Encoding(object):
    def encode_individual(self, ind):
        """
        Args:
            ind: An input individual
        Returns:
            out: An output individual
        """
        raise NotImplementedError('Encoding individuals is unsupported by %s' % self.__class__.__name__)

    def encode_single(self, inpt):
        """
        Args:
            inpt: An input object, usually an iterable of individuals
        Returns:
            outpt: An output object, usually an iterable of individuals
        """
        return map(self.encode_individual, inpt)

    def encode_multiple(self, inputs):
        """
        Args:
            inputs: An iterable of input objects
        Returns:
            outputs: An iterable of output objects
        """
        return map(self.encode_single, inputs)

    def get_empty_single(self):
        """
        Returns:
            inpt: An empty encode_single input object
        """
        raise NotImplementedError('Empty single objects are unsupported by %s' % self.__class__.__name__)

    def get_alphabet_size(self, prev_alphabet_size):
        return None

class Numbers(Encoding):
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size

    def encode_individual(self, ind):
        if not 0 <= ind < self.alphabet_size:
            raise ValueError('Invalid ind %s' % ind)
        return ind

    def get_empty_single(self):
        return []

    def get_alphabet_size(self, prev_alphabet_size):
        return self.alphabet_size

class StringToChars(Encoding):
    def encode_individual(self, ind):
        return ind

    def get_empty_single(self):
        return ''

class CharsToString(Encoding):
    def encode_individual(self, ind):
        return ind

    def encode_single(self, inpt):
        return ''.join(inpt)

class StringToBytes(Encoding):
    def encode_single(self, inpt):
        return tools.string_to_bytes(inpt)

    def get_alphabet_size(self, prev_alphabet_size):
        return 256

    def get_empty_single(self):
        return ''

class BytesToString(Encoding):
    def encode_individual(self, ind):
        return tools.bytes_to_string([ind])

    def encode_single(self, inpt):
        return tools.bytes_to_string(inpt)

class PathToChars(Encoding):
    def encode_single(self, inpt):
        return tools.file_to_chars(inpt)

class PathToBytes(Encoding):
    def encode_single(self, inpt):
        return tools.file_to_bytes(inpt)

    def get_alphabet_size(self, prev_alphabet_size):
        return 256

class PathToLines(Encoding):
    def encode_multiple(self, inputs):
        return tools.flatmap(tools.file_to_lines, inputs)

class AlphabetToNums(Encoding):
    def __init__(self, alphabet):
        """
        Args:
            alphabet: An Alphabet object.
        """
        self.alphabet = alphabet

    def encode_individual(self, ind):
        return self.alphabet.encode_ind(ind)

    def get_empty_single(self):
        return []

    def get_alphabet_size(self, prev_alphabet_size):
        return self.alphabet.alphabet_size

class NumsToAlphabet(Encoding):
    def __init__(self, alphabet):
        """
        Args:
            alphabet: An Alphabet object.
        """
        self.alphabet = alphabet

    def encode_individual(self, ind):
        """
        Args:
            ind: An integer in [0, self.alphabet.alphabet_size)
        """
        return self.alphabet.decode_num(ind)

encoders = {
    'numbers': Numbers,
    'string-to-chars': StringToChars,
    'chars-to-string': CharsToString,
    'string-to-bytes': StringToBytes,
    'bytes-to-string': BytesToString,
    'path-to-chars': PathToChars,
    'path-to-bytes': PathToBytes,
    'path-to-lines': PathToLines,
    'alphabet-to-nums': AlphabetToNums,
    'nums-to-alphabet': NumsToAlphabet
}

