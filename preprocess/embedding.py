import fasttext

from abc import ABC, abstractmethod, abstractproperty


class Embedding(ABC):
    not_impl_mes = "Each Model must re-implement this method."
    @abstractmethod
    def get(self):
        raise NotImplementedError(not_impl_mes)

    @abstractmethod
    def load(self):
        raise NotImplementedError(not_impl_mes)

    @abstractproperty
    def name(self):
        raise NotImplementedError(not_impl_mes)


class Glove(Embedding):

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.embeddings_index = {}
        self.load()

    def load(self):
        print('loading...')
        f = open(self.weights_path, encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        print('loaded')

    def get(self, token):
        return self.embeddings_index.get(token)

    @property
    def name(self):
        return self.name


class FastText(Embedding):

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.load()

    def load(self):
        print('loading...')
        self.fbkv = fasttext.load_model(self.weights_path)
        print('loaded')

    def get(self, token):
        return self.fbkv.get_word_vector(token)

    @property
    def name(self):
        return self.name
