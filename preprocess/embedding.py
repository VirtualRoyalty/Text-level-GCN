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

    def __init__(weights_path):
        self.weights_path = weights_path
        self.name = 'GloVe'
        self.embeddings_index = {}
        self.load()

    def load():
        f = open(self.weights_path, encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

    def get(token):
        return self.embeddings_index.get(token)

    def name():
        return self.name


class FastText(Embedding):

    def __init__(weights_path):
        self.weights_path = weights_path
        self.name = 'FastText'
        self.load()

    def load():
        self.fbkv = fasttext.load_model(self.weights_path)

    def get(token):
        return self.fbkv.get_word_vector(token)

    def name():
        return self.name
