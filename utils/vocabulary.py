# Adapted from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
import pickle

class Vocabulary():
    """Simple vocabulary wrapper."""
    def __init__(self, unknown_token='<UNK>', start_token='<SOS>',
                       end_token='<EOS>', padding_token='<PAD>',
                       add_special_tokens=True):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token

        if add_special_tokens:
            # add padding first so that it corresponds to 0
            self.add_word(padding_token)
            self.add_word(start_token)
            self.add_word(end_token)
            self.add_word(unknown_token)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word_from_idx(self, idx):
        if not idx in self.idx2word:
            return self.unknown_token
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unknown_token]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        assert isinstance(vocab, cls)
        return vocab

    @classmethod
    def save(cls, vocab, path):
        assert isinstance(vocab, cls)
        with open(path, 'wb') as f:
            pickle.dump(vocab, f)
