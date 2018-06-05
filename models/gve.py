import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from functools import partial

from .lrcn import LRCN

class GVE(LRCN):
    def __init__(self, input, word_embed_size, hidden_size,
                 vocab_size, sentence_classifier, num_classes, layers_to_truncate=1, dropout_prob=0.5):
        super().__init__(input, word_embed_size, hidden_size, vocab_size, layers_to_truncate, dropout_prob)

        self.sentence_classifier = sentence_classifier
        self.num_classes = num_classes
        lstm2_input_size = 2*hidden_size + num_classes
        self.lstm2 = nn.LSTM(lstm2_input_size, hidden_size, batch_first=True)

    def convert_onehot(self, labels):
        labels_onehot = torch.zeros(labels.size(0),
                self.num_classes)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        return labels_onehot

    def get_labels_append_func(self, labels, labels_onehot):
        if labels_onehot is None:
            labels_onehot = self.convert_onehot(labels)

        def append_labels(labels_to_append, image_features):
            return torch.cat((image_features,
                labels_to_append.to(image_features.device)), 1)

        return partial(append_labels, labels_onehot)

    def forward(self, image_inputs, captions, lengths, labels,
            labels_onehot=None):

        feat_func = self.get_labels_append_func(labels, labels_onehot)
        return super().forward(image_inputs, captions, lengths, feat_func)


    def generate_sentence(self, image_inputs, start_word, end_word,
            labels, labels_onehot=None, states=(None,None), max_sampling_length=50, sample=False):

        feat_func = self.get_labels_append_func(labels, labels_onehot)
        return super().generate_sentence(image_inputs, start_word, end_word, states, max_sampling_length, sample, feat_func)

