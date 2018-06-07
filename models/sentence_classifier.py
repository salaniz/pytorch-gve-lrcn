import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from .pretrained_models import PretrainedModel

class SentenceClassifier(nn.Module):
    def __init__(self, word_embed_size, hidden_size, vocab_size, num_classes,
            dropout_prob=0.5):
        super(SentenceClassifier, self).__init__()

        self.word_embed = nn.Embedding(vocab_size, word_embed_size, padding_idx=0)

        lstm1_input_size = word_embed_size

        self.lstm = nn.LSTM(lstm1_input_size, hidden_size, batch_first=True)
                #bidirectional=True)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.init_weights()

        self.input_size = vocab_size
        self.output_size = num_classes
        self.dropout_prob = dropout_prob

    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def state_dict(self, *args, full_dict=False, **kwargs):
        return super().state_dict(*args, **kwargs)

    def forward(self, captions, lengths):
        embeddings = self.word_embed(captions)
        embeddings = F.dropout(embeddings, p=self.dropout_prob, training=self.training)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)

        # Extract the outputs for the last timestep of each example
        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), hiddens.size(2))
        idx = idx.unsqueeze(1)
        idx = idx.to(hiddens.device)

        # Shape: (batch_size, hidden_size)
        last_hiddens = hiddens.gather(1, idx).squeeze(1)

        last_hiddens = F.dropout(last_hiddens, p=self.dropout_prob, training=self.training)
        outputs = self.linear(last_hiddens)
        return outputs
