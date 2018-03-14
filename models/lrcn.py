import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from .pretrained_models import PretrainedModel

class LRCN(nn.Module):
    def __init__(self, pretrained_model_name, word_embed_size, hidden_size,
                 vocab_size, layers_to_truncate=1, is_factored=True):
        super().__init__()
        self.vision_model = PretrainedModel(pretrained_model_name,
                layers_to_truncate=layers_to_truncate)
        self.word_embed = nn.Embedding(vocab_size, word_embed_size, padding_idx=0)
        self.is_factored = is_factored

        lstm1_input_size = word_embed_size
        lstm2_input_size = hidden_size
        if not self.is_factored:
            # Vision features are input to 1st LSTM
            lstm1_input_size += self.vision_model.output_size
        else:
            # Vision features are input to 2nd LSTM (when is_factored == True)
            lstm2_input_size += self.vision_model.output_size

        self.dropout0 = nn.Dropout()
        self.lstm1 = nn.LSTM(lstm1_input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout()
        self.lstm2 = nn.LSTM(lstm2_input_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

        self.input_size = (self.vision_model.input_size, vocab_size)
        self.output_size = vocab_size


    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, image_inputs, captions, lengths):
        image_features = self.vision_model(image_inputs)
        image_features = image_features.unsqueeze(1)
        embeddings = self.word_embed(captions)
        embeddings = self.dropout0(embeddings)
        image_features = image_features.expand(-1, embeddings.size(1), -1)
        if not self.is_factored:
            # TODO: fix dropout for non-factored models
            embeddings = torch.cat((image_features, embeddings), 2)
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
            hiddens, _ = self.lstm1(packed)
            hiddens, _ = self.lstm2(hiddens)
        else:
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
            hiddens, _ = self.lstm1(packed)
            unpacked_hiddens, new_lengths = pad_packed_sequence(hiddens, batch_first=True)
            unpacked_hiddens = torch.cat((image_features, unpacked_hiddens), 2)
            unpacked_hiddens = self.dropout1(unpacked_hiddens)
            packed_hiddens = pack_padded_sequence(unpacked_hiddens, lengths,
                    batch_first=True)
            hiddens, _ = self.lstm2(packed_hiddens)

        hiddens = self.dropout2(hiddens[0])
        outputs = self.linear(hiddens)
        return outputs

    def state_dict(self):
        state_dict = super().state_dict()
        for key in self.vision_model.state_dict().keys():
            del state_dict['vision_model.{}'.format(key)]
        return state_dict

    def sample(self, image_inputs, start_word, end_word, states=(None,None),
            max_sampling_length=50):
        sampled_ids = []
        image_features = self.vision_model(image_inputs)
        image_features = image_features.unsqueeze(1)
        embedded_word = self.word_embed(start_word)
        embedded_word = embedded_word.expand(image_features.size(0), -1, -1)
        lstm1_states, lstm2_states = states

        end_word = end_word.squeeze().expand(image_features.size(0))
        reached_end = torch.zeros_like(end_word).byte()

        i = 0
        while not reached_end.all() and i < max_sampling_length:
            lstm1_input = embedded_word

            if not self.is_factored:
                lstm1_input = torch.cat((image_features, lstm1_input), 2)

            # LSTM 1
            lstm1_output, lstm1_states = self.lstm1(lstm1_input, lstm1_states)

            if self.is_factored:
                lstm1_output = torch.cat((image_features, lstm1_output), 2)

            # LSTM 2
            lstm2_output, lstm2_states = self.lstm2(lstm1_output, lstm2_states)

            outputs = self.linear(lstm2_output.squeeze(1))
            predicted = outputs.max(1)[1]
            reached_end = reached_end | predicted.eq(end_word)
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze()
