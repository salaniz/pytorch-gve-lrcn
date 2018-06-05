import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from .lrcn_trainer import LRCNTrainer

class GVETrainer(LRCNTrainer):

    REQ_EVAL = True

    def __init__(self, args, model, dataset, data_loader, logger, device, checkpoint=None):
        super().__init__(args, model, dataset, data_loader, logger, device, checkpoint)
        self.rl_lambda = args.loss_lambda

    def train_step(self, image_input, word_inputs, word_targets, lengths,
            labels):
        # Forward, Backward and Optimize
        labels_onehot = self.model.convert_onehot(labels)
        labels_onehot = labels_onehot.to(self.device)
        self.model.zero_grad()
        outputs = self.model(image_input, word_inputs, lengths, labels,
                labels_onehot=labels_onehot)

        # Reinforce loss
        # Sample sentences
        sample_ids, log_ps, lengths = self.model.generate_sentence(image_input, self.start_word,
                self.end_word, labels, labels_onehot=labels_onehot, max_sampling_length=50, sample=True)
        # Order sampled sentences/log_probabilities/labels by sentence length (required by LSTM)
        lengths = lengths.cpu().numpy()
        sort_idx = np.argsort(-lengths)
        lengths = lengths[sort_idx]
        sort_idx = torch.tensor(sort_idx, device=self.device, dtype=torch.long)
        labels = labels[sort_idx]
        labels = labels.to(self.device)
        log_ps = log_ps[sort_idx,:]
        sample_ids = sample_ids[sort_idx,:]

        class_pred = self.model.sentence_classifier(sample_ids, lengths)
        class_pred = F.softmax(class_pred, dim=1)
        rewards = class_pred.gather(1, labels.view(-1,1)).squeeze()
        r_loss = -(log_ps.sum(dim=1) * rewards).sum()

        loss = self.rl_lambda * r_loss/labels.size(0) + self.criterion(outputs, word_targets)
        loss.backward()
        #nn.utils.clip_grad_norm(self.params, 10)
        self.optimizer.step()

        return loss

