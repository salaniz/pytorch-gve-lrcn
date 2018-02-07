import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from utils.misc import to_var

class LRCNTrainer:
    def __init__(self, args, model, dataset, data_loader, checkpoint=None):
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader
        self.cuda = args.cuda

        if self.cuda:
            model.cuda()

        # TODO: Implement checkpoints
        if checkpoint is None:
            self.criterion = nn.CrossEntropyLoss()
            self.params = filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)
            self.total_steps = len(data_loader)
            self.num_epochs = args.num_epochs
            self.log_step = args.log_step

    def train_epoch(self, epoch_num):
        for i, (images, word_inputs, word_targets, lengths, ids) in enumerate(self.data_loader):
            # Prepare mini-batch dataset
            images = to_var(images, self.cuda)
            word_inputs = to_var(word_inputs, self.cuda)
            word_targets = to_var(word_targets, self.cuda)
            word_targets = pack_padded_sequence(word_targets, lengths, batch_first=True)[0]

            loss = self.train_step(images, word_inputs, word_targets, lengths)

            # TODO: Add proper logging
            # Print log info
            if i % self.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch_num, self.num_epochs, i, self.total_steps,
                        loss.data[0], np.exp(loss.data[0])))


    def train_step(self, images, word_inputs, word_targets, lengths):
        # Forward, Backward and Optimize
        self.model.zero_grad()
        outputs = self.model(images, word_inputs, lengths)
        loss = self.criterion(outputs, word_targets)
        loss.backward()
        self.optimizer.step()

        return loss
