import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class LRCNTrainer:

    REQ_EVAL = True

    def __init__(self, args, model, dataset, data_loader, logger, device, checkpoint=None):
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader
        self.train = args.train
        self.logger = logger
        self.device = device

        model.to(self.device)

        # TODO: Implement checkpoint recovery
        if checkpoint is None:
            self.criterion = nn.CrossEntropyLoss()
            self.params = filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)
            self.total_steps = len(data_loader)
            self.num_epochs = args.num_epochs
            self.log_step = args.log_step
            self.curr_epoch = 0

        vocab = self.dataset.vocab
        start_word = torch.tensor([vocab(vocab.start_token)],
                device=self.device, dtype=torch.long)
        self.start_word = start_word.unsqueeze(0)
        end_word = torch.tensor([vocab(vocab.end_token)], device=device,
                dtype=torch.long)
        self.end_word = end_word.unsqueeze(0)


    def train_epoch(self):
        # Result is list of losses during training
        # and generated captions during evaluation
        result = []

        for i, (image_input, word_inputs, word_targets, lengths, ids, *excess) in enumerate(self.data_loader):
            # Prepare mini-batch dataset
            image_input = image_input.to(self.device)

            if self.train:
                word_inputs = word_inputs.to(self.device)
                word_targets = word_targets.to(self.device)
                word_targets = pack_padded_sequence(word_targets, lengths, batch_first=True)[0]

                loss = self.train_step(image_input, word_inputs, word_targets,
                        lengths, *excess)
                result.append(loss.data.item())

                step = self.curr_epoch * self.total_steps + i + 1
                self.logger.scalar_summary('batch_loss', loss.data.item(), step)

            else:
                generated_captions = self.eval_step(image_input, ids, *excess)
                result.extend(generated_captions)

            # TODO: Add proper logging
            # Print log info
            if i % self.log_step == 0:
                print("Epoch [{}/{}], Step [{}/{}]".format(self.curr_epoch,
                    self.num_epochs, i, self.total_steps), end='')
                if self.train:
                    print(", Loss: {:.4f}, Perplexity: {:5.4f}".format(loss.data.item(),
                        np.exp(loss.data.item())), end='')
                print()


        self.curr_epoch += 1

        if self.train:
            self.logger.scalar_summary('epoch_loss', np.mean(result), self.curr_epoch)

        return result


    def train_step(self, image_input, word_inputs, word_targets, lengths,
            *args):
        # Forward, Backward and Optimize
        self.model.zero_grad()
        outputs = self.model(image_input, word_inputs, lengths)
        loss = self.criterion(outputs, word_targets)
        loss.backward()
        #nn.utils.clip_grad_norm(self.params, 10)
        self.optimizer.step()

        return loss


    def eval_step(self, image_input, ids, *args):
        # TODO: max_sampling_length
        vocab = self.dataset.vocab
        generated_captions = []
        outputs = self.model.generate_sentence(image_input, self.start_word, self.end_word, *args)
        for out_idx in range(len(outputs)):
            sentence = []
            for w in outputs[out_idx]:
                word = vocab.get_word_from_idx(w.data.item())
                if word != vocab.end_token:
                    sentence.append(word)
                else:
                    break
            generated_captions.append({"image_id": ids[out_idx], "caption": ' '.join(sentence)})

        return generated_captions

