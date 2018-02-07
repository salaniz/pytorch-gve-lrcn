import os
from collections import Counter
from enum import Enum
import pickle
from PIL import Image

import torch
import torch.utils.data as data
import numpy as np
from pycocotools.coco import COCO
#import nltk

from utils.vocabulary import Vocabulary
from utils.tokenizer.ptbtokenizer import PTBTokenizer

# Adapted from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
class CocoDataset(data.Dataset):

    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    image_train_path = 'train2014'
    image_val_path = 'val2014'
    caption_train_path = 'annotations/captions_train2014.json'
    caption_val_path = 'annotations/captions_val2014.json'
    vocab_file_name = 'coco_vocab.pkl'
    tokens_train_file_name = 'coco_tokens_train.pkl'
    tokens_val_file_name = 'coco_tokens_val.pkl'

    # punctuations to be removed from the sentences
    PUNCTUATIONS = ["''", "'", "``", "`", "(", ")", "{", "}",
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    class ID_BASE(Enum):
        CAPTIONS = 0
        IMAGES = 1


    def __init__(self, root, json, vocab, tokenized_captions, transform=None,
            ids_based_on=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """

        if ids_based_on is None:
            ids_based_on = self.ID_BASE.CAPTIONS
        assert isinstance(ids_based_on, self.ID_BASE)
        self.ids_based_on = ids_based_on

        self.root = root
        self.coco = COCO(json)
        if ids_based_on == self.ID_BASE.CAPTIONS:
            self.ids = list(self.coco.anns.keys())
        elif ids_based_on == self.ID_BASE.IMAGES:
            self.ids = list(self.coco.imgs.keys())
        else:
            raise ValueError("Chosen base for COCO IDs is not implemented")
        self.vocab = vocab
        self.tokens = tokenized_captions
        self.transform = transform


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        base_id = self.ids[index]
        #caption = coco.anns[ann_id]['caption']

        if self.ids_based_on == self.ID_BASE.CAPTIONS:
            ann_id = base_id
            img_id = coco.anns[ann_id]['image_id']
        elif self.ids_based_on == self.ID_BASE.IMAGES:
            img_id = base_id
            img_anns = coco.imgToAnns[img_id]
            rand_idx = np.random.randint(len(img_anns))
            ann_id = img_anns[rand_idx]['id']

        tokens = self.tokens[ann_id]
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        """
        # Convert caption (string) to word ids.
        tokens = CocoDataset.tokenize(caption)
        """
        caption = []
        caption.append(vocab(vocab.start_token))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab(vocab.end_token))
        target = torch.Tensor(caption)
        return image, target, base_id


    def __len__(self):
        return len(self.ids)


    @staticmethod
    def collate_fn(data):
        """Creates mini-batch tensors from the list of tuples (image, caption).

        We should build custom collate_fn rather than using default collate_fn,
        because merging caption (including padding) is not supported in default.
        Args:
            data: list of tuple (image, caption).
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.
        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, ids = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap)-1 for cap in captions]
        word_inputs = torch.zeros(len(captions), max(lengths)).long()
        word_targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            word_inputs[i, :end] = cap[:-1]
            word_targets[i, :end] = cap[1:]

        return images, word_inputs, word_targets, lengths, ids


    @staticmethod
    def tokenize(caption):
        """
        return [word for word in
                nltk.tokenize.word_tokenize(str(caption).rstrip('.').lower()) if word not
                in CocoDataset.PUNCTUATIONS]
        """
        t = PTBTokenizer()
        return t.tokenize_caption(caption)


    @staticmethod
    def build_tokenized_captions(json):
        coco = COCO(json)
        t = PTBTokenizer()
        tokenized_captions = t.tokenize(coco.anns)
        return tokenized_captions


    @staticmethod
    def get_tokenized_captions(data_path, train=True):
        # Load or construct tokenized captions
        if train:
            tokens_name = CocoDataset.tokens_train_file_name
        else:
            tokens_name = CocoDataset.tokens_val_file_name
        tokens_path = os.path.join(data_path, tokens_name)
        if os.path.exists(tokens_path):
            with open(tokens_path, 'rb') as f:
                tokens = pickle.load(f)
        else:
            if train:
                caption_path = CocoDataset.caption_train_path
            else:
                caption_path = CocoDataset.caption_val_path
            path = os.path.join(data_path, caption_path)
            tokens = CocoDataset.build_tokenized_captions(path)
            with open(tokens_path, 'wb') as f:
                pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved the tokenized captions to '%s'" %tokens_path)
        return tokens


    @staticmethod
    def build_vocab(json, tokenized_captions, threshold):
        """Build a simple vocabulary wrapper."""
        coco = COCO(json)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            """
            caption = str(coco.anns[id]['caption'])
            tokens = CocoDataset.tokenize(caption)
            """
            tokens = tokenized_captions[id]
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] captions tokenized." %(i, len(ids)))

        # If the word frequency is less than 'threshold', then the word is discarded.
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        # Creates a vocab wrapper and add some special tokens.
        vocab = Vocabulary()

        # Adds the words to the vocabulary.
        for word in words:
            vocab.add_word(word)

        print("Total vocabulary size: %d" %len(vocab))
        return vocab


    @staticmethod
    def get_vocabulary(data_path, tokenized_captions, threshold=1):
        # Load or construct vocabulary
        vocab_path = os.path.join(data_path, CocoDataset.vocab_file_name)
        if os.path.exists(vocab_path):
            vocab = Vocabulary.load(vocab_path)
        else:
            path = os.path.join(data_path, CocoDataset.caption_train_path)
            vocab = CocoDataset.build_vocab(path, tokenized_captions, threshold)
            Vocabulary.save(vocab, vocab_path)
            print("Saved the vocabulary to '%s'" %vocab_path)
        return vocab
