import os
from collections import Counter
from enum import Enum
import pickle
import json
from PIL import Image

import torch
import torch.utils.data as data
import numpy as np
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
#import nltk

from utils.vocabulary import Vocabulary
from utils.tokenizer.ptbtokenizer import PTBTokenizer

# Adapted from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
class CubDataset(data.Dataset):

    """CUB Custom Dataset compatible with torch.utils.data.DataLoader."""

    dataset_prefix = 'cub'
    image_features_path = 'CUB_feature_dict.p'
    class_labels_path = 'CUB_label_dict.p'
    image_train_path = ''
    image_val_path = ''
    image_test_path = ''
    caption_train_path = 'descriptions_bird.train_noCub.fg.json'
    caption_val_path = 'descriptions_bird.val.fg.json'
    caption_test_path = 'descriptions_bird.test.fg.json'
    vocab_file_name = 'cub_vocab.pkl'
    tokens_train_file_name = 'cub_tokens_train.pkl'
    tokens_val_file_name = 'cub_tokens_val.pkl'
    tokens_test_file_name = 'cub_tokens_test.pkl'

    # Available data splits (must contain 'train')
    DATA_SPLITS = set(['train', 'val', 'test'])

    # punctuations to be removed from the sentences
    PUNCTUATIONS = ["''", "'", "``", "`", "(", ")", "{", "}",
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    class ID_BASE(Enum):
        CAPTIONS = 0
        IMAGES = 1


    def __init__(self, root, json, vocab, tokenized_captions, transform=None,
            ids_based_on=None, use_image_features=True):
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

        if use_image_features:
            self.load_img_features()
        self.load_class_labels()

        self.vocab = vocab
        self.tokens = tokenized_captions
        self.transform = transform

        self.input_size = next(iter(self.img_features.values())).shape[0]

    def load_img_features(self):
        path = os.path.join(self.root, CubDataset.image_features_path)
        with open(path, 'rb') as f:
            feature_dict = pickle.load(f, encoding='latin1')
        self.img_features = feature_dict

    def load_class_labels(self):
        path = os.path.join(self.root, CubDataset.class_labels_path)
        with open(path, 'rb') as f:
            label_dict = pickle.load(f, encoding='latin1')

        self.num_classes = len(set(label_dict.values()))
        self.class_labels = label_dict


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

        class_label = torch.LongTensor([int(self.class_labels[img_id])-1])
        tokens = self.tokens[ann_id]

        if self.img_features is not None:
            image = self.img_features[img_id]
            image = torch.Tensor(image)
        else:
            path = coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

        """
        # Convert caption (string) to word ids.
        tokens = CubDataset.tokenize(caption)
        """
        caption = []
        caption.append(vocab(vocab.start_token))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab(vocab.end_token))
        target = torch.Tensor(caption)
        return image, target, base_id, class_label


    def __len__(self):
        return len(self.ids)


    def eval(self, captions, checkpoint_path, score_metric='CIDEr'):
        # TODO: Make strings variables
        captions_path = checkpoint_path + "-val-captions.json"
        with open(captions_path, 'w') as f:
            json.dump(captions, f)
        cocoRes = self.coco.loadRes(captions_path)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        json.dump(cocoEval.evalImgs, open(checkpoint_path + "-val-metrics-imgs.json", 'w'))
        json.dump(cocoEval.eval,     open(checkpoint_path + "-val-metrics-overall.json", 'w'))

        print(cocoEval.eval.items())
        return cocoEval.eval[score_metric]



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
        images, captions, ids, labels = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)
        labels = torch.cat(labels, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap)-1 for cap in captions]
        word_inputs = torch.zeros(len(captions), max(lengths)).long()
        word_targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            word_inputs[i, :end] = cap[:-1]
            word_targets[i, :end] = cap[1:]

        return images, word_inputs, word_targets, lengths, ids, labels


    @staticmethod
    def tokenize(caption):
        """
        return [word for word in
                nltk.tokenize.word_tokenize(str(caption).rstrip('.').lower()) if word not
                in CubDataset.PUNCTUATIONS]
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
            tokens_name = CubDataset.tokens_train_file_name
        else:
            tokens_name = CubDataset.tokens_val_file_name
        tokens_path = os.path.join(data_path, tokens_name)
        if os.path.exists(tokens_path):
            with open(tokens_path, 'rb') as f:
                tokens = pickle.load(f)
        else:
            if train:
                caption_path = CubDataset.caption_train_path
            else:
                caption_path = CubDataset.caption_val_path
            path = os.path.join(data_path, caption_path)
            tokens = CubDataset.build_tokenized_captions(path)
            with open(tokens_path, 'wb') as f:
                pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved the tokenized captions to '%s'" %tokens_path)
        return tokens


    @staticmethod
    def build_vocab(json, tokenized_captions, threshold):
        print("Building vocabulary")
        coco = COCO(json)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            """
            caption = str(coco.anns[id]['caption'])
            tokens = CubDataset.tokenize(caption)
            """
            tokens = tokenized_captions[id]
            counter.update(tokens)

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
        vocab_path = os.path.join(data_path, CubDataset.vocab_file_name)
        if os.path.exists(vocab_path):
            vocab = Vocabulary.load(vocab_path)
        else:
            path = os.path.join(data_path, CubDataset.caption_train_path)
            vocab = CubDataset.build_vocab(path, tokenized_captions, threshold)
            Vocabulary.save(vocab, vocab_path)
            print("Saved the vocabulary to '%s'" %vocab_path)
        return vocab
