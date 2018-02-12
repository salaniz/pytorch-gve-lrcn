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
class CocoDataset(data.Dataset):

    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    dataset_prefix = 'coco'
    image_path = '{}2014'
    caption_path = 'annotations/captions_{}2014.json'
    vocab_file_name = 'coco_vocab.pkl'
    tokens_file_name = 'coco_tokens_{}.pkl'
    class_labels_path = 'annotations/instances_{}2014.json'

    # Available data splits (must contain 'train')
    DATA_SPLITS = set(['train', 'val'])

    # punctuations to be removed from the sentences
    PUNCTUATIONS = ["''", "'", "``", "`", "(", ")", "{", "}",
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    class ID_BASE(Enum):
        CAPTIONS = 0
        IMAGES = 1


    def __init__(self, root, split='train', vocab=None, tokenized_captions=None,
            transform=None):
        """
        Args:
            root: directory of coco data
            split: one of ['train', 'val']
        """

        cls = self.__class__
        assert split in cls.DATA_SPLITS
        self.split = split

        self.root = root

        self.caption_path = os.path.join(self.root, cls.caption_path.format(split))
        self.image_path = os.path.join(self.root, cls.image_path.format(split))
        self.tokens_path = os.path.join(self.root, cls.tokens_file_name.format(split))
        self.vocab_path = os.path.join(self.root, cls.vocab_file_name)
        self.labels_path = os.path.join(self.root, cls.class_labels_path.format(split))

        if tokenized_captions is None:
            tokenized_captions = cls.get_tokenized_captions(self.caption_path,
                    self.tokens_path)

        if vocab is None:
            if split != 'train':
                cap_path_train = os.path.join(self.root, cls.caption_path.format('train'))
                tokens_path_train = os.path.join(self.root, cls.tokens_file_name.format('train'))
                tokens_train = cls.get_tokenized_captions(cap_path_train,
                        tokens_path_train)
            else:
                cap_path_train = self.caption_path
                tokens_train = tokenized_captions
            vocab = cls.get_vocabulary(self.vocab_path, cap_path_train, tokens_train)

        if split == 'train':
            self.ids_based_on = cls.ID_BASE.CAPTIONS
        else:
            self.ids_based_on = cls.ID_BASE.IMAGES

        #images_path = os.path.join(data_path, images_path)
        #cap_path = os.path.join(data_path, cap_path)
        # TODO: separate
        #self.root = os.path.join(root, self.image_path)

        self.coco = COCO(self.caption_path)
        if self.ids_based_on == self.ID_BASE.CAPTIONS:
            self.ids = list(self.coco.anns.keys())
        elif self.ids_based_on == self.ID_BASE.IMAGES:
            self.ids = list(self.coco.imgs.keys())
        else:
            raise ValueError("Chosen base for COCO IDs is not implemented")

        #self.load_class_labels(self.labels_path)
        self.return_labels = False

        self.vocab = vocab
        self.tokens = tokenized_captions
        self.transform = transform


    def set_label_usage(self, return_labels):
        if return_labels and not hasattr(self, 'class_labels'):
            self.load_class_labels(self.labels_path)
        self.return_labels = return_labels


    def load_class_labels(self, category_path, use_supercategories=False):
        coco = COCO(category_path)
        id_to_label = {}
        class_labels = {}
        i = 0
        sam = 0
        for key, info in coco.cats.items():
            if use_supercategories:
                cat = info["supercategory"]
            else:
                cat = key

            if cat not in id_to_label:
                id_to_label[cat] = i
                i += 1

            label_id = id_to_label[cat]
            for img in coco.catToImgs[key]:
                if img not in class_labels:
                    class_labels[img] = [label_id]
                elif label_id not in class_labels[img]:
                    class_labels[img].append(label_id)

            sam += len(coco.catToImgs[key])

        # Add label for all images that have no label
        l = 'not labeled'
        for img in self.coco.imgs.keys():
            if img not in class_labels:
                if l not in id_to_label:
                    id_to_label[l] = i
                    i += 1
                class_labels[img] = [id_to_label[l]]

        self.class_labels = class_labels
        self.num_classes = len(id_to_label)


    def get_image(self, img_id):
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.image_path, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_class_label(self, img_id):
        img_labels = self.class_labels[img_id]
        rand_idx = np.random.randint(len(img_labels))
        class_label = torch.LongTensor([int(img_labels[rand_idx])])
        return class_label


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


        if self.return_labels:
            class_label = self.get_class_label(img_id)

        tokens = self.tokens[ann_id]
        image = self.get_image(img_id)

        """
        # Convert caption (string) to word ids.
        tokens = CocoDataset.tokenize(caption)
        """
        caption = []
        caption.append(vocab(vocab.start_token))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab(vocab.end_token))
        target = torch.Tensor(caption)
        if self.return_labels:
            return image, target, base_id, class_label
        else:
            return image, target, base_id


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
        images, captions, ids, *labels = zip(*data)

        if len(labels) > 0:
            return_labels = True
            labels = torch.cat(labels[0], 0)
        else:
            return_labels = False

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

        if return_labels:
            return images, word_inputs, word_targets, lengths, ids, labels
        else:
            return images, word_inputs, word_targets, lengths, ids



    @classmethod
    def tokenize(cls, caption):
        """
        return [word for word in
                nltk.tokenize.word_tokenize(str(caption).rstrip('.').lower()) if word not
                in CocoDataset.PUNCTUATIONS]
        """
        t = PTBTokenizer()
        return t.tokenize_caption(caption)


    @classmethod
    def build_tokenized_captions(cls, json):
        coco = COCO(json)
        t = PTBTokenizer()
        tokenized_captions = t.tokenize(coco.anns)
        return tokenized_captions


    @classmethod
    def get_tokenized_captions(cls, caption_path, target_path):
        # Load or construct tokenized captions
        if os.path.exists(target_path):
            with open(target_path, 'rb') as f:
                tokens = pickle.load(f)
        else:
            tokens = cls.build_tokenized_captions(caption_path)
            with open(target_path, 'wb') as f:
                pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved the tokenized captions to '{}'".format(target_path))
        return tokens


    @classmethod
    def build_vocab(cls, json, tokenized_captions, threshold):
        print("Building vocabulary")
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

        # If the word frequency is less than 'threshold', then the word is discarded.
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        # Creates a vocab wrapper and add some special tokens.
        vocab = Vocabulary()

        # Adds the words to the vocabulary.
        for word in words:
            vocab.add_word(word)

        print("Total vocabulary size: %d" %len(vocab))
        return vocab


    @classmethod
    def get_vocabulary(cls, vocab_path, captions_path, tokenized_captions, threshold=1):
        # Load or construct vocabulary
        if os.path.exists(vocab_path):
            vocab = Vocabulary.load(vocab_path)
        else:
            vocab = cls.build_vocab(captions_path, tokenized_captions, threshold)
            #TODO: check if saving is safe
            Vocabulary.save(vocab, vocab_path)
            print("Saved the vocabulary to '%s'" %vocab_path)
        return vocab
