# Python packages
import os
from collections import Counter
import pickle
from PIL import Image
#from pathlib import Path

# Third party packages
import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import nltk

# Local packages
from .vocabulary import Vocabulary

class DataPreparation:
    def __init__(self, data_path='./data', batch_size=64, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_coco_data(self, vocab=None):
        train_path = os.path.join(self.data_path, CocoDataset.image_train_path)
        val_path = os.path.join(self.data_path, CocoDataset.image_val_path)
        cap_train_path = os.path.join(self.data_path,
                                      CocoDataset.caption_train_path)
        cap_val_path = os.path.join(self.data_path,
                                    CocoDataset.caption_val_path)

        if vocab is None:
            vocab = CocoDataset.get_vocabulary(self.data_path)

        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))])

        coco_caption_train = CocoDataset(root=train_path,
                                         json=cap_train_path,
                                         vocab=vocab,
                                         transform=transform)

        coco_caption_val = CocoDataset(root=val_path,
                                         json=cap_val_path,
                                         vocab=vocab,
                                         transform=transform)


        train_loader = torch.utils.data.DataLoader(dataset=coco_caption_train,
                                                           batch_size=self.batch_size,
                                                           shuffle=True,
                                                           num_workers=self.num_workers,
                                                           collate_fn=CocoDataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(dataset=coco_caption_val,
                                                         batch_size=self.batch_size,
                                                         shuffle=False,
                                                         num_workers=self.num_workers,
                                                         collate_fn=CocoDataset.collate_fn)


        return train_loader, val_loader, vocab



# Adapted from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    image_train_path = 'train2014'
    image_val_path = 'val2014'
    caption_train_path = 'annotations/captions_train2014.json'
    caption_val_path = 'annotations/captions_val2014.json'
    vocab_file_name = 'coco_vocab.pkl'

    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab(vocab.start_token))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab(vocab.end_token))
        target = torch.Tensor(caption)
        return image, target

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
        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return images, targets, lengths


    @staticmethod
    def build_vocab(json, threshold):
        """Build a simple vocabulary wrapper."""
        coco = COCO(json)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
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
    def get_vocabulary(data_path, threshold=4):
        # Load or construct vocabulary
        print(data_path)
        vocab_path = os.path.join(data_path, CocoDataset.vocab_file_name)
        print(data_path)
        if os.path.exists(vocab_path):
            vocab = Vocabulary.load(vocab_path)
        else:
            path = os.path.join(data_path, CocoDataset.caption_train_path)
            vocab = CocoDataset.build_vocab(path, threshold)
            Vocabulary.save(vocab, vocab_path)
            print("Saved the vocabulary to '%s'" %vocab_path)
        return vocab


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
