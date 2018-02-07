# Python packages
import os

# Third party packages
import torch

# Local packages
from .coco_dataset import CocoDataset
from utils.transform import get_transform

class DataPreparation:
    def __init__(self, data_path='./data', batch_size=128, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def coco(self, vision_model, train, vocab=None, tokens=None):
        if tokens is None:
            tokens = CocoDataset.get_tokenized_captions(self.data_path, train)
        if vocab is None:
            if train:
                tokens_train = tokens
            else:
                tokens_train = CocoDataset.get_tokenized_captions(self.data_path, True)
            vocab = CocoDataset.get_vocabulary(self.data_path, tokens_train)

        if train:
            images_path = CocoDataset.image_train_path
            cap_path = CocoDataset.caption_train_path
            ids_based_on = CocoDataset.ID_BASE.CAPTIONS
        else:
            images_path = CocoDataset.image_val_path
            cap_path = CocoDataset.caption_val_path
            ids_based_on = CocoDataset.ID_BASE.IMAGES

        images_path = os.path.join(self.data_path, images_path)
        cap_path = os.path.join(self.data_path, cap_path)
        transform = get_transform(vision_model, train)


        dataset, loader = self.prepare_coco_loader(images_path,
                                                   cap_path,
                                                   vocab,
                                                   tokens,
                                                   ids_based_on,
                                                   transform,
                                                   train)

        return dataset, loader


    def prepare_coco_loader(self, images_path, captions_path, vocab, tokens,
                            ids_based_on, transform, shuffle):
        """Returns torch.utils.data.DataLoader for custom coco dataset."""
        # COCO caption dataset

        coco_dataset = CocoDataset(root=images_path,
                                   json=captions_path,
                                   vocab=vocab,
                                   tokenized_captions=tokens,
                                   ids_based_on=ids_based_on,
                                   transform=transform)

        # Data loader for COCO dataset
        # This will return (images, captions, lengths) for every iteration.
        # images: tensor of shape (batch_size, 3, 224, 224).
        # captions: tensor of shape (batch_size, padded_length).
        # lengths: list indicating valid length for each caption. length is (batch_size).
        coco_loader = torch.utils.data.DataLoader(dataset=coco_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=self.num_workers,
                                                  collate_fn=CocoDataset.collate_fn)

        return coco_dataset, coco_loader
