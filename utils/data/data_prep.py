# Python packages
import os

# Third party packages
import torch
import torchvision.transforms as transforms

# Local packages
from .coco_dataset import CocoDataset
from .cub_dataset import CubDataset
from utils.transform import get_transform

class DataPreparation:
    def __init__(self, data_path='./data', batch_size=128, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def coco_cub_preparation(self, vision_model, train, vocab=None,
            tokens=None, dataset_name='coco'):


        if dataset_name == 'coco':
            Dataset = CocoDataset
            transform = get_transform(vision_model, train)
        elif dataset_name == 'cub':
            Dataset = CubDataset
            transform = None

        data_path = os.path.join(self.data_path, Dataset.dataset_prefix)

        if tokens is None:
            tokens = Dataset.get_tokenized_captions(data_path, train)
        if vocab is None:
            if train:
                tokens_train = tokens
            else:
                tokens_train = Dataset.get_tokenized_captions(data_path, True)
            vocab = Dataset.get_vocabulary(data_path, tokens_train)

        if train:
            images_path = Dataset.image_train_path
            cap_path = Dataset.caption_train_path
            ids_based_on = Dataset.ID_BASE.CAPTIONS
        else:
            images_path = Dataset.image_val_path
            cap_path = Dataset.caption_val_path
            ids_based_on = Dataset.ID_BASE.IMAGES

        images_path = os.path.join(data_path, images_path)
        cap_path = os.path.join(data_path, cap_path)

        dataset, loader = self.prepare_coco_cub_loader(Dataset,
                                                       images_path,
                                                       cap_path,
                                                       vocab,
                                                       tokens,
                                                       ids_based_on,
                                                       transform,
                                                       train)

        return dataset, loader

    def coco(self, vision_model, train, vocab=None, tokens=None):
        return self.coco_cub_preparation(vision_model, train, vocab=None,
                tokens=None, dataset_name='coco')

    def cub(self, vision_model, train, vocab=None, tokens=None):
        return self.coco_cub_preparation(vision_model, train, vocab=None,
                tokens=None, dataset_name='cub')

    def prepare_coco_cub_loader(self, Dataset, images_path, captions_path, vocab, tokens,
                            ids_based_on, transform, shuffle):
        """Returns torch.utils.data.DataLoader for custom coco dataset."""
        # COCO caption dataset

        coco_dataset = Dataset(root=images_path,
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
                                                  collate_fn=Dataset.collate_fn)

        return coco_dataset, coco_loader
