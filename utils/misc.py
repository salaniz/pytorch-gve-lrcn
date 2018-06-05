def get_split_str(train, test=False, dataset=None):
    if train:
        return 'train'
    if test and dataset != 'coco':
        return 'test'
    return 'val'
