def get_split_str(train, test=False):
    if train:
        return 'train'
    if test:
        return 'test'
    return 'val'
