from torch.autograd import Variable

def to_var(x, use_cuda, volatile=False):
    if use_cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)

def get_split_str(train, test=False):
    if train:
        return 'train'
    if test:
        return 'test'
    return 'val'
