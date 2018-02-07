from torch.autograd import Variable

def to_var(x, use_cuda, volatile=False):
    if use_cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)
