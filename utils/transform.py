import torchvision.transforms as transforms

def get_transform(net, train=True):

    if net is None:
        transform = None
    elif net.startswith('vgg'):
        if train:
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.RandomCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))])
        else:
            transform = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))])
    else:
        raise NotImplementedError("{} is missing a data transform "
                                  "implementation".format(net))

    return transform


# Credits: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


