import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models

class PretrainedModel(nn.Module):

    ERR_TRUNC_MSG = ("{} currently only supports to be truncated "
                     "by its last {} FC layer(s). Please choose a value "
                     "between 0 and {}.")

    ERR_MODEL = "{} is currently not supported as a pretrained model."

    SUPPORTED_MODEL_NAMES = ['resnet18', 'resnet34', 'resnet50',
                             'resnet101', 'resnet152',
                             'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                             'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']
                             #'inception_v3']

    def __init__(self, model_name, layers_to_truncate=1):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(PretrainedModel, self).__init__()

        if model_name not in self.SUPPORTED_MODEL_NAMES:
            raise NotImplementedError(self.ERR_MODEL.format(model_name))

        """
        if model_name == 'inception_v3':
            self.input_size = (3, 299, 299)
            layer_size = 1
            max_trunc = 1
        """
        if model_name.startswith('resnet'):
            self.input_size = (3, 224, 224)
            layer_size = 1
            max_trunc = 1
        elif model_name.startswith('vgg'):
            self.input_size = (3, 224, 224)
            layer_size = 3
            max_trunc = 3
        else:
            raise NotImplementedError(self.ERR_MODEL.format(model_name))

        if layers_to_truncate > max_trunc:
            raise ValueError(self.ERR_TRUNC_MSG.format(model_name, max_trunc, max_trunc))

        model = getattr(torchvision.models, model_name)(pretrained=True)

        if layers_to_truncate < 1:
            # Do not truncate
            self.pretrained_model = model
        else:
            # Truncate last FC layer(s)
            if model_name.startswith('vgg'):
                layers = list(model.classifier.children())
            else:
                layers = list(model.children())
            trunc = self._get_num_truncated_layers(layers_to_truncate, layer_size)
            last_layer = layers[trunc]

            # Delete the last layer(s).
            modules = layers[:trunc]
            if model_name.startswith('vgg'):
                self.pretrained_model = model
                self.pretrained_model.classifier = nn.Sequential(*modules)
            else:
                self.pretrained_model = nn.Sequential(*modules)

        # Freeze all parameters of pretrained model
        for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # Switch model to eval mode (affects Dropout & BatchNorm)
        self.pretrained_model.eval()

        # TODO: Test if last_layer.in_features can be reliably used instead
        self.output_size = self._get_output_size()

    def _get_output_size(self):
        dummy_input = Variable(torch.rand(1, *self.input_size))
        output = self(dummy_input)
        output_size = output.data.view(-1).size(0)
        return output_size

    def _get_num_truncated_layers(self, num_to_trunc, layer_size, initial_layer_size=1):
        num = 0
        if num_to_trunc > 0:
            num += initial_layer_size
            num_to_trunc -= 1
        while num_to_trunc > 0:
            num += layer_size
            num_to_trunc -= 1
        return -num


    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.pretrained_model(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        return features
