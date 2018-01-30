import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

class PretrainedModel(nn.Module):
    def __init__(self, model_name):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(PretrainedModel, self).__init__()

        self.supported_model_names = ['inception_v3',
                                      'resnet18', 'resnet34', 'resnet50',
                                      'resnet101', 'resnet152',
                                      'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                                      'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']

        if model_name == 'inception_v3':
            self.input_size = (3, 299, 299)
        elif model_name.startswith('resnet') or model_name.startswith('vgg'):
            self.input_size = (3, 224, 224)
        else:
            # Default to (i.e. 'try')
            self.input_size = (3, 224, 224)

        if model_name not in self.supported_model_names:
            print("WARNING: The requested pretrained model was not tested " +
                  "in this implementation. The code might still run if " +
                  "torch supports this model.")

        model = getattr(models, model_name)(pretrained=True)
        layers = list(model.children())
        last_layer = layers[-1]
        if isinstance(last_layer, nn.Sequential):
            last_layer = last_layer[0]

        # Delete the last layer.
        modules = layers[:-1]
        self.pretrained_model = nn.Sequential(*modules)
        # Freeze all parameters of pretrained model
        for param in self.pretrained_model.parameters():
                param.requires_grad = False

        self.output_size = self._get_output_size()

    def _get_output_size(self):
        dummy_input = Variable(torch.rand(1, *self.input_size))
        output = self(dummy_input)
        output_size = output.data.view(-1).size(0)
        return output_size

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.pretrained_model(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        return features
