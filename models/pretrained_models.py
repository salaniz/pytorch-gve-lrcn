import torch.nn as nn
import torchvision.models as models

class PretrainedModel(nn.Module):
    def __init__(self, model_name, output_dim):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(PretrainedModel, self).__init__()

        self.supported_model_names = ['inception_v3',
                                      'resnet18', 'resnet34', 'resnet50',
                                      'resnet101', 'resnet152',
                                      'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                                      'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']

        if model_name not in self.supported_model_names:
            print("WARNING: The requested pretrained model was not tested " +
                    "in this implementation. The code will still run if " +
                    "torch supports this model.")

        model = getattr(models, model_name)(pretrained=True)
        layers = list(model.children())
        last_layer = layers[-1]
        if isinstance(last_layer, nn.Sequential):
            last_layer = last_layer[0]

        modules = layers[:-1] # delete the last fc layer.
        self.pretrained_model = nn.Sequential(*modules)
        # Freeze all parameters of pretrained model
        for param in self.pretrained_model.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(last_layer.in_features, output_dim)
        #self.bn = nn.BatchNorm1d(output_dim, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.retrained_model(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        #features = self.bn(self.linear(features))
        features = self.linear(features)
        return features
