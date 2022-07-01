import torch
from torchvision import transforms
import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape
from hypergan.gan_component import ValidationException

from timm.data import resolve_data_config


class Pretrained(hg.Layer):

    """
        ---
        description: layer pretrained
        ---

        # pretrained

        loads the model from timm

        ## syntax

        ```json
          "pretrained resnet18 layer=-1"
        ```

    """
    def __init__(self, component, args, options):
        super(Pretrained, self).__init__(component, args, options)
        channels = component.current_size.channels
        import timm

        model = timm.create_model(args[0], pretrained=options.pretrained or False)
        data = resolve_data_config({}, model=model)
        self.normalize = transforms.Normalize(mean=data['mean'], std=data['std'])
        trainable = True
        if options.trainable == False:
            trainable = False
        model.train(trainable)
        if options.layer:
            layers = list(model.children())[0:options.layer] + [list(model.children())[options.layer]]
            if options.sublayer:
                sublayers = layers[-1][0:options.sublayer] + [layers[-1][options.sublayer]]
                layers[-1] = nn.Sequential(*sublayers)
            print("Using pretrained network", options.layer, layers)
        else:
            #layers = [model]
            layers = list(model.children())
            print("List of pretrained layers:", layers)
            raise ValidationException("layer=-1 required for pretrained, sublayer=-1 optional.  Layers outputted above.")
        self.network = nn.Sequential(*layers).cuda()
        inp = component.gan.inputs.next()
        if type(inp) == type({}):
            inp = inp['img']
        if type(inp) == type(()):
            inp = inp[0]
        test_activation = self.network(inp.cuda())
        self.size = LayerShape(*list(test_activation.shape[1:]))

    def forward(self, input, context):
        return self.network(self.normalize((input + 1) / 2))

    def output_size(self):
        return self.size

