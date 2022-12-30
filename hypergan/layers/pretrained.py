import torch

import torchvision.transforms.functional as fn

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
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])
        channels = component.current_size.channels
        self.gan = component.gan
        self.jepa = False
        self.convnext = False
        model_name = args[0]
        self.resize = transforms.Resize(224)
        if model_name == 'convnext':
            self.convnext = True
            from transformers import ConvNextImageProcessor, ConvNextModel
            import torch

            self.feature_extractor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224", local_files_only=True)
            self.model = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
            self.model.eval().requires_grad_(False)
            self.size = component.current_size
            return

        if model_name == "jepa":
            from data2vec.models.data2vec_vision import Data2VecVisionConfig, Data2VecVisionModel
            from data2vec.models.data2vec_image_classification import Data2VecImageClassificationConfig
            from fairseq.dataclass.initialize import hydra_init
            from fairseq import checkpoint_utils, tasks


            self.jepa = True
            state = checkpoint_utils.load_checkpoint_to_cpu("base_imagenet.pt", {})
            pretrained_args = state.get("cfg", None)
            pretrained_args.criterion = None
            pretrained_args.lr_scheduler = None
            task = tasks.setup_task(pretrained_args.task)
            #pretrained_args.task.data = cfg.data
            model = task.build_model(pretrained_args.model, from_checkpoint=True)
            model.remove_pretraining_modules()
            model.load_state_dict(state["model"], strict=True)
            self.data2vec = model
            self.data2vec = self.data2vec.eval().requires_grad_(False).to('cuda:0')
            return


        import timm

        model = timm.create_model(model_name, pretrained=options.pretrained or False)
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
        self.network.eval()
        self.network.requires_grad_(False)

        inp = component.gan.inputs.next()
        if type(inp) == type({}):
            inp = inp['img']
        if type(inp) == type(()):
            inp = inp[0]
        test_activation = self.network(self.resize(inp.cuda()))
        self.size = LayerShape(*list(test_activation.shape[1:]))

    def forward(self, input, context):
        if self.convnext:
            inputs = self.normalize(self.resize(input/2+0.5))
            return self.model(inputs).last_hidden_state
        if self.jepa:
            return self.data2vec(inpx, mask=False, features_only=True)["layer_results"][-1]
        s = input.shape
        input = input.reshape(-1,3,s[2],s[3])
        input = self.resize(input)
        #self.gan.add_metric('w', self.network[0].weight.mean())
        with torch.no_grad():
            return self.network((self.normalize(input/2+0.5)))#, size=[self.size.dims[-1]])

    def output_size(self):
        return self.size

