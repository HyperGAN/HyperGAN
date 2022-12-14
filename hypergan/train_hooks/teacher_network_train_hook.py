import torch
from torchvision import transforms
import hyperchamber as hc
import open_clip
import numpy as np
import inspect
from torch.nn import functional as F
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.nn.parameter import Parameter
from PIL import Image
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from hypergan.train_hooks.clip_prompt_train_hook import ClipPromptTrainHook, MakeCutouts
from data2vec.models.data2vec_vision import Data2VecVisionConfig, Data2VecVisionModel
from data2vec.models.data2vec_image_classification import Data2VecImageClassificationConfig
from data2vec.tasks.mae_image_pretraining import *
from data2vec.models.data2vec2 import *
from fairseq.dataclass.initialize import hydra_init
from fairseq import checkpoint_utils, tasks

class TeacherNetworkTrainHook(BaseTrainHook):
    def create(self):
        cut_size = 224
        self.make_cutouts = MakeCutouts(cut_size, self.config.cutn or 1, cut_pow=1)
        state = checkpoint_utils.load_checkpoint_to_cpu("base_imagenet.pt", {})
        pretrained_args = state.get("cfg", None)
        pretrained_args.criterion = None
        pretrained_args.lr_scheduler = None
        task = tasks.setup_task(pretrained_args.task)
        pretrained_args.task.data = cfg.data
        model = task.build_model(pretrained_args.model, from_checkpoint=True)
        model.remove_pretraining_modules()
        model.load_state_dict(state["model"], strict=True)
        self.data2vec = model
        self.data2vec = self.data2vec.eval().requires_grad_(False).to('cuda:0')

    def forward(self, d_loss, g_loss):
        inpx = self.gan.x
        layer = self.data2vec(inpx, mask=False, features_only=True)["layer_results"][-1]
        prediction = self.gan.discriminator.context[self.config.layer_name]
        target = (layer > 0).float()
        #loss = F.cross_entropy(prediction.view(target.shape[0], -1), target.view(target.shape[0], -1))# * 0.05
        loss = F.mse_loss(prediction.view(target.shape[0], -1), target.view(target.shape[0], -1))# * 0.05
        self.add_metric('ce', loss)
        self.add_metric('t', target.mean())
        return loss, loss
