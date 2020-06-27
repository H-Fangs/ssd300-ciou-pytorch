import pretrainedmodels
import torch.nn as nn
from torchsummary import summary
from ..utils import ConvModule

from .new_senet import *


class Backbone(nn.Module):
    def __init__(self, model_name, feature_map, phase):
        super(Backbone, self).__init__()
        self.normalize = {'type': 'BN'}
        self.phase = phase
        lay, channal = self.get_pretrainedmodel(model_name)
        self.model = self.add_extras(lay, channal)
        self.model_length = len(self.model)
        self.feature_map = feature_map

    def get_pretrainedmodel(self, model_name, pretrained='imagenet'):  # 'imagenet'
        '''
        get the pretraindmodel lay
        args:
            model_name
            pretrained:None or imagenet
        '''
        if self.phase == "test":
            pretrained = None
        new_model_name = model_name
        new_model = eval(new_model_name)(num_classes=1000, pretrained=pretrained)
        # if self.phase == "train":
        #     init_model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=pretrained)
        #     init_model_dict = init_model.state_dict()
        #     new_model_dict = new_model.state_dict()
        #     init_model_dict = {k: v for k, v in init_model_dict.items() if k in new_model_dict}
        #     new_model_dict.update(init_model_dict)
        #     new_model.load_state_dict(new_model_dict)

        lay = nn.Sequential(*list(new_model.children())[:-2])
        out_channels = 2048
        return lay, out_channels

    def add_extras(self, lay, in_channel):
        exts1 = nn.Sequential(
            ConvModule(2048, 256, 1, normalize=None, stride=1,
                       bias=True, inplace=False),
            ConvModule(256, 512, 3, normalize=None, stride=2, padding=1,
                       bias=True, inplace=False)

            # nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3 ,stride = 2, padding = 1)
        )
        lay.add_module("exts1", exts1)

        exts2 = nn.Sequential(
            ConvModule(512, 128, 1, normalize=None, stride=1,
                       bias=True, inplace=False),
            ConvModule(128, 256, 3, normalize=None, stride=2, padding=1,
                       bias=True, inplace=False)

        )
        lay.add_module("exts2", exts2)

        exts3 = nn.Sequential(
            ConvModule(256, 128, 1, normalize=None, stride=1,
                       bias=True, inplace=False),
            ConvModule(128, 256, 3, normalize=None, stride=1, padding=0,
                       bias=True, inplace=False)
        )
        lay.add_module("exts3", exts3)

        # new add
        # exts4 = nn.Sequential(
        #     ConvModule(256,128,1,normalize=None,stride = 1,
        #         bias=True,inplace=False),
        #     ConvModule(128,256,1,normalize=None,stride = 1,padding = 0,
        #         bias=True,inplace=False)
        #     )
        # lay.add_module("exts4",exts4)
        return lay

    def forward(self, x):
        outs = []

        for i in range(self.model_length):
            x = self.model[i](x)

            tmp_size = x.size(2)
            if tmp_size in self.feature_map:
                outs.append(x)
        # for i in range(len(outs)):
        # print(outs[i].shape[1])
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)


if __name__ == '__main__':
    import torch.nn as nn

    use_gpu = True
    model_name = 'resnet50'

    # could be fbresnet152 or inceptionresnetv2
    feature_map = [6, 7, 8, 9, 10, 11]
    bone_model = Backbone(model_name, feature_map)
    if use_gpu:
        bone_model.cuda()
        summary(bone_model, (3, 300, 300))
