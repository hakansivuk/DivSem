import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_network import BaseNetwork
import random
from .blocks import Conv2dBlock

def inst_id_to_label_id(inst_id: int) -> int:
    return int(inst_id) % 120

class DeeperZencoder(BaseNetwork):
    def __init__(self,cfg, input_nc = 3, output_nc = 512, ngf=32, n_downsampling=2, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.cfg = cfg
        self.n_downsampling = 6
        self.non_spade_norm_layer = norm_layer
        self.output_nc = cfg["style_length"]
        input_nc = input_nc +1 #RGB + Valid

        self.gamma = nn.Linear(1, 1)
        self.beta = nn.Linear(1, 1)
        ### downsample
        input_nc = cfg['input_nc']
        ngf = cfg['ngf']
        output_nc = cfg['style_length']
        lab_nc = cfg['lab_dim'] + 1
        g_norm = cfg['G_norm_type']
        self.enc1 = Conv2dBlock(4, 16, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu') # 16, 256, 256
        self.enc2 = Conv2dBlock(16, 32, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu') # 32, 128, 128
        self.enc3 = Conv2dBlock(32, 64, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu') # 64, 64, 64
        self.enc4 = Conv2dBlock(64, 128, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu') # 128, 32, 32
        self.enc5 = Conv2dBlock(128, 256, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu') # 256, 16, 16
        self.enc6 = Conv2dBlock(256, 512, kernel_size=3, stride=1, padding=1, dilation=1, norm=g_norm, activation='lrelu') # 512, 16, 16

        # Decoder layers
        self.dec5 = Conv2dBlock(512+128, 512, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec4 = Conv2dBlock(512+64, 256, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec3 = Conv2dBlock(256+32, 256, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec2 = Conv2dBlock(256+16, 256, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec1 = Conv2dBlock(256, output_nc, kernel_size=3, stride=1, padding=1, norm='none', activation='tanh')

    def forward(self, input, segmap, valids, instance_map=None, cached_codes=None):
        masked_input = input*valids
        masked_input = torch.cat((masked_input, valids), 1)
        codes = self._forward_layers(masked_input)

        original_input = torch.cat((input, torch.ones(valids.shape, device=valids.device, dtype=valids.dtype)), 1)
        unmasked_codes = self._forward_layers(original_input)

        if instance_map is None:
            styles_map = segmap
        else:
            styles_map = instance_map
        styles_map = F.interpolate(styles_map, size=codes.size()[2:], mode='nearest')
        valids = F.interpolate(valids, size=codes.size()[2:], mode='nearest')
        # instance-wise average pooling
        style_codes = codes.clone()
        for b in range(input.size()[0]):
            inst_list = torch.unique(styles_map[b]).to(torch.long)
            for i in inst_list:
                indices = (styles_map[b:b+1] == int(i)).nonzero() # n x 4 
                valid_ins = valids[indices[:,0] + b, :, indices[:,2], indices[:,3]]
                if valid_ins.sum() > 0:
                    output_ins = codes[indices[:,0] + b, :, indices[:,2], indices[:,3]]
                    mean_feat = output_ins.mean(dim=0).expand_as(output_ins)
                    valid_ratio = valid_ins.mean().unsqueeze(0)
                    mean_feat = mean_feat * torch.sigmoid(self.gamma(valid_ratio)) + torch.sigmoid(self.beta(valid_ratio))
                elif self.cfg["is_train"]:
                    unmasked_output_ins = unmasked_codes[indices[:,0] + b, :, indices[:,2], indices[:,3]]
                    mean_feat = unmasked_output_ins.mean(dim=0).expand_as(unmasked_output_ins)
                    valid_ratio = torch.ones((1,))
                    mean_feat = mean_feat * torch.sigmoid(self.gamma(valid_ratio)) + torch.sigmoid(self.beta(valid_ratio))
                else:
                    code_list = cached_codes[inst_id_to_label_id(i)]
                    random_code = random.choice(code_list)
                    mean_feat = random_code.expand_as(codes[indices[:,0] + b, :, indices[:,2], indices[:,3]])
                    valid_ratio = torch.ones((1,))
                    mean_feat = mean_feat * torch.sigmoid(self.gamma(valid_ratio)) + torch.sigmoid(self.beta(valid_ratio))
                style_codes[indices[:,0] + b, :, indices[:,2], indices[:,3]] = mean_feat
        return style_codes

    def _forward_layers(self, input):
        # Encoder
        e1 = self.enc1(input)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        x = self.enc6(e5)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((x, e4), dim=1)
        x = self.dec5(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((x, e3), dim=1)
        x = self.dec4(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((x, e2), dim=1)
        x = self.dec3(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat((x, e1), dim=1)
        x = self.dec2(x)
        
        x = self.dec1(x)

        return x

