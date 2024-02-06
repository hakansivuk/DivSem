from model.networks.generator import Generator
from model.utils import weights_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class InferenceHandler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # setting basic params
        self.cfg = cfg
        self.model_names = ['netG']

        # Initiate the submodules and initialization params
        self.netG = Generator(self.cfg)

        self.netG.apply(weights_init('gaussian'))

        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() \
            else torch.ByteTensor

    def set_input(self, input):
        # scatter_ require .long() type
        input['lab'] = input['lab'].long()
        self.masked_img = input['masked_img']    # mask image
        self.gt = input['img']   # real image
        # self.img_know = input['img_know'].cuda()
        self.mask = input['mask']    # mask image
        self.lab = input['lab']  # label image

        self.name = input['name']

        # create one-hot label map
        lab_map = self.lab
        bs, _, h, w = lab_map.size()
        nc = self.cfg['lab_dim']
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        self.segmap = input_label.scatter_(1, lab_map, 1.0)
        # print(' segmap ',self.lab.shape)
        
        self.segmap = self.segmap * self.mask

        self.inst_map = input['inst_map']
        self.edge_map = self.get_edges(self.inst_map)
        self.edge_map = self.edge_map * self.mask
        
        self.segmap_edge = torch.cat((self.segmap, self.edge_map), dim=1)

        self.segmap_G1 = F.interpolate(self.segmap, size=(64, 64), mode='nearest')
        self.segmap_G2 = F.interpolate(self.segmap, size=(128, 128), mode='nearest')
        self.segmap_G3 = self.segmap


    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        
        return edge.float()

    def forward(self, cached_codes):
        gt_list, input_list, mask_fake_list, fake_list = self.netG(self.gt, self.masked_img, self.segmap_edge, self.inst_map, self.mask, cached_codes=cached_codes)

        self.gt_G1, self.gt_G2, self.gt_G3 = gt_list
        self.input_G1, self.input_G2, self.input_G3 = input_list
        self.mask_fake_G1, self.mask_fake_G2, self.mask_fake_G3 = mask_fake_list
        self.fake_G1, self.fake_G2, self.fake_G3 = fake_list

    def get_results(self):
        return self.mask_fake_G3

    def load_checkpoint(self, ckpt_filename):
        ckpt = torch.load(os.path.join(ckpt_filename), map_location=torch.device("cpu"))
        for name in self.model_names:
            net = getattr(self, name)
            net.load_state_dict(ckpt[name])
