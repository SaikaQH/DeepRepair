
import numpy as np
import os
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

from .model import WaveEncoder, WaveDecoder

from .utils.core import feature_wct
from .utils.io import Timer, open_image, load_segment, compute_label_info

import tqdm


class WCT2:
    def __init__(self, model_path='./model_checkpoints', transfer_at=['encoder', 'skip', 'decoder'], option_unpool='cat5', device='cuda:0', verbose=False):

        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1):
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                # self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component], style_skips[skip_level][component],
                                                                       content_segment, style_segment,
                                                                       label_set, label_indicator,
                                                                       alpha=alpha, device=self.device)
                # self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                # self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat


class StyleTransfer:

    def __init__(self,
        style_transfer_config
    ):
        self.config = style_transfer_config

        self.dataset = self.config.dataset
        self.content_dir = self.config.content_dir
        self.content_segment_dir = self.config.content_segment_dir
        self.style_dir = self.config.style_dir
        self.style_segment_dir = self.config.style_segment_dir
        self.output_dir = self.config.output_dir
        self.sample_type = self.config.sample_type
        self.gpu = self.config.gpu

        # set device
        self.device_txt = 'cuda:{:1d}'.format(self.gpu)
        self.device = torch.device(self.device_txt)

        # wct2 transfer
        self.wct_model_path = self.config.wct_model_path
        self.wct2 = WCT2(model_path=self.wct_model_path, transfer_at={'decoder', 'encoder'}, option_unpool='cat5', device=self.device_txt, verbose=True)

    # transfer single image
    def single_transfer(self, cont_path, sty_path, output_path):
        content_img = open_image(cont_path, 32).to(self.device)
        style_img = open_image(sty_path, 32).to(self.device)

        content_segment = load_segment(None, 32)
        style_segment = load_segment(None, 32)

        with torch.no_grad():
            img = self.wct2.transfer(content_img, style_img, content_segment, style_segment, alpha=1)
        save_image(img.clamp_(0, 1), output_path, padding=0)

    # sample style image
    def __sample_sty_img(self, sty_img_list):
        if self.sample_type == "random":
            return np.random.choice(sty_img_list)

    # get content image's name
    def __get_cont_name(self, cont_full_name):
        if self.dataset == 'test':
            return cont_full_name[:-4]
        elif self.dataset == 'cifar':
            return cont_full_name[:7]   # 01234_5.jpg: 01234 -> name, 5 -> label

    # get style image's name
    def __get_sty_name(self, sty_full_name):
        if self.dataset == 'test':
            return sty_full_name[:-4]
        elif self.dataset == 'cifar':
            return sty_full_name[:5]

    # transfer all images in a directory
    def whole_dir_transfer(self, cont_dir, sty_dir, out_dir):
        
        cont_img_list = os.listdir(cont_dir)
        cont_img_list.sort()

        sty_img_list = os.listdir(sty_dir)
        sty_img_list.sort()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for cont_img in tqdm.tqdm(cont_img_list):
            cont_path = os.path.join(cont_dir, cont_img)
            cont_name = self.__get_cont_name(cont_img)

            sty_img = self.__sample_sty_img(sty_img_list)
            sty_path = os.path.join(sty_dir, sty_img)
            sty_name = self.__get_sty_name(sty_img)

            output_name = f"{cont_name}_{sty_name}.jpg"
            output_path = os.path.join(out_dir, output_name)

            self.single_transfer(cont_path, sty_path, output_path)







# ---- for python 3.9, not debug yet ----
# class transfer:
#     # dataset name (eg. cifar10, tiny-imagenet)     # not support tiny-imagenet yet
#     dataset: str = 'cifar10'

#     # source image dir
#     content_dir: str = "./examples/content"
#     content_segment_dir: str = None
#     # style image dir
#     style_dir: str = "./examples/style"
#     style_segment_dir: str = None
#     # output dir
#     output_dir: str = "./examples/outputs"

#     # use distance
#     use_dist: bool = False

#     # select device
#     gpu: int = 0
#     device_txt = 'cuda:{:1d}'.format(self.gpu)
#     device = torch.device(device_txt)

#     # wct2 transfer
#     wct2 = WCT2(transfer_at={'decoder', 'encoder'}, option_unpool='cat5', device=device_txt, verbose=True)

#     def __init__(self,
#         dataset: str,
#         content_dir: str, content_segment_dir: str,
#         style_dir: str, style_segment_dir: str, 
#         output_dir: str,
#         gpu: int
#     ):
#         self.dataset = dataset
#         self.content_dir = content_dir
#         self.content_segment_dir = content_segment_dir
#         self.style_dir = style_dir
#         self.style_segment_dir = style_segment_dir
#         self.output_dir = output_dir
#         self.gpu = gpu

#     # transfer single image
#     def single_transfer(cont_path, sty_path, output_path):
#         content_img = open_image(cont_path, 32).to(device)
#         style_img = open_image(sty_path, 32).to(device)

#         content_segment = load_segment(None, 32)
#         style_segment = load_segment(None, 32)

#         with torch.no_grad():
#             img = wct2.transfer(content_img, style_img, content_segment, style_segment, alpha=1)
#         save_image(img.clamp_(0, 1), output_path, padding=0)

#     # sample style image
#     def __sample_sty_img(sty_img_list: list):
#         if not use_dist:
#             return np.random.choice(sty_img_list)

#     # get content image's name
#     def __get_cont_name(cont_full_name: str):
#         if self.dataset == 'cifar':
#             return cont_full_name[:7]   # 01234_5.jpg: 01234 -> name, 5 -> label

#     # get style image's name
#     def __get_sty_name(sty_full_name: str):
#         if self.dataset == 'cifar':
#             return sty_full_name[:5]

#     # transfer all images in a directory
#     def whole_dir_transfer(
#         cont_dir: str = self.content_dir,
#         sty_dir: str = self.style_dir,
#         out_dir: str = self.output_dir 
#     ):
#         cont_img_list = os.listdir(cont_dir)
#         cont_img_list.sort()

#         sty_img_list = os.listdir(sty_dir)
#         sty_img_list.sort()

#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)

#         for cont_img in tqdm.tqdm(cont_img_list):
#             cont_path = os.path.join(cont_dir, cont_img)
#             cont_name = __get_cont_name(cont_img)

#             sty_img = __sample_sty_img(sty_img_list)
#             sty_path = os.path.join(sty_dir, sty_img)
#             sty_name = __get_sty_name(sty_img)

#             output_name = f"{cont_name}_{sty_name}.jpg"
#             output_path = os.path.join(out_dir, output_name)

#             single_transfer(cont_path, sty_path, output_path)

if __name__=="__main__":
    style_trans = StyleTransfer(gpu=1)
    style_trans.whole_dir_transfer()