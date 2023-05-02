import os
import os.path as osp
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage


def anon_exp_dir(args_dict, makedir=True):
    dir_name = osp.join('datasets', 'anon', args_dict['dataset'],
                        '{}-{}_anon_{}_m-{}_lambda-id-{}_lambda-attr-{}_{}-{}_epochs-{}_{}'.format(
                            args_dict['dataset'],args_dict['subset'],
                            args_dict['latent_space'],
                            args_dict['id_margin'],
                            args_dict['lambda_id'],
                            args_dict['lambda_attr'],
                            args_dict['optim'],
                            args_dict['lr'],
                            args_dict['epochs'],
                            osp.basename(args_dict['fake_nn_map']).split('.')[0]))
    if makedir:
        os.makedirs(dir_name, exist_ok=True)

    return dir_name


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def tensor2image(tensor, img_size=None, adaptive=False):
    # Squeeze tensor image
    tensor = tensor.squeeze(dim=0)
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
