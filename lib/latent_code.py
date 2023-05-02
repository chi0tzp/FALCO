import os.path as osp
import torch
import torch.nn as nn
from lib import STYLEGAN2_STYLE_SPACE_LAYERS, STYLEGAN2_STYLE_SPACE_TARGET_LAYERS


class LatentCode(nn.Module):
    def __init__(self, latent_code_real, latent_code_fake_nn, img_id, out_code_dir, gan='stylegan2_ffhq1024',
                 latent_space='W+'):
        """Anonymization latent code class.

        Args:
            latent_code_real (torch.Tensor)    : real (inverted by e4e) image's latent code
            latent_code_fake_nn (torch.Tensor) : fake (NN) image's latent code
            img_id (int)                       : image id (corresponding to the given latent code)
            out_code_dir (str)                 : Latent code's output directory
            gan (str)                          : StyleGAN2 type ('stylegan2_ffhq1024' or 'stylegan2_ffhq512')
            latent_space (str)                 : StyleGAN2's latent space (W+ or S)

        """
        super(LatentCode, self).__init__()
        self.latent_code_real = latent_code_real
        self.latent_code_fake_nn = latent_code_fake_nn
        self.img_id = img_id
        self.gan = gan
        self.latent_space = latent_space

        if osp.isdir(out_code_dir):
            self.out_code_dir = out_code_dir
        else:
            raise NotADirectoryError("Invalid output latent code directory: {}".format(out_code_dir))

        # Define and initialise latent code parameters
        if self.latent_space == 'W+':

            self.layer_start = 3
            self.layer_end = 8

            self.nontrainable_layers_start = nn.Parameter(data=self.latent_code_real[:, :self.layer_start, :],
                                                          requires_grad=False)
            self.trainable_layers = nn.Parameter(data=self.latent_code_fake_nn[:, self.layer_start:self.layer_end, :],
                                                 requires_grad=True)
            self.nontrainable_layers_end = nn.Parameter(data=self.latent_code_real[:, self.layer_end:, :],
                                                        requires_grad=False)

        elif self.latent_space == 'S':
            raise NotImplementedError
            # self.latent_code = nn.ParameterDict()
            # for layer_id, dim in STYLEGAN2_STYLE_SPACE_LAYERS[self.gan].items():
            #     self.latent_code.update(
            #         {
            #             layer_id: nn.Parameter(data=self.latent_code_init[layer_id],
            #                                    requires_grad=layer_id in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[self.gan])
            #         })

    def do_optim(self):
        """Check whether optimization latent code has been saved."""
        if not osp.exists(osp.join(self.out_code_dir, '{}.pt'.format(self.img_id))):
            return True
        return False

    def save(self):
        """Save anonymization latent code."""
        if self.latent_space == 'W+':
            torch.save(torch.cat([self.nontrainable_layers_start,
                                  self.trainable_layers,
                                  self.nontrainable_layers_end], dim=1),
                       osp.join(self.out_code_dir, '{}.pt'.format(self.img_id)))
        else:
            raise NotImplementedError

    def forward(self):
        return torch.cat([self.nontrainable_layers_start,
                          self.trainable_layers,
                          self.nontrainable_layers_end], dim=1)
