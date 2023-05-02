import os.path as osp
import torch
import torch.nn as nn
from torchvision import transforms
import clip
from lib import FARL_PRETRAIN_MODEL


class AttrLoss(nn.Module):
    def __init__(self, feat_ext='farl', use_cuda=True):
        super(AttrLoss, self).__init__()
        self.feat_ext = feat_ext
        self.use_cuda = use_cuda

        if feat_ext not in ('clip', 'farl', 'dino'):
            raise NotImplementedError

        self.feat_ext_model = None
        self.feat_ext_transform = None
        if self.feat_ext == 'clip':
            self.feat_ext_model, _ = clip.load("ViT-B/32", device='cuda' if self.use_cuda else 'cpu', jit=False)
            self.feat_ext_model.float()
            self.feat_ext_model.eval()

            self.feat_ext_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                          transforms.CenterCrop(224),
                                                          transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                               (0.26862954, 0.26130258, 0.27577711))])

        elif self.feat_ext == 'farl':
            self.feat_ext_model, _ = clip.load("ViT-B/16", device='cuda' if self.use_cuda else 'cpu', jit=False)
            farl_state = torch.load(osp.join('models', 'pretrained', 'farl', FARL_PRETRAIN_MODEL))
            self.feat_ext_model.load_state_dict(farl_state["state_dict"], strict=False)
            self.feat_ext_model.eval()
            self.feat_ext_model.float()
            self.feat_ext_model = self.feat_ext_model.visual

            # self.feat_ext_transform = transforms.Compose([transforms.Resize(224),
            #                                               transforms.CenterCrop(224),
            #                                               transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
            #                                                                    (0.26862954, 0.26130258, 0.27577711))])
            self.feat_ext_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                          transforms.CenterCrop(224)])

        elif self.feat_ext == 'dino':
            # REVIEW: CUDA????
            self.feat_ext_model = torch.hub.load("facebookresearch/dino:main", 'dino_vitb16')
            self.feat_ext_model.eval()
            self.feat_ext_model.float()

            self.feat_ext_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                          transforms.CenterCrop(224),
                                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                                               (0.229, 0.224, 0.225))])

        # REVIEW
        self.feat_ext_model.to('cuda' if self.use_cuda else 'cpu')

        # TODO: add comment
        self.l1_loss = nn.L1Loss()

    def extract_visual(self, x):
        x = self.feat_ext_model.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.feat_ext_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.feat_ext_model.positional_embedding.to(x.dtype)
        x = self.feat_ext_model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.feat_ext_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        fts = x.clone()

        x = self.feat_ext_model.ln_post(x[:, 0, :])

        if self.feat_ext_model.proj is not None:
            x = x @ self.feat_ext_model.proj
        return x, fts

    # TODO write description
    def forward(self, y_hat, y):
        """

        Args:
            y_hat (torch.Tensor):
            y (torch.Tensor):

        Returns:

        """
        _, y_fts = self.extract_visual(self.feat_ext_transform(y))
        _, y_hat_fts = self.extract_visual(self.feat_ext_transform(y_hat))

        att_loss = 0.0
        for ix in range(0, y_fts.shape[1]):
            tmp_loss = self.l1_loss(y_fts[:, ix].float(),
                                    y_hat_fts[:, ix].float())
            att_loss += tmp_loss

        return att_loss
