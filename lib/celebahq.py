import sys
import torch
import os.path as osp
import json
from torchvision import transforms
from torch.utils import data
from PIL import Image


class CelebAHQ(data.Dataset):
    def __init__(self, root_dir,
                 subset='train',
                 fake_nn_map=None,
                 inv=False,
                 anon=None,
                 transform=transforms.Compose([transforms.ToTensor()])):
        """CelebA-HQ dataset class.

        Args:
            root_dir (str): dataset root directory
            subset (str): dataset subset ('train', 'val', 'test', 'train+val', or 'train+val+test')
            fake_nn_map (str): TODO: +++
            inv (bool): TODO: +++
            anon (bool): TODO: +++
            transform (torchvision.transforms.transforms.Compose): image transforms

        """
        self.root_dir = root_dir
        self.data_dir = osp.join(self.root_dir, 'data')
        self.anno_dir = osp.join(self.root_dir, 'annotations')
        if subset not in ('train', 'val', 'test', 'train+val', 'train+val+test'):
            raise ValueError("Invalid dataset subset: {}".format(subset))
        else:
            self.subset = subset

        # TODO: add comment
        self.fake_dataset_root = None
        self.nn_type = None

        if fake_nn_map is None:
            self.fake_nn_map = fake_nn_map
        else:
            if not osp.isfile(fake_nn_map):
                raise FileNotFoundError
            else:
                self.fake_nn_map = fake_nn_map
                self.fake_dataset_root = osp.dirname(fake_nn_map)
                self.nn_type = osp.basename(fake_nn_map).split('.')[0]

        # TODO: add comment
        self.inv = inv
        self.inv_data_dir = osp.join('datasets', 'inv', 'celebahq', 'data')
        self.inv_codes_dir = osp.join('datasets', 'inv', 'celebahq', 'latent_codes')

        # TODO: add comment
        if anon is None:
            self.anon = anon
        else:
            if not osp.exists(anon):
                raise NotADirectoryError
            else:
                self.anon = anon

        # TODO: add comment
        self.transform = transform

        # Build CelebA to CelebA-HQ map
        self.celebahq_to_celeba_map_file = osp.join(self.anno_dir, 'CelebA-HQ-to-CelebA-mapping.txt')
        self.celeba_to_celebahq_map = dict()
        with open(self.celebahq_to_celeba_map_file) as f:
            content_list = f.readlines()
        content_list = [x.strip() for x in content_list]
        for item in content_list[1:]:
            self.celeba_to_celebahq_map.update({item.split()[2]: int(item.split()[0])})

        # Get CelebA train/val/test split
        self.list_eval_partition = osp.join(self.anno_dir, 'list_eval_partition.txt')
        with open(self.list_eval_partition) as f:
            content_list = f.readlines()
        content_list = [x.strip() for x in content_list]

        # TODO: add comment
        train_img_files = []
        val_img_files = []
        test_img_files = []
        for item in content_list:
            img_filename = item.split(' ')[0]
            img_label = int(item.split(' ')[1])
            if img_filename in self.celeba_to_celebahq_map:
                if img_label == 0:
                    train_img_files.append(osp.join(self.data_dir,
                                                    '{}.jpg'.format(self.celeba_to_celebahq_map[img_filename])))
                elif img_label == 1:
                    val_img_files.append(osp.join(self.data_dir,
                                                  '{}.jpg'.format(self.celeba_to_celebahq_map[img_filename])))
                elif img_label == 2:
                    test_img_files.append(osp.join(self.data_dir,
                                                   '{}.jpg'.format(self.celeba_to_celebahq_map[img_filename])))

        # TODO: add comment
        self.images = []
        if 'train' in self.subset:
            self.images.extend(train_img_files)
        if 'val' in self.subset:
            self.images.extend(val_img_files)
        if 'test' in self.subset:
            self.images.extend(test_img_files)

        # Get attribute annotations
        self.annotation_file = osp.join(self.anno_dir, 'CelebAMask-HQ-attribute-anno.txt')
        with open(self.annotation_file) as f:
            anno_list = f.readlines()
        anno_list = [x.strip() for x in anno_list]

        self.attributes = dict()
        for anno_item in anno_list[2:]:
            self.attributes.update({anno_item.split()[0]:
                                    [1 if int(a) == 1 else 0 for a in anno_item.split()[1:]]})

        # TODO: add comment
        self.nn_map_dict = {}
        if self.fake_nn_map is not None:
            with open(self.fake_nn_map) as f:
                self.nn_map_dict = json.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """TODO: +++ CelebA-HQ getitem function

        Args:
            index ():

        Returns:
            output (list): The following list

                    [img_orig, img_orig_attr, img_orig_path, img_nn, img_nn_code, img_recon, img_recon_code],

                where
                    img_orig ()
                    img_orig_attr ()
                    img_orig_path ()
                    img_nn ()
                    img_nn_code ()
                    img_recon ()
                    img_recon_code ()

            If a fake NN map file is given (i.e., if `self.fake_nn_map` is not None ) and

        """
        # Get original image
        img_orig_path = self.images[index]
        img_orig_basename = osp.basename(img_orig_path)
        img_orig = self.transform(Image.open(img_orig_path))

        # Get attributes of the original images
        img_orig_attr = torch.tensor(self.attributes[img_orig_basename], dtype=torch.int64)

        # TODO: add comment
        img_nn = torch.zeros_like(img_orig)
        # REVIEW: W+ code
        img_nn_code = torch.zeros((18, 512))
        if self.fake_nn_map is not None:
            img_nn = self.transform(
                Image.open(osp.join(self.fake_dataset_root, self.nn_map_dict[img_orig_basename], 'image.jpg')))
            img_nn_code = torch.load(osp.join(self.fake_dataset_root, self.nn_map_dict[img_orig_basename],
                                              'latent_code_w+.pt')).squeeze(0)
            # TODO::Christos img_nn_code needs to be squeezed like above, check the way it is stored and fix this.

        # TODO: add comment
        img_recon = torch.zeros_like(img_orig)
        # REVIEW: W+ code
        img_recon_code = torch.zeros((18, 512))
        if self.inv:
            img_recon_path = osp.join(self.inv_data_dir, '{}_recon.{}'.format(img_orig_basename.split('.')[0],
                                                                              img_orig_basename.split('.')[1]))
            img_recon = self.transform(Image.open(img_recon_path))

            img_recon_code_path = osp.join(self.inv_codes_dir, '{}.pt'.format(img_orig_basename.split('.')[0]))
            img_recon_code = torch.load(img_recon_code_path)

        # TODO: add comment
        img_anon = torch.zeros_like(img_orig)
        # REVIEW: W+ code
        img_anon_code = torch.zeros((18, 512))
        if self.anon is not None:
            img_anon_path = osp.join(self.anon, 'data', '{}.{}'.format(img_orig_basename.split('.')[0],
                                                                       img_orig_basename.split('.')[1]))
            img_anon = self.transform(Image.open(img_anon_path))

            img_anon_code_path = osp.join(self.anon, 'latent_codes', '{}.pt'.format(img_orig_basename.split('.')[0]))
            img_anon_code = torch.load(img_anon_code_path)

        # Build output list
        output = [img_orig, img_orig_attr, img_orig_path, img_nn, img_nn_code, img_recon, img_recon_code, img_anon,
                  img_anon_code]

        return output
