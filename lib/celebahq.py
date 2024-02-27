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
            fake_nn_map (str): fake NN map file (created by `create_fake_dataset.py`)
            inv (bool): get inverted images (created by `invert.py`)
            anon (bool): get anonymized images (create by `anonymize.py`)
            transform (torchvision.transforms.transforms.Compose): image transforms

        """
        # Get (real) dataset's root directory
        self.root_dir = root_dir

        # Get image transforms
        self.transform = transform

        ################################################################################################################
        ##                                                                                                            ##
        ##                                         [ Original (real) images ]                                         ##
        ##                                                                                                            ##
        ################################################################################################################
        self.data_dir = osp.join(self.root_dir, 'data')
        self.anno_dir = osp.join(self.root_dir, 'annotations')
        if subset not in ('train', 'val', 'test', 'train+val', 'train+val+test'):
            raise ValueError("Invalid dataset subset: {}".format(subset))
        else:
            self.subset = subset

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

        # Get subset images' filenames
        train_img_files = []
        val_img_files = []
        test_img_files = []
        self.orig_landmarks_available = []
        self.orig_angles_available = []
        for item in content_list:
            img_filename = item.split(' ')[0]
            img_label = int(item.split(' ')[1])
            if img_filename in self.celeba_to_celebahq_map:
                if img_label == 0:
                    train_img_files.append(
                        osp.join(self.data_dir, '{}.jpg'.format(self.celeba_to_celebahq_map[img_filename])))
                elif img_label == 1:
                    val_img_files.append(
                        osp.join(self.data_dir, '{}.jpg'.format(self.celeba_to_celebahq_map[img_filename])))
                elif img_label == 2:
                    test_img_files.append(
                        osp.join(self.data_dir, '{}.jpg'.format(self.celeba_to_celebahq_map[img_filename])))

                # Check landmark and angles files
                self.orig_landmarks_available.append(osp.exists(osp.join(
                    self.root_dir, 'landmarks', '{}.pt'.format(self.celeba_to_celebahq_map[img_filename]))))
                self.orig_angles_available.append(osp.exists(
                        osp.join(self.root_dir, 'angles', '{}.pt'.format(self.celeba_to_celebahq_map[img_filename]))))
        self.orig_landmarks_available = all(self.orig_landmarks_available)
        self.orig_angles_available = all(self.orig_angles_available)

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
            self.attributes.update({anno_item.split()[0]: [1 if int(a) == 1 else 0 for a in anno_item.split()[1:]]})

        ################################################################################################################
        ##                                                                                                            ##
        ##                                             [ Fake NN images ]                                             ##
        ##                                                                                                            ##
        ################################################################################################################
        self.fake_dataset_root = None
        self.nn_type = None
        self.nn_landmarks_available = False
        self.nn_angles_available = False
        self.nn_map_dict = {}
        if fake_nn_map is None:
            self.fake_nn_map = fake_nn_map
        else:
            if not osp.isfile(fake_nn_map):
                raise FileNotFoundError
            else:
                self.fake_nn_map = fake_nn_map
                self.fake_dataset_root = osp.dirname(fake_nn_map)
                self.nn_type = osp.basename(fake_nn_map).split('.')[0]
                with open(self.fake_nn_map) as f:
                    self.nn_map_dict = json.load(f)
                self.nn_landmarks_available = all(
                    [osp.exists(osp.join(self.fake_dataset_root, f[0], 'deca_landmarks.pt')) for f in
                     self.nn_map_dict.values()])
                self.nn_angles_available = all(
                    [osp.exists(osp.join(self.fake_dataset_root, f[0], 'deca_angles.pt')) for f in
                     self.nn_map_dict.values()])

        ################################################################################################################
        ##                                                                                                            ##
        ##                                          [ Inverted (e4e) images ]                                         ##
        ##                                                                                                            ##
        ################################################################################################################
        self.inv = inv
        self.inv_data_dir = osp.join(self.root_dir, 'inv', 'data')
        self.inv_codes_dir = osp.join(self.root_dir, 'inv', 'latent_codes')
        self.inv_landmarks_available = False
        self.inv_angles_available = False

        ################################################################################################################
        ##                                                                                                            ##
        ##                                            [ Anonymized images ]                                           ##
        ##                                                                                                            ##
        ################################################################################################################
        self.anon_landmarks_available = False
        self.anon_angles_available = False
        if anon is None:
            self.anon = anon
        else:
            if not osp.exists(anon):
                raise NotADirectoryError
            else:
                self.anon = anon
                self.anon_landmarks_available = False
                self.anon_angles_available = False

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
                    img_orig_landmarks ()
                    img_orig_angles ()
                    img_nn ()
                    img_nn_code ()
                    img_recon ()
                    img_recon_code ()

            If a fake NN map file is given (i.e., if `self.fake_nn_map` is not None ) and

        """
        ################################################################################################################
        ##                                                                                                            ##
        ##                                         [ Original (real) images ]                                         ##
        ##                                                                                                            ##
        ################################################################################################################
        img_orig_path = self.images[index]
        img_orig_basename = osp.basename(img_orig_path)
        img_orig = self.transform(Image.open(img_orig_path))
        img_orig_attr = torch.tensor(self.attributes[img_orig_basename], dtype=torch.int64)

        img_orig_landmarks = torch.zeros(2, 68)
        if self.orig_landmarks_available:
            img_orig_landmarks = torch.load(
                osp.join(self.root_dir, 'landmarks', '{}.pt'.format(img_orig_basename.split('.')[0]))).squeeze(0)

        img_orig_angles = torch.zeros(3)
        if self.orig_angles_available:
            img_orig_angles = torch.load(
                osp.join(self.root_dir, 'angles', '{}.pt'.format(img_orig_basename.split('.')[0]))).squeeze(0)

        ################################################################################################################
        ##                                                                                                            ##
        ##                                             [ Fake NN images ]                                             ##
        ##                                                                                                            ##
        ################################################################################################################
        img_nn = torch.zeros_like(img_orig)
        img_nn_code = torch.zeros((18, 512))
        if self.fake_nn_map is not None:
            img_nn = self.transform(
                Image.open(osp.join(self.fake_dataset_root, self.nn_map_dict[img_orig_basename][0], 'image.jpg')))
            img_nn_code = torch.load(osp.join(self.fake_dataset_root, self.nn_map_dict[img_orig_basename][0],
                                              'latent_code_w+.pt')).squeeze(0)

        img_nn_landmarks = torch.zeros(2, 68)
        if self.nn_landmarks_available:
            img_nn_landmarks = torch.load(
                osp.join(self.root_dir, 'landmarks', '{}.pt'.format(img_orig_basename.split('.')[0]))).squeeze(0)

        img_nn_angles = torch.zeros(3)
        if self.nn_angles_available:
            img_nn_angles = torch.load(
                osp.join(self.root_dir, 'angles', '{}.pt'.format(img_orig_basename.split('.')[0]))).squeeze(0)

        ################################################################################################################
        ##                                                                                                            ##
        ##                                          [ Inverted (e4e) images ]                                         ##
        ##                                                                                                            ##
        ################################################################################################################
        img_recon = torch.zeros_like(img_orig)
        img_recon_code = torch.zeros((18, 512))
        if self.inv:
            img_recon_path = osp.join(self.inv_data_dir, '{}_recon.{}'.format(img_orig_basename.split('.')[0],
                                                                              img_orig_basename.split('.')[1]))
            img_recon = self.transform(Image.open(img_recon_path))
            img_recon_code_path = osp.join(self.inv_codes_dir, '{}.pt'.format(img_orig_basename.split('.')[0]))
            img_recon_code = torch.load(img_recon_code_path)

        img_recon_landmarks = torch.zeros(2, 68)
        if self.inv_landmarks_available:
            pass

        img_recon_angles = torch.zeros(3)
        if self.inv_angles_available:
            pass

        ################################################################################################################
        ##                                                                                                            ##
        ##                                            [ Anonymized images ]                                           ##
        ##                                                                                                            ##
        ################################################################################################################
        img_anon = torch.zeros_like(img_orig)
        img_anon_code = torch.zeros((18, 512))
        if self.anon is not None:
            img_anon_path = osp.join(self.anon, 'data', '{}.{}'.format(img_orig_basename.split('.')[0],
                                                                       img_orig_basename.split('.')[1]))
            img_anon = self.transform(Image.open(img_anon_path))

            img_anon_code_path = osp.join(self.anon, 'latent_codes', '{}.pt'.format(img_orig_basename.split('.')[0]))
            img_anon_code = torch.load(img_anon_code_path)

        img_anon_landmarks = torch.zeros(2, 68)
        if self.anon_landmarks_available:
            img_anon_landmarks = torch.load(
                osp.join(self.root_dir, 'landmarks', '{}.pt'.format(img_orig_basename.split('.')[0]))).squeeze(0)

        img_anon_angles = torch.zeros(3)
        if self.anon_angles_available:
            img_anon_angles = torch.load(
                osp.join(self.root_dir, 'angles', '{}.pt'.format(img_orig_basename.split('.')[0]))).squeeze(0)

        # Build output list
        output = [img_orig, img_orig_attr, img_orig_path, img_orig_landmarks, img_orig_angles,  # Original (real) images
                  img_nn, img_nn_code, img_nn_landmarks, img_nn_angles,                         # Fake (NN) images
                  img_recon, img_recon_code, img_recon_landmarks, img_recon_angles,             # Recon. (e4e) images
                  img_anon, img_anon_code, img_anon_landmarks, img_anon_angles]                 # Anonymized images

        return output
