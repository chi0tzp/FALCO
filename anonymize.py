import argparse
import torch
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR
import os
import os.path as osp
from lib import DATASETS, CelebAHQ, DataParallelPassthrough, IDLoss, AttrLoss, LatentCode, tensor2image, anon_exp_dir
from models.load_generator import load_generator
from tqdm import tqdm
import json


def main():
    """Anonymize the images of a given real dataset.

    Options:
        -v, --verbose   : set verbose mode on
        --dataset       : choose dataset (see lib/config.py:DATASETS.keys())
        --dataset-root  : choose dataset root directory (if none is given, lib/config.py:DATASETS[args.dataset] will be
                          used)
        --subset        : choose dataset subset ('train', 'val', 'test', 'train+val', 'train+val+test'), if applicable
        --fake-nn-map   : choose fake NN map file (created by `pair_nn.py`)
        --latent-space  : choose StyleGAN2's latent space ('W+' or 'S')
        -m, --id-margin : set identity loss margin
        --epochs        : set number of training epochs
        --optim         : set optimizer ('sgd' or 'adam')
        --lr            : set (initial) learning rate
        --lr-milestones : set learning rate scheduler milestones (list of floats in (0.0, 1.0)
        --lr-gamma      : set learning rate decay parameter
        --lambda_id     : set identity loss weighting parameter
        --lambda_attr   : set attribute loss weighting parameter
        --cuda          : use CUDA (default)
        --no-cuda       : do not use CUDA

    References:
        [1] Tov, Omer, et al. "Designing an encoder for stylegan image manipulation."
            ACM Transactions on Graphics (TOG) 40.4 (2021): 1-14.


    """
    parser = argparse.ArgumentParser("Anonymization script")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys(), help="choose real dataset")
    parser.add_argument('--dataset-root', type=str, help="set dataset root directory")
    parser.add_argument('--subset', type=str, default='train+val+test',
                        choices=('train', 'val', 'test', 'train+val', 'train+val+test'), help="choose dataset's subset")
    parser.add_argument('--fake-nn-map', type=str, required=True, help="fake NN map file (created by `pair_nn.py`)")
    parser.add_argument('--latent-space', type=str, default='W+', choices=('W+', 'S'), help="StyleGAN2's latent space")
    parser.add_argument('-m', '--id-margin', type=float, default=0.0, help="identity loss margin")
    parser.add_argument('--epochs', type=int, default=50, help="Number of anonymization steps")
    parser.add_argument('--optim', type=str, default='adam', choices=('sgd', 'adam'),
                        help="set optimizer ('sgd' or 'adam')")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--lr-milestones', nargs="+", default=[0.75, 0.9], help="learning rate scheduler milestones")
    parser.add_argument('--lr-gamma', type=float, default=0.8, help="learning rate decay parameter")
    parser.add_argument('--lambda-id', type=float, default=10.0, help="Scaling parameter of the ID loss")
    parser.add_argument('--lambda-attr', type=float, default=0.1, help="Scaling parameter of the attribute loss")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)

    # Parse given arguments
    args = parser.parse_args()

    # Set real dataset root dir
    if args.dataset_root is None:
        args.dataset_root = DATASETS[args.dataset]

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                      [ Anonymized Dataset Directory  ]                                         ##
    ##                                                                                                                ##
    ####################################################################################################################
    args_dict = args.__dict__.copy()
    out_dir = anon_exp_dir(args_dict)
    if args.verbose:
        print("#. Create dir for storing the anonymized {} dataset...".format(args.dataset))
        print("  \\__{}".format(out_dir))

    # Save experiment's arguments
    del args_dict["lr_milestones"]
    with open(osp.join(out_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args_dict, args_json_file)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                    [ CUDA ]                                                    ##
    ##                                                                                                                ##
    ####################################################################################################################
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
    device = 'cuda' if use_cuda else 'cpu'

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                         [ Pre-trained GAN Generator ]                                          ##
    ##                                                                                                                ##
    ####################################################################################################################

    # Build GAN generator model and load with pre-trained weights
    if args.verbose:
        print("#. Build StyleGAN2 generator model G and load with pre-trained weights...")

    G = load_generator(model_name='stylegan2_ffhq1024', latent_is_w=True, verbose=args.verbose).eval().to(device)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                [ Data Loader ]                                                 ##
    ##                                                                                                                ##
    ####################################################################################################################
    if args.verbose:
        print("#. Load {} dataset...".format(args.dataset))

    dataloader = None
    out_data_dir = None
    out_code_dir = None
    ####################################################################################################################
    ##                                                   [ CelebA ]                                                   ##
    ####################################################################################################################
    if args.dataset == 'celeba':
        raise NotImplementedError

    ####################################################################################################################
    ##                                                 [ CelebA-HQ ]                                                  ##
    ####################################################################################################################
    elif args.dataset == 'celebahq':
        dataset = CelebAHQ(root_dir=args.dataset_root,
                           subset=args.subset,
                           fake_nn_map=args.fake_nn_map,
                           inv=True)
        dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        # Create output directory to save images
        out_data_dir = osp.join(out_dir, 'data')
        os.makedirs(out_data_dir, exist_ok=True)

        # Create output directory to save latent codes
        out_code_dir = osp.join(out_dir, 'latent_codes')
        os.makedirs(out_code_dir, exist_ok=True)

    ####################################################################################################################
    ##                                                    [ LFW ]                                                     ##
    ####################################################################################################################
    elif args.dataset == 'lfw':
        raise NotImplementedError

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                   [ Losses ]                                                   ##
    ##                                                                                                                ##
    ####################################################################################################################
    id_criterion = IDLoss(id_margin=args.id_margin).eval().to(device)
    attribute_loss = AttrLoss(feat_ext='farl').eval().to(device)

    # Parallelize GAN's generator G
    if multi_gpu:
        G = DataParallelPassthrough(G)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                               [ Anonymization ]                                                ##
    ##                                                                                                                ##
    ####################################################################################################################
    for data_idx, data_ in enumerate(
            tqdm(dataloader, desc="#. Invert {} images".format(args.dataset) if args.verbose else '')):

        # Get data
        img_orig = data_[0]
        img_orig_id = int(osp.basename(data_[2][0]).split('.')[0])
        img_nn_code = data_[4]
        img_recon_code = data_[6]

        # Build anonymization latent code
        latent_code = LatentCode(latent_code_real=img_recon_code, latent_code_fake_nn=img_nn_code, img_id=img_orig_id,
                                 out_code_dir=out_code_dir, latent_space='W+')
        latent_code.to(device)

        # Count trainable parameters
        # latent_code_trainable_parameters = sum(p.numel() for p in latent_code.parameters() if p.requires_grad)
        # print("latent_code_trainable_parameters: {}".format(latent_code_trainable_parameters))

        # Check whether anonymization latent code has already been optimized -- if so, continue with the next one
        if not latent_code.do_optim():
            continue

        # Build optimizer
        optimizer = None
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(params=latent_code.parameters(), lr=args.lr)
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(params=latent_code.parameters(), lr=args.lr)

        # Set learning rate scheduler
        lr_scheduler = MultiStepLR(optimizer=optimizer,
                                   milestones=[int(m * args.epochs) for m in args.lr_milestones],
                                   gamma=args.lr_gamma)

        # Zero out gradients
        G.zero_grad()
        id_criterion.zero_grad()
        attribute_loss.zero_grad()

        # Training (anonymization) loop for the current batch of images / latent codes
        for epoch in range(args.epochs):
            # Clear gradients wrt parameters
            optimizer.zero_grad()

            # Generate anonymized image
            img_anon = G(latent_code())

            # Calculate identity and attribute preservation losses
            id_loss = id_criterion(img_anon.to(device), img_orig.to(device))
            att_loss = attribute_loss(img_orig.to(device), img_anon.to(device))

            # Calculate total loss
            loss = args.lambda_id * id_loss + args.lambda_attr * att_loss

            # Back-propagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # Store optimized anonymization latent codes
        latent_code.save()

        # Generate and save anonymized image
        with torch.no_grad():
            anonymized_image = G(latent_code())
        tensor2image(anonymized_image.cpu(), adaptive=True).save(osp.join(out_data_dir, '{}.jpg'.format(img_orig_id)),
                                                                 "JPEG", quality=75, subsampling=0, progressive=True)


if __name__ == '__main__':
    main()
