import os
import os.path as osp
import argparse
import torch
from torch.utils import data
from torchvision import transforms
from lib import DATASETS, CelebAHQ, DECA_model, calculate_shapemodel
from tqdm import tqdm


def main():
    """TODO: Extract features for the images of a given real dataset in the CLIP [1] and/or OpenCLIP [X] and/or FaRL [2]
     and/or DINO [3] and/or DINOv2 [Y] ArcFace [4] and/or DECA [5] feature spaces.

    Options:
        -v, --verbose  : set verbose mode on
        --dataset      : choose dataset (see lib/config.py:DATASETS.keys())
        --dataset-root : choose dataset root directory (if none is given, lib/config.py:DATASETS[args.dataset] will be
                         used)
        --batch-size   : set batch size
        --cuda         : use CUDA (default)
        --no-cuda      : do not use CUDA

    References:
        [1] Yao Feng, Haiwen Feng, Michael J Black, and Timo Bolkart. Learning an animatable detailed 3d face model from
            in-the-wild images. ACM Transactions on Graphics (TOG), 2021

    """
    parser = argparse.ArgumentParser(
        description="Real dataset feature extraction in the CLIP/OpenCLIP/FaRL/DINO/DINOv2/ArcFace/DECA spaces.")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys(), help="choose real dataset")
    parser.add_argument('--dataset-root', type=str, help="set dataset root directory")
    parser.add_argument('--batch-size', type=int, default=128, help="set batch size")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)

    # Parse given arguments
    args = parser.parse_args()

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                    [ CUDA ]                                                    ##
    ##                                                                                                                ##
    ####################################################################################################################
    use_cuda = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
    # Set device
    device = 'cuda' if use_cuda else 'cpu'

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                    [ Output Landmarks and Angles Directory ]                                   ##
    ##                                                                                                                ##
    ####################################################################################################################
    landmarks_dir = osp.join(args.dataset_root, 'landmarks')
    angles_dir = osp.join(args.dataset_root, 'angles')
    if args.verbose:
        print("#. Create dir for storing {} landmarks and angles (DECA)...".format(args.dataset))
        print("  \\__{}".format(landmarks_dir))
        print("  \\__{}".format(angles_dir))
    os.makedirs(landmarks_dir, exist_ok=True)
    os.makedirs(angles_dir, exist_ok=True)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                              [ Pre-trained DECA ]                                              ##
    ##                                                                                                                ##
    ####################################################################################################################
    if args.verbose:
        print("#. Build pre-trained DECA model...")

    deca_model = DECA_model(device=device)
    deca_img_transform = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                [ Data Loader ]                                                 ##
    ##                                                                                                                ##
    ####################################################################################################################
    if args.verbose:
        print("#. Load {} dataset...".format(args.dataset))

    if args.dataset_root is None:
        args.dataset_root = DATASETS[args.dataset]

    dataloader = None
    ####################################################################################################################
    ##                                                 [ CelebA-HQ ]                                                  ##
    ####################################################################################################################
    if args.dataset == 'celebahq':
        dataset = CelebAHQ(root_dir=args.dataset_root, subset='train+val+test')
        dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    ####################################################################################################################
    ##                                                    [ LFW ]                                                     ##
    ####################################################################################################################
    elif args.dataset == 'lfw':
        raise NotImplementedError

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                 [ Pose Estimation (landmarks + Euler angles) ]                                 ##
    ##                                                                                                                ##
    ####################################################################################################################
    # Process images
    for i_batch, data_batch in enumerate(
            tqdm(dataloader, desc="#. Process {} images".format(args.dataset) if args.verbose else '')):

        # Get batch images' names
        img_orig_id = [osp.basename(f) for f in data_batch[2]]

        # Calculate landmarks and Euler angles (DECA)
        with torch.no_grad():
            landmarks2d, angles = calculate_shapemodel(deca_model=deca_model,
                                                       images=deca_img_transform(data_batch[0]).to(device))

        # Save
        for t in range(landmarks2d.shape[0]):
            torch.save(landmarks2d[t].T.unsqueeze(0).cpu(),
                       osp.join(landmarks_dir, '{}.pt'.format(img_orig_id[t].split('.')[0])))
            torch.save(angles[t].unsqueeze(0).cpu(),
                       osp.join(angles_dir, '{}.pt'.format(img_orig_id[t].split('.')[0])))


if __name__ == '__main__':
    main()
