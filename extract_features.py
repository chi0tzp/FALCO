import sys
import os
import os.path as osp
import argparse
import torch
import clip
import open_clip
from torch.utils import data
from torchvision import transforms
from lib import CelebAHQ, LFW, FARL_PRETRAIN_MODEL, ArcFace, DECA_model, calculate_shapemodel
from tqdm import tqdm


def main():
    """Extract features for the images of a given real dataset in the CLIP [1] and/or OpenCLIP [X] and/or FaRL [2]
     and/or DINO [3] and/or DINOv2 [Y] ArcFace [4] and/or DECA [5] feature spaces.

    Options:
        -v, --verbose  : set verbose mode on
        --dataset      : choose dataset (see lib/config.py:DATASETS.keys())
        --dataset-root : choose dataset root directory (if none is given, lib/config.py:DATASETS[args.dataset] will be
                         used)
        --batch-size   : set batch size
        --no-clip      : do NOT extract CLIP features
        --no-openclip  : do NOT extract OpenCLIP features
        --no-farl      : do NOT extract FaRL features
        --no-dino      : do NOT extract DINO features
        --no-dinov2    : do NOT extract DINOv2 features
        --no-arcface   : do NOT extract ArcFace features
        --no-deca      : do NOT extract DECA features
        --cuda         : use CUDA (default)
        --no-cuda      : do not use CUDA

    References:
        [1] Radford, Alec, et al. "Learning transferable visual models from natural language supervision."
            International Conference on Machine Learning. PMLR, 2021.
        [X] Cherti, Mehdi, et al. "Reproducible scaling laws for contrastive language-image learning." Proceedings of
            the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
        [2] Zheng, Yinglin, et al. "General Facial Representation Learning in a Visual-Linguistic Manner."
            Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
        [3] Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." Proceedings of the
            IEEE/CVF International Conference on Computer Vision. 2021.
        [Y] Oquab, Maxime, et al. "Dinov2: Learning robust visual features without supervision."
            arXiv preprint arXiv:2304.07193 (2023).
        [4] Deng, Jiankang, et al. "ArcFace: Additive angular margin loss for deep face recognition." Proceedings of
            the IEEE/CVF conference on computer vision and pattern recognition. 2019.
        [5] Yao Feng, Haiwen Feng, Michael J Black, and Timo Bolkart. Learning an animatable detailed 3d face model from
            in-the-wild images. ACM Transactions on Graphics (TOG), 2021

    """
    parser = argparse.ArgumentParser(
        description="Real dataset feature extraction in the CLIP/OpenCLIP/FaRL/DINO/DINOv2/ArcFace/DECA spaces.")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--dataset', type=str, required=True, choices=('celebahq', 'lfw'),
                        help="choose real dataset")
    parser.add_argument('--dataset-root', type=str, help="set dataset root directory")
    parser.add_argument('--batch-size', type=int, default=128, help="set batch size")
    parser.add_argument('--no-clip', action='store_true', help="do NOT extract CLIP features")
    parser.add_argument('--no-openclip', action='store_true', help="do NOT extract OpenCLIP features")
    parser.add_argument('--no-farl', action='store_true', help="do NOT extract FaRL features")
    parser.add_argument('--no-dino', action='store_true', help="do NOT extract DINO features")
    parser.add_argument('--no-dinov2', action='store_true', help="do NOT extract DINOv2 features")
    parser.add_argument('--no-arcface', action='store_true', help="do NOT extract ArcFace features")
    parser.add_argument('--no-deca', action='store_true', help="do NOT extract DECA features")
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
    ##                                         [ Output Features' Directory ]                                         ##
    ##                                                                                                                ##
    ####################################################################################################################
    out_dir = osp.join(args.dataset_root, 'features')
    if args.verbose:
        print("#. Create dir for storing {} features...".format(args.dataset))
        print("  \\__{}".format(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                     [ Pre-trained CLIP / OpenCLIP / FaRL / DINO / DINOv2 / ArcFace / DECA ]                    ##
    ##                                                                                                                ##
    ####################################################################################################################
    # === CLIP ===
    clip_model = None
    clip_img_transform = None
    clip_features_file = osp.join(out_dir, 'clip_features.pt')
    if osp.exists(clip_features_file):
        args.no_clip = True

    if not args.no_clip:
        if args.verbose:
            print("#. Build pre-trained CLIP model...")

        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        clip_model.float()
        clip_model.eval()

        clip_img_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                 transforms.CenterCrop(224),
                                                 transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                      (0.26862954, 0.26130258, 0.27577711))])

    # === OpenCLIP ===
    openclip_model = None
    openclip_img_transform = None
    openclip_features_file = osp.join(out_dir, 'openclip_features.pt')
    if osp.exists(openclip_features_file):
        args.no_openclip = True

    if not args.no_openclip:
        if args.verbose:
            print("#. Build pre-trained OpenCLIP model...")
        openclip_model, _, _ = open_clip.create_model_and_transforms(model_name='ViT-B-32',
                                                                     pretrained='laion2b_s34b_b79k')
        openclip_model.eval().to(device)
        openclip_img_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                     transforms.CenterCrop(224),
                                                     transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                          (0.26862954, 0.26130258, 0.27577711))])

    # === FaRL ===
    farl_model = None
    farl_img_transform = None
    farl_features_file = osp.join(out_dir, 'farl_features.pt')
    if osp.exists(farl_features_file):
        args.no_farl = True

    if not args.no_farl:
        if args.verbose:
            print("#. Build pre-trained FaRL model...")

        farl_model, _ = clip.load("ViT-B/16", device=device, jit=False)
        farl_state = torch.load(osp.join('models', 'pretrained', 'farl', FARL_PRETRAIN_MODEL))
        farl_model.load_state_dict(farl_state["state_dict"], strict=False)
        farl_model.eval()
        farl_model.float()

        farl_img_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                 transforms.CenterCrop(224),
                                                 transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                      (0.26862954, 0.26130258, 0.27577711))])

    # === DINO ===
    dino_model = None
    dino_img_transform = None
    dino_features_file = osp.join(out_dir, 'dino_features.pt')
    if osp.exists(dino_features_file):
        args.no_dino = True

    if not args.no_dino:
        # =================================
        # Available pre-trained DINO models
        # =================================
        # dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        # dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        # dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        # dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        # dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p16')
        # dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_small_12_p8')
        # dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p16')
        # dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')
        # dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        if args.verbose:
            print("#. Build pre-trained DINO model...")

        dino_model = torch.hub.load("facebookresearch/dino:main", 'dino_vitb16')
        dino_model.eval().to(device)
        dino_model.float()

        dino_img_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                 transforms.CenterCrop(224),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # === DINOv2 ===
    # TODO: Address the following warnings:
    #   Using cache found in /home/chi/.cache/torch/hub/facebookresearch_dinov2_main
    #   /home/chi/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:43: UserWarning: xFormers is available (SwiGLU)
    #   warnings.warn("xFormers is available (SwiGLU)")
    #   /home/chi/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:27: UserWarning: xFormers is available (Attention)
    #   warnings.warn("xFormers is available (Attention)")
    #   /home/chi/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:33: UserWarning: xFormers is available (Block)
    #   warnings.warn("xFormers is available (Block)")
    dinov2_model = None
    dinov2_img_transform = None
    dinov2_features_file = osp.join(out_dir, 'dinov2_features.pt')
    if osp.exists(dinov2_features_file):
        args.no_dinov2 = True

    if not args.no_dinov2:
        if args.verbose:
            print("#. Build pre-trained DINOv2 model...")

        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
        dinov2_model.eval().to(device)
        dinov2_model.float()

        dinov2_img_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    # === ArcFace ===
    arcface_model = None
    arcface_img_transform = None
    arcface_features_file = osp.join(out_dir, 'arcface_features.pt')
    if osp.exists(arcface_features_file):
        args.no_arcface = True

    if not args.no_arcface:
        if args.verbose:
            print("#. Build pre-trained ArcFace model...")

        arcface_model = ArcFace()
        arcface_model.eval().to(device)
        arcface_model.float()

        arcface_img_transform = transforms.Compose([transforms.Resize(256, antialias=True),
                                                    transforms.CenterCrop(256)])

    # === DECA ===
    deca_model = None
    deca_img_transform = None
    deca_landmarks_file = osp.join(out_dir, 'deca_landmarks.pt')
    deca_angles_file = osp.join(out_dir, 'deca_angles.pt')
    if osp.exists(deca_landmarks_file) and osp.exists(deca_angles_file):
        args.no_deca = True

    if not args.no_deca:
        if args.verbose:
            print("#. Build pre-trained DECA model...")

        deca_model = DECA_model(device=device)
        deca_img_transform = transforms.Compose([transforms.Resize((256, 256)),
                                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    if (args.no_clip and args.no_openclip and args.no_farl and args.no_dino and args.no_dinov2 and args.no_arcface and
            args.dec):
        print("#. All required features have already been calculated and stored under {}.".format(out_dir))
        return

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                [ Data Loader ]                                                 ##
    ##                                                                                                                ##
    ####################################################################################################################
    if args.verbose:
        print("#. Load {} dataset...".format(args.dataset))

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
        dataset = LFW(root_dir=args.dataset_root, subset='train+val+test')
        dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                             [ Feature Extraction ]                                             ##
    ##                                                                                                                ##
    ####################################################################################################################
    # Process images
    img_filenames = []
    clip_features = []
    openclip_features = []
    farl_features = []
    dino_features = []
    dinov2_features = []
    arcface_features = []
    deca_landmarks = []
    deca_angles = []
    for i_batch, data_batch in enumerate(
            tqdm(dataloader, desc="#. Process {} images".format(args.dataset) if args.verbose else '')):

        # Get batch images' names
        img_orig_id = []
        for f in data_batch[2]:
            img_orig_id.append(osp.basename(f))
        img_filenames.extend(list(img_orig_id))

        # Calculate CLIP features
        if not args.no_clip:
            with torch.no_grad():
                img_feat = clip_model.encode_image(clip_img_transform(data_batch[0]).to(device))
            clip_features.append(img_feat.cpu())

        if not args.no_openclip:
            with torch.no_grad():
                img_feat = openclip_model.encode_image(openclip_img_transform(data_batch[0]).to(device))
            openclip_features.append(img_feat.cpu())

        # Calculate FaRL features
        if not args.no_farl:
            with torch.no_grad():
                img_feat = farl_model.encode_image(farl_img_transform(data_batch[0]).to(device))
            farl_features.append(img_feat.cpu())

        # Calculate DINO features
        if not args.no_dino:
            with torch.no_grad():
                img_feat = dino_model(dino_img_transform(data_batch[0]).to(device))
            dino_features.append(img_feat.cpu())

        # Calculate DINOv2 features
        if not args.no_dinov2:
            with torch.no_grad():
                img_feat = dinov2_model(dinov2_img_transform(data_batch[0]).to(device))
            dinov2_features.append(img_feat.cpu())

        # Calculate ArcFace features
        if not args.no_arcface:
            with torch.no_grad():
                img_feat = arcface_model(arcface_img_transform(data_batch[0]).to(device))
            arcface_features.append(img_feat.cpu())

        # Calculate DECA (landmarks + Euler angles) features
        if not args.no_deca:
            with torch.no_grad():
                landmarks2d, angles = calculate_shapemodel(deca_model=deca_model,
                                                           images=deca_img_transform(data_batch[0]).to(device))
            deca_landmarks.append(landmarks2d.reshape(landmarks2d.shape[0], -1).cpu())
            deca_angles.append(angles.cpu())

    # Save dataset images' filenames
    img_filenames_file = osp.join(out_dir, 'image_filenames.txt')
    if args.verbose:
        print("#. Save image filenames list @ {}".format(img_filenames_file))
    with open(img_filenames_file, 'w') as f:
        for h in img_filenames:
            f.write(f"{h}\n")

    # Save features
    if args.verbose:
        print("#. Save features....")

    # Save CLIP features
    if not args.no_clip:
        clip_features = torch.cat(clip_features)
        if args.verbose:
            print("  \\__CLIP features : {}".format(clip_features.shape))
            print("  \\__Save @ {}".format(clip_features_file))
        torch.save(clip_features, clip_features_file)

    # Save OpenCLIP features
    if not args.no_openclip:
        openclip_features = torch.cat(openclip_features)
        if args.verbose:
            print("  \\__OpenCLIP features : {}".format(openclip_features.shape))
            print("  \\__Save @ {}".format(openclip_features_file))
        torch.save(openclip_features, openclip_features_file)

    # Save FaRL features
    if not args.no_farl:
        farl_features = torch.cat(farl_features)
        if args.verbose:
            print("  \\__FaRL features : {}".format(farl_features.shape))
            print("  \\__Save @ {}".format(farl_features_file))
        torch.save(farl_features, farl_features_file)

    # Save DINO features
    if not args.no_dino:
        dino_features = torch.cat(dino_features)
        if args.verbose:
            print("  \\__DINO features : {}".format(dino_features.shape))
            print("  \\__Save @ {}".format(dino_features_file))
        torch.save(dino_features, dino_features_file)

    # Save DINOv2 features
    if not args.no_dinov2:
        dinov2_features = torch.cat(dinov2_features)
        if args.verbose:
            print("  \\__DINOv2 features : {}".format(dinov2_features.shape))
            print("  \\__Save @ {}".format(dinov2_features_file))
        torch.save(dinov2_features, dinov2_features_file)

    # Save ArcFace features
    if not args.no_arcface:
        arcface_features = torch.cat(arcface_features)
        if args.verbose:
            print("  \\__ArcFace features : {}".format(arcface_features.shape))
            print("  \\__Save @ {}".format(arcface_features_file))
        torch.save(arcface_features, arcface_features_file)

    # Save DECA landmarks and angles
    if not args.no_deca:
        deca_landmarks = torch.cat(deca_landmarks)
        if args.verbose:
            print("  \\__DECA landmarks : {}".format(deca_landmarks.shape))
            print("  \\__Save @ {}".format(deca_landmarks_file))
        torch.save(deca_landmarks, deca_landmarks_file)

        deca_angles = torch.cat(deca_angles)
        if args.verbose:
            print("  \\__DECA angles : {}".format(deca_angles.shape))
            print("  \\__Save @ {}".format(deca_angles_file))
        torch.save(deca_angles, deca_angles_file)


if __name__ == '__main__':
    main()
