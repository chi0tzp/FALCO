import os
import os.path as osp
import argparse
import json
import clip
import torch
from hashlib import sha1
from lib import GENFORCE_MODELS, FARL_PRETRAIN_MODEL, ArcFace, tensor2image
from torchvision import transforms
import shutil
from tqdm import tqdm
from models.load_generator import load_generator


def main():
    """
    A script for generating a fake image dataset by sampling from a pre-trained StyleGAN2 generator. The generated
    images, along with the corresponding W+ latent codes and their representations in the CLIP [1] and/or FaRL [2]
    and/or DINO [3] and/or ArcFace [4] feature spaces will be stored under `datasets/`.

    Options:
        -v, --verbose : set verbose mode on
        --gan         : set GAN generator (see GENFORCE_MODELS in lib/config.py)
        --truncation  : set W-space truncation parameter (default: 0.7)
        --num-samples : set the number of latent codes to sample for generating images (default: 60000)
        --no-clip     : do NOT extract CLIP features
        --no-farl     : do NOT extract FaRL features
        --no-dino     : do NOT extract DINO features
        --no-arcface  : do NOT extract ArcFace features
        --cuda        : use CUDA (default)
        --no-cuda     : do not use CUDA

    References:
        [1] Radford, Alec, et al. "Learning transferable visual models from natural language supervision."
            International Conference on Machine Learning. PMLR, 2021.
        [2] Zheng, Yinglin, et al. "General Facial Representation Learning in a Visual-Linguistic Manner."
            Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
        [3] Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." Proceedings of the
            IEEE/CVF International Conference on Computer Vision. 2021.
        [4] Deng, Jiankang, et al. "ArcFace: Additive angular margin loss for deep face recognition." Proceedings of
            the IEEE/CVF conference on computer vision and pattern recognition. 2019.

    """
    parser = argparse.ArgumentParser(description="Create a fake image dataset using a pre-trained GAN generator.")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--gan', type=str, default='stylegan2_ffhq1024', choices=GENFORCE_MODELS.keys(),
                        help='pre-trained GAN generator')
    parser.add_argument('--truncation', type=float, default=0.7, help="W-space truncation parameter")
    parser.add_argument('--num-samples', type=int, default=100, help="number of latent codes to sample")
    parser.add_argument('--no-clip', action='store_true', help="do NOT extract CLIP features")
    parser.add_argument('--no-farl', action='store_true', help="do NOT extract FaRL features")
    parser.add_argument('--no-dino', action='store_true', help="do NOT extract DINO features")
    parser.add_argument('--no-arcface', action='store_true', help="do NOT extract ArcFace features")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)

    # Parse given arguments
    args = parser.parse_args()

    if args.verbose:
        print("#. Generate fake dataset...")
        print("  \\__Pre-trained GAN         : {}".format(args.gan))
        print("  \\__Truncation              : {}".format(args.truncation))
        print("  \\__Number of samples       : {}".format(args.num_samples))
        print("  \\__Calculate CLIP features : {}".format(not args.no_clip))
        print("  \\__Calculate FaRL features : {}".format(not args.no_farl))
        print("  \\__Calculate DINO features : {}".format(not args.no_dino))
        print("  \\__Calculate ArcFace features : {}".format(not args.no_arcface))

    # Create output dir for generated fake dataset
    feat_config = ''
    if not args.no_clip:
        feat_config += '-CLIP'
    if not args.no_farl:
        feat_config += '-FaRL'
    if not args.no_dino:
        feat_config += '-DINO'
    if not args.no_arcface:
        feat_config += '-ArcFace'
    out_dir = osp.join('datasets', 'fake', 'fake_dataset_{}-{}-{}{}'.format(
        args.gan, args.truncation, args.num_samples, feat_config))

    if args.verbose:
        print("#. Create output directory for the generated fake dataset:")
        print("  \\__{}".format(out_dir))

    if osp.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save argument in json file
    with open(osp.join(out_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                    [ CUDA ]                                                    ##
    ##                                                                                                                ##
    ####################################################################################################################
    use_cuda = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                         [ Pre-trained GAN Generator ]                                          ##
    ##                                                                                                                ##
    ####################################################################################################################

    # Build GAN generator model and load with pre-trained weights
    if args.verbose:
        print("#. Build GAN generator model G and load with pre-trained weights...")
        print("  \\__GAN generator : {} (res: {})".format(args.gan, GENFORCE_MODELS[args.gan][1]))
        print("  \\__Pre-trained weights: {}".format(GENFORCE_MODELS[args.gan][0]))

    G = load_generator(model_name=args.gan, latent_is_s='stylegan' in args.gan, verbose=args.verbose).eval()

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                 [ Pre-trained CLIP / FaRL / DINO / ArcFace ]                                   ##
    ##                                                                                                                ##
    ####################################################################################################################
    clip_model = farl_model = dino_model = arcface_model = None
    clip_img_transform = farl_img_transform = dino_img_transform = arcface_img_transform = None

    # === CLIP ===
    if not args.no_clip:
        if args.verbose:
            print("#. Build pre-trained CLIP model...")

        clip_model, _ = clip.load("ViT-B/32", device='cuda' if use_cuda else 'cpu', jit=False)
        clip_model.float()
        clip_model.eval()

        clip_img_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                 transforms.CenterCrop(224),
                                                 transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                      (0.26862954, 0.26130258, 0.27577711))])
    # === FaRL ===
    if not args.no_farl:
        if args.verbose:
            print("#. Build pre-trained FaRL model...")

        farl_model, _ = clip.load("ViT-B/16", device='cuda' if use_cuda else 'cpu', jit=False)
        farl_state = torch.load(osp.join('models', 'pretrained', 'farl', FARL_PRETRAIN_MODEL))
        farl_model.load_state_dict(farl_state["state_dict"], strict=False)
        farl_model.eval()
        farl_model.float()

        farl_img_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                 transforms.CenterCrop(224),
                                                 transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                      (0.26862954, 0.26130258, 0.27577711))])
    # === DINO ===
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
        dino_model.eval()
        dino_model.float()
        # TODO: cuda()???

        dino_img_transform = transforms.Compose([transforms.Resize(224, antialias=True),
                                                 transforms.CenterCrop(224),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # === ArcFace ===
    if not args.no_arcface:
        if args.verbose:
            print("#. Build pre-trained ArcFace model...")

        arcface_model = ArcFace()
        arcface_model.eval()
        arcface_model.float()

        arcface_img_transform = transforms.Compose([transforms.Resize(256, antialias=True),
                                                    transforms.CenterCrop(256)])

    ####################################################################################################################
    ##                                                                                                                ##
    ##                           [ Latent Code Sampling / Generation / Feature Extraction ]                           ##
    ##                                                                                                                ##
    ####################################################################################################################

    # Latent codes sampling
    if args.verbose:
        print("#. Sample {} {}-dimensional latent codes (Z space)...".format(args.num_samples, G.dim_z))
    zs = torch.randn(args.num_samples, G.dim_z)

    if use_cuda:
        zs = zs.cuda()

    if args.verbose:
        print("#. Generate images...")
        print("  \\__{}".format(out_dir))

    # Iterate over given latent codes
    latent_code_hashes = []
    clip_features = []
    farl_features = []
    dino_features = []
    arcface_features = []
    for i in tqdm(range(args.num_samples)):
        # Un-squeeze current latent code in shape [1, dim] and create hash code for it
        z = zs[i, :].unsqueeze(0)
        latent_code_hash = sha1(z.cpu().numpy()).hexdigest()
        latent_code_hashes.append(latent_code_hash)

        # Create directory for current latent code
        latent_code_dir = osp.join(out_dir, '{}'.format(latent_code_hash))
        os.makedirs(latent_code_dir, exist_ok=True)

        # Get W+ latent codes from z code
        wp = G.get_w(z, truncation=args.truncation)

        # Get S latent codes from wp codes
        styles_dict = G.get_s(wp)

        # Generate image
        with torch.no_grad():
            img = G(styles_dict)

        # Calculate CLIP features
        if not args.no_clip:
            with torch.no_grad():
                img_feat = clip_model.encode_image(clip_img_transform(img))
            torch.save(img_feat.cpu(), osp.join(latent_code_dir, 'clip_features.pt'))
            clip_features.append(img_feat.cpu())

        # Calculate FaRL features
        if not args.no_farl:
            with torch.no_grad():
                img_feat = farl_model.encode_image(farl_img_transform(img))
            torch.save(img_feat.cpu(), osp.join(latent_code_dir, 'farl_features.pt'))
            farl_features.append(img_feat.cpu())

        # Calculate DINO features
        if not args.no_dino:
            with torch.no_grad():
                img_feat = dino_model(dino_img_transform(img))
            torch.save(img_feat.cpu(), osp.join(latent_code_dir, 'dino_features.pt'))
            dino_features.append(img_feat.cpu())

        # Calculate ArcFace features
        if not args.no_arcface:
            with torch.no_grad():
                img_feat = arcface_model(arcface_img_transform(img))
            torch.save(img_feat.cpu(), osp.join(latent_code_dir, 'arcface_features.pt'))
            arcface_features.append(img_feat.cpu())

        # Save image
        tensor2image(img.cpu(), adaptive=True).save(osp.join(latent_code_dir, 'image.jpg'),
                                                    "JPEG", quality=95, subsampling=0, progressive=True)

        # Save latent codes in W and S spaces
        torch.save(wp.cpu(), osp.join(latent_code_dir, 'latent_code_w+.pt'))
        torch.save(styles_dict, osp.join(latent_code_dir, 'latent_code_s.pt'))

    # Write latent codes hashes to file
    with open(osp.join(out_dir, 'latent_code_hashes.txt'), 'w') as f:
        for h in latent_code_hashes:
            f.write(f"{h}\n")

    # Save features
    if args.verbose:
        print("#. Save features....")

    # Save CLIP features
    if not args.no_clip:
        clip_features = torch.cat(clip_features)
        clip_features_file = osp.join(out_dir, 'clip_features.pt')
        if args.verbose:
            print("  \\__CLIP features : {}".format(clip_features.shape))
            print("  \\__Save @ {}".format(clip_features_file))
        torch.save(clip_features, clip_features_file)

    # Save FaRL features
    if not args.no_farl:
        farl_features = torch.cat(farl_features)
        farl_features_file = osp.join(out_dir, 'farl_features.pt')
        if args.verbose:
            print("  \\__FaRL features : {}".format(farl_features.shape))
            print("  \\__Save @ {}".format(farl_features_file))
        torch.save(farl_features, farl_features_file)

    # Save DINO features
    if not args.no_dino:
        dino_features = torch.cat(dino_features)
        dino_features_file = osp.join(out_dir, 'dino_features.pt')
        if args.verbose:
            print("  \\__DINO features : {}".format(dino_features.shape))
            print("  \\__Save @ {}".format(dino_features_file))
        torch.save(dino_features, dino_features_file)

    # Save ArcFace features
    if not args.no_arcface:
        arcface_features = torch.cat(arcface_features)
        arcface_features_file = osp.join(out_dir, 'arcface_features.pt')
        if args.verbose:
            print("  \\__ArcFace features : {}".format(arcface_features.shape))
            print("  \\__Save @ {}".format(arcface_features_file))
        torch.save(arcface_features, arcface_features_file)


if __name__ == '__main__':
    main()
