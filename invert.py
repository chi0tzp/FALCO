import os
import os.path as osp
import argparse
import torch
from torch.utils import data
from torchvision import transforms
from models.psp import pSp
from lib import DATASETS, CelebAHQ, FaceAligner
from tqdm import tqdm
import cv2
from PIL import Image
from models.load_generator import load_generator
from lib import tensor2image


def get_img_id(img_file):
    return osp.basename(img_file).split('.')[0]


def save_img(img_file, img):
    cv2.imwrite(img_file, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))


def get_latents(net, x):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)

    return codes

# TODO:
#   /home/lab/code/FALCO_cvpr23/falco-venv/lib/python3.12/site-packages/torch/utils/cpp_extension.py:1967: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
#   If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
#   warnings.warn(
#   /home/lab/code/FALCO_cvpr23/falco-venv/lib/python3.12/site-packages/torch/utils/cpp_extension.py:1967: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
#   If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
#   warnings.warn(


def main():
    """Invert the images of a given real dataset using the e4e [1] encoder.

    Options:
        -v, --verbose                    : set verbose mode on
        --dataset                        : choose dataset (see lib/config.py:DATASETS.keys())
        --dataset-root                   : choose dataset root directory (if none is given,
                                           lib/config.py:DATASETS[args.dataset] will be used)
        --batch-size                     : set generation batch size
        --save-aligned-images            : save cropped and aligned images (default)
        --dont-save-aligned-images       : do not save cropped and aligned images
        --save-reconstructed-images      : save e4e reconstructed images (default)
        --dont-save-reconstructed-images : do not save e4e reconstructed images
        --cuda                           : use CUDA (default)
        --no-cuda                        : do not use CUDA

    References:
        [1] Tov, Omer, et al. "Designing an encoder for stylegan image manipulation."
            ACM Transactions on Graphics (TOG) 40.4 (2021): 1-14.

    """
    parser = argparse.ArgumentParser(description="Real dataset GAN inversion script.")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys(), help="choose real dataset")
    parser.add_argument('--dataset-root', type=str, help="set dataset root directory")
    parser.add_argument('--batch-size', type=int, default=4, help="set generation batch size")
    parser.add_argument('--save-aligned-images', dest='save_aligned_images', action='store_true',
                        help="save aligned images")
    parser.add_argument('--dont-save-aligned-images', dest='save_aligned_images', action='store_false',
                        help="do NOT save aligned images")
    parser.set_defaults(save_aligned_images=True)
    parser.add_argument('--save-reconstructed-images', dest='save_reconstructed_images', action='store_true',
                        help="save reconstructed images")
    parser.add_argument('--dont-save-reconstructed-images', dest='save_reconstructed_images', action='store_false',
                        help="do NOT save reconstructed images")
    parser.set_defaults(save_reconstructed_images=True)
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
    device = 'cuda' if use_cuda else 'cpu'

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                       [ Inverted Dataset Directory  ]                                          ##
    ##                                                                                                                ##
    ####################################################################################################################
    out_dir = osp.join('datasets', 'inv', '{}'.format(args.dataset))
    if args.verbose:
        print("#. Create dir for storing the inverted {} dataset...".format(args.dataset))
        print("  \\__{}".format(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                              [ Face Alignment ]                                                ##
    ##                                                                                                                ##
    ####################################################################################################################
    # Build landmark-based face aligner (required by e4e inversion)
    face_aligner = FaceAligner(device=device)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                          [ Pre-trained e4e Encoder ]                                           ##
    ##                                                                                                                ##
    ####################################################################################################################
    if args.verbose:
        print("#. Build pre-trained e4e model...")

    e4e_checkpoint_path = osp.join('models', 'pretrained', 'e4e', 'e4e_ffhq_encode.pt')
    e4e_checkpoint = torch.load(e4e_checkpoint_path, map_location='cpu')
    e4e_opts = e4e_checkpoint['opts']
    e4e_opts['checkpoint_path'] = e4e_checkpoint_path
    e4e_opts['device'] = 'cuda' if use_cuda else 'cpu'
    e4e_opts = argparse.Namespace(**e4e_opts)
    e4e = pSp(e4e_opts)
    e4e.eval()
    e4e = e4e.to(device)

    e4e_transforms = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                         [ Pre-trained GAN Generator ]                                          ##
    ##                                                                                                                ##
    ####################################################################################################################

    # Build GAN generator model and load with pre-trained weights
    if args.verbose:
        print("#. Build StyleGAN2 generator model G and load with pre-trained weights...")

    G = load_generator(model_name='stylegan2_ffhq1024', latent_is_w=True, verbose=args.verbose).eval()

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

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
    out_data_dir = None
    out_codes_dir = None
    alignment_errors_file = None
    face_detection_errors_file = None
    ####################################################################################################################
    ##                                                   [ CelebA ]                                                   ##
    ####################################################################################################################
    if args.dataset == 'celeba':
        raise NotImplementedError

    ####################################################################################################################
    ##                                                 [ CelebA-HQ ]                                                  ##
    ####################################################################################################################
    elif args.dataset == 'celebahq':
        dataset = CelebAHQ(root_dir=args.dataset_root, subset='train+val+test')
        dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

        # Create output directory to save images
        out_data_dir = osp.join(out_dir, 'data')
        os.makedirs(out_data_dir, exist_ok=True)

        # Create output directory to save the e4e latent codes
        out_codes_dir = osp.join(out_dir, 'latent_codes')
        os.makedirs(out_codes_dir, exist_ok=True)

        # TODO: Copy annotations dir

        # Create files to store errors on alignment and face detection
        alignment_errors_file = osp.join(out_dir, 'alignment_errors.txt')
        with open(alignment_errors_file, 'w') as f:
            pass
        face_detection_errors_file = osp.join(out_dir, 'face_detection_errors.txt')
        with open(face_detection_errors_file, 'w') as f:
            pass

    ####################################################################################################################
    ##                                                    [ LFW ]                                                     ##
    ####################################################################################################################
    elif args.dataset == 'lfw':
        raise NotImplementedError

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                 [ Inversion ]                                                  ##
    ##                                                                                                                ##
    ####################################################################################################################
    for i_batch, data_batch in enumerate(
            tqdm(dataloader, desc="#. Process {} images".format(args.dataset) if args.verbose else '')):

        aligned_images = []
        for i in range(args.batch_size):

            ############################################################################################################
            ##                                    [ Face alignment and crop ]                                         ##
            ############################################################################################################
            with torch.no_grad():
                img_aligned = face_aligner.align_face(image_file=data_batch[2][i],
                                                      alignment_errors_file=alignment_errors_file,
                                                      face_detection_errors_file=face_detection_errors_file)

            img_aligned_file = osp.join(out_data_dir, '{}_aligned.jpg'.format(get_img_id(data_batch[2][i])))
            save_img(img_file=img_aligned_file, img=img_aligned)

            # TODO: do not re-read the image from disk -- use `img_aligned` from above
            aligned_images.append(e4e_transforms(Image.open(img_aligned_file).convert('RGB')).unsqueeze(0))

        aligned_images = torch.cat(aligned_images).cuda()

        ################################################################################################################
        ##                                       [ Image Inversion (e4e) ]                                            ##
        ################################################################################################################
        image_latent_codes = get_latents(e4e, aligned_images)

        # Save latent codes
        for i in range(args.batch_size):
            img_latent_code_file = osp.join(out_codes_dir, '{}.pt'.format(get_img_id(data_batch[2][i])))
            torch.save(image_latent_codes[i], img_latent_code_file)

        ################################################################################################################
        ##                                     [ Generate reconstructed images ]                                      ##
        ################################################################################################################
        with torch.no_grad():
            img_recon = G(image_latent_codes)

        ################################################################################################################
        ##                        [ Save reconstructed images (using inverted latent codes) ]                         ##
        ################################################################################################################
        if args.save_reconstructed_images:
            for i in range(args.batch_size):
                img_recon_file = osp.join(out_data_dir, '{}_recon.jpg'.format(get_img_id(data_batch[2][i])))
                tensor2image(img_recon[i].cpu(), adaptive=True).save(
                    img_recon_file, "JPEG", quality=90, subsampling=0, progressive=True)


if __name__ == '__main__':
    main()
