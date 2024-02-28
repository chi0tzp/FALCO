import sys
import os
import os.path as osp
import argparse
from torch.utils import data
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from lib import CelebAHQ, LFW


def draw_landmarks():
    pass


def draw_angles():
    pass


def show_images(data_batch, nn_type, inv, anon,
                plot_orig_landmarks=False, plot_orig_angles=False,
                plot_nn_landmarks=False, plot_nn_angles=False,
                plot_recon_landmarks=False, plot_recon_angles=False,
                plot_anon_landmarks=False, plot_anon_angles=False,
                out_fig_file=None):
    """Show figure with images of:

          i) the original dataset, and/or
         ii) inversions of real images produced by e4e (if `--inv` is set), and/or
        iii) fake NNs (if an appropriate json NN map file is given using `--fake-nn-map`), and/or
         iv) the anonymized versions of the real images (if an appropriate directory is given using `--anon`).

    in the above order; i.e., 'Original', 'Recon (e4e)', 'Fake NN', 'Anon'.

    Args:
        data_batch (list)           : data batch as returned by the data loader having the following structure:
                                          -- data_batch[0]  : `img_orig`
                                          -- data_batch[1]  : `img_orig_attr`
                                          -- data_batch[2]  : `img_orig_path`
                                          -- data_batch[3]  : `img_orig_landmarks`
                                          -- data_batch[4]  : `img_orig_angles`
                                          -- data_batch[5]  : `img_nn`
                                          -- data_batch[6]  : `img_nn_code`
                                          -- data_batch[7]  : `img_nn_landmarks`
                                          -- data_batch[8]  : `img_nn_angles`
                                          -- data_batch[9]  : `img_recon`
                                          -- data_batch[10] : `img_recon_code`
                                          -- data_batch[11] : `img_recon_landmarks`
                                          -- data_batch[12] : `img_recon_angles`
                                          -- data_batch[13] : `img_anon`
                                          -- data_batch[14] : `img_anon_code`
                                          -- data_batch[15] : `img_anon_landmarks`
                                          -- data_batch[16] : `img_anon_angles`
        nn_type (str)               : type of NN mapping (based on the given file). If None is given, the corresponding
                                      tensor (`img_nn`) will be zero, and it will not be plotted in the figure.
        inv (bool)                  : show inversion results; i.e., reconstructed images produced by e4e using
                                      `invert.py`
        anon (str)                  : directory of the anonymized version of the dataset produced using `anonymize.py`.
                                      If None is given, the corresponding tensor (`img_anon`) will be zero, and it will
                                      not be plotted in the figure.
        plot_orig_landmarks (bool)  : plot landmarks of the original images
        plot_orig_angles (bool)     : plot Euler angles (yaw, pitch, roll) of the original images
        plot_nn_landmarks (bool)    : plot landmarks of the fake NN images
        plot_nn_angles (bool)       : plot Euler angles (yaw, pitch, roll) of the fake NN images
        plot_recon_landmarks (bool) : plot landmarks of the reconstructed (inverted) images
        plot_recon_angles (bool)    : plot Euler angles (yaw, pitch, roll) of the reconstructed (inverted) images
        plot_anon_landmarks (bool)  : plot landmarks of the anonymized images
        plot_anon_angles (bool)     : plot Euler angles (yaw, pitch, roll) of the anonymized images
        out_fig_file (str)          : output figure filename

    """
    img_orig = data_batch[0]
    img_orig_attr = data_batch[1]
    img_orig_filename = data_batch[2]
    img_orig_landmarks = data_batch[3]
    img_orig_angles = data_batch[4]
    img_nn = data_batch[5]
    img_nn_code = data_batch[6]
    img_nn_landmarks = data_batch[7]
    img_nn_angles = data_batch[8]
    img_recon = data_batch[9]
    img_recon_code = data_batch[10]
    img_recon_landmarks = data_batch[11]
    img_recon_angles = data_batch[12]
    img_anon = data_batch[13]
    img_anon_code = data_batch[14]
    img_anon_landmarks = data_batch[15]
    img_anon_angles = data_batch[16]

    # Tensor to PIL image transform
    tensor2pil = transforms.ToPILImage()

    # Define header height
    header_h = 30

    # Define image size (each cell of the figure)
    img_size = 256

    # Calculate figure's width and height
    grid_w = img_size
    if inv:
        grid_w += img_size
    if nn_type is not None:
        grid_w += img_size
    if anon is not None:
        grid_w += img_size

    grid_h = header_h + img_orig.shape[0] * img_size

    # Create image for the figure
    grid = Image.new(mode='RGB', size=(grid_w, grid_h))

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                   [ Header ]                                                   ##
    ##                                                                                                                ##
    ####################################################################################################################
    header_draw = ImageDraw.Draw(grid)
    header_draw.rectangle(xy=((0, 0), (grid_w, header_h)), fill=(255, 255, 255))

    header_orig = 'Original'
    header_recon = 'Recon (e4e)'
    header_nn = 'Fake NN'
    header_anon = 'Anon'

    # Calculate the position for each figure column
    orig_col = recon_col = nn_col = anon_col = 0
    if inv:
        recon_col = 1
        if nn_type is not None:
            nn_col = 2
            if anon is not None:
                anon_col = 3
        else:
            if anon is not None:
                anon_col = 2
    else:
        if nn_type is not None:
            nn_col = 1
            if anon is not None:
                anon_col = 2
        else:
            if anon is not None:
                anon_col = 1

    # Calculate header font size
    header_cols = header_orig
    if inv:
        header_cols += header_recon
    if nn_type is not None:
        header_cols += header_nn
    if anon is not None:
        header_cols += header_anon

    header_font_size = 1
    while True:
        header_font = ImageFont.truetype("lib/fonts/RobotoCondensed-Bold.ttf", header_font_size)
        header_size = header_font.getbbox(header_cols)
        if (header_size[2] - header_size[0] > 0.7 * grid_w) or (header_size[3] - header_size[1] > 0.7 * header_h):
            break
        else:
            header_font_size += 1

    # Define header font type and colour
    header_font = ImageFont.truetype("lib/fonts/RobotoCondensed-Bold.ttf", header_font_size)
    header_font_colour = (0, 0, 0)

    # Draw header text
    header_orig_size = header_font.getbbox(header_orig)
    header_orig_width = header_orig_size[2] - header_orig_size[0]
    header_recon_size = header_font.getbbox(header_recon)
    header_recon_width = header_recon_size[2] - header_recon_size[0]
    header_nn_size = header_font.getbbox(header_nn)
    header_nn_width = header_nn_size[2] - header_nn_size[0]
    header_anon_size = header_font.getbbox(header_anon)
    header_anon_width = header_anon_size[2] - header_anon_size[0]

    header_draw.text(xy=(orig_col * img_size + 0.5 * img_size - 0.5 * header_orig_width, 0), text=header_orig,
                     font=header_font, fill=header_font_colour)
    if inv:
        header_draw.text(xy=(recon_col * img_size + 0.5 * img_size - 0.5 * header_recon_width, 0), text=header_recon,
                         font=header_font, fill=header_font_colour)
    if nn_type is not None:
        header_draw.text(xy=(nn_col * img_size + 0.5 * img_size - 0.5 * header_nn_width, 0), text=header_nn,
                         font=header_font, fill=header_font_colour)
    if anon is not None:
        header_draw.text(xy=(anon_col * img_size + 0.5 * img_size - 0.5 * header_anon_width, 0), text=header_anon,
                         font=header_font, fill=header_font_colour)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                   [ Images ]                                                   ##
    ##                                                                                                                ##
    ####################################################################################################################
    batch_size = img_orig.shape[0]
    img_name_font = ImageFont.truetype("lib/fonts/RobotoCondensed-Bold.ttf", 16)
    img_name_font_colour = (0, 0, 0)
    img_name_bg_colour = (255, 208, 0)

    for i in range(batch_size):

        # Original (real) image
        img_orig_i = tensor2pil(img_orig[i]).resize((img_size, img_size))
        grid.paste(img_orig_i, (0, header_h + i * img_size))

        img_orig_name_i_draw = ImageDraw.Draw(grid)
        img_orig_name_i = '{}'.format(osp.basename(img_orig_filename[i]))
        img_orig_name_i_size = img_name_font.getbbox(img_orig_name_i)
        img_orig_name_i_draw.rectangle(xy=((0, header_h + i * img_size),
                                           (img_orig_name_i_size[2],
                                            header_h + i * img_size + img_orig_name_i_size[3])),
                                       fill=img_name_bg_colour)
        img_orig_name_i_draw.text(xy=(0, header_h + i * img_size),
                                  text=img_orig_name_i,
                                  font=img_name_font,
                                  fill=img_name_font_colour)

        # TODO: plot landmarks and/or angles
        if plot_orig_landmarks:
            img_orig_landmarks_i = img_orig_landmarks[i]
            raise NotImplementedError

        if plot_orig_angles:
            img_orig_angles_i = img_orig_angles[i]
            raise NotImplementedError

        # e4e inversion (reconstructed) image
        if inv:
            img_recon_i = tensor2pil(img_recon[i]).resize((img_size, img_size))
            grid.paste(img_recon_i, (recon_col * img_size, header_h + i * img_size))

            # TODO: plot landmarks and/or angles
            if plot_recon_landmarks:
                img_recon_landmarks_i = img_recon_landmarks[i]
                raise NotImplementedError

            if plot_recon_angles:
                img_recon_angles_i = img_recon_angles[i]
                raise NotImplementedError

        # Nearest fake neighbor image
        if nn_type is not None:
            img_nn_i = tensor2pil(img_nn[i]).resize((img_size, img_size))
            grid.paste(img_nn_i, (nn_col * img_size, header_h + i * img_size))

            # TODO: plot landmarks and/or angles
            if plot_nn_landmarks:
                img_nn_landmarks_i = img_nn_landmarks[i]
                raise NotImplementedError

            if plot_nn_angles:
                img_nn_angles_i = img_nn_angles[i]
                raise NotImplementedError

        # Anonymized image
        if anon is not None:
            img_anon_i = tensor2pil(img_anon[i]).resize((img_size, img_size))
            grid.paste(img_anon_i, (anon_col * img_size, header_h + i * img_size))

            # TODO: plot landmarks and/or angles
            if plot_anon_landmarks:
                img_anon_landmarks_i = img_anon_landmarks[i]
                raise NotImplementedError

            if plot_anon_angles:
                img_anon_angles_i = img_anon_angles[i]
                raise NotImplementedError

    # Save figure
    if out_fig_file is not None:
        grid.save(out_fig_file, "JPEG", quality=95, subsampling=0, progressive=True)

    # Show grid
    grid.show()


"""
python visualize_dataset.py -v --dataset=celebahq --dataset-root=datasets/CelebA-HQ/
python visualize_dataset.py -v --dataset=celebahq --dataset-root=datasets/CelebA-HQ/ --fake-nn-map=
"""


def main():
    """A script for visualizing a real dataset and, optionally, its anonymized version along with the fake NN of each
    image and its e4e reconstruction.

    Options:
        -v, --verbose       : set verbose mode on
        --dataset           : choose dataset (required)
        --dataset-root      : set dataset's root directory (required)
        --subset            : choose dataset subset ('train', 'val', 'test', 'train+val', 'train+val+test'), if
                              applicable
        --fake-nn-map       : visualize the corresponding fake NN for each original (real) image using the given NN map
                              file (i.e., a json file created by `pair_nn.py`)
        --inv               : visualize the e4e reconstructed images
        --anon              : visualize the anonymized images using the given directory of anonymized dataset
        --batch-size        : set batch size
        --shuffle           : shuffle data samples
        --save              : save figures under `visualization/`

    References:
        [1] Tov, Omer, et al. "Designing an encoder for stylegan image manipulation." ACM Transactions on Graphics (TOG)
            40.4 (2021): 1-14.
        [2] Simone Barattin, Christos Tzelepis, Ioannis Patras, Nicu Sebe, "Attribute-preserving Face Dataset
            Anonymization via Latent Code Optimization", Proceedings of the IEEE Conference on Computer Vision and
            Pattern Recognition (CVPR), 2023.

    """
    parser = argparse.ArgumentParser(description="Dataset visualization")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--dataset', type=str, required=True, choices=('celebahq', 'lfw'),
                        help="choose dataset")
    parser.add_argument('--dataset-root', type=str, required=True, help="set dataset root directory")
    parser.add_argument('--subset', type=str, default='train+val+test',
                        choices=('train', 'val', 'test', 'train+val', 'train+val+test'), help="choose dataset's subset")
    parser.add_argument('--lfw-anno', type=str, help="TODO")
    parser.add_argument('--fake-nn-map', type=str,
                        help="visualize NNs using the given NN map file (created by `pair_nn.py`)")
    parser.add_argument('--inv', action='store_true', help="visualize e4e inversions (created by `invert.py`)")
    parser.add_argument('--anon', type=str,
                        help="visualize the anonymized images using the given directory of anonymized dataset")
    parser.add_argument('--save', action='store_true', help="save figures under visualization/<dataset>")
    parser.add_argument('--batch-size', type=int, default=4, help="set batch size")
    parser.add_argument('--shuffle', action='store_true', help="shuffle data samples")

    # Parse given arguments
    args = parser.parse_args()

    # Check given dataset root directory
    if not osp.isdir(args.dataset_root):
        raise NotADirectoryError("Invalid dataset root directory: {}".format(args.dataset_root))

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                [ Data Loader ]                                                 ##
    ##                                                                                                                ##
    ####################################################################################################################
    dataset = None
    dataloader = None

    ####################################################################################################################
    ##                                                 [ CelebA-HQ ]                                                  ##
    ####################################################################################################################
    if args.dataset == 'celebahq':
        dataset = CelebAHQ(root_dir=args.dataset_root,
                           subset=args.subset,
                           fake_nn_map=args.fake_nn_map,
                           inv=args.inv,
                           anon=args.anon)
        dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    ####################################################################################################################
    ##                                                    [ LFW ]                                                     ##
    ####################################################################################################################
    elif args.dataset == 'lfw':
        dataset = LFW(root_dir=args.dataset_root,
                      subset=args.subset,
                      annotation_file=args.lfw_anno,
                      fake_nn_map=args.fake_nn_map,
                      inv=args.inv,
                      anon=args.anon)
        dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                               [ Visualization ]                                                ##
    ##                                                                                                                ##
    ####################################################################################################################
    # Create directory to save figures
    save_dir = None
    if args.save:
        save_dir = '{}'.format(args.dataset)
        if args.fake_nn_map is not None:
            save_dir += '+nn'
        if args.inv:
            save_dir += '+inv'
        if args.anon is not None:
            save_dir += '+anon'
        save_dir = osp.join('visualization', save_dir)
        if args.verbose:
            print("#. Create dir for storing visualization figures: {}".format(save_dir))
        os.makedirs(save_dir, exist_ok=True)

    # Create and show figures
    for data_batch in dataloader:

        out_fig_file = None
        if args.save:
            out_fig_file = '{}'.format('_'.join([osp.basename(f).split('.')[0] for f in data_batch[2]]))
            if args.fake_nn_map is not None:
                out_fig_file += '_{}.jpg'.format(osp.basename(args.fake_nn_map).split('.')[0])
            out_fig_file = osp.join(save_dir, out_fig_file)

        show_images(data_batch=data_batch, nn_type=dataset.nn_type, inv=args.inv, anon=args.anon,
                    plot_orig_landmarks=False, plot_orig_angles=False,
                    plot_nn_landmarks=False, plot_nn_angles=False,
                    plot_recon_landmarks=False, plot_recon_angles=False,
                    plot_anon_landmarks=False, plot_anon_angles=False,
                    out_fig_file=out_fig_file)

        input("__> Press ENTER to continue...\n")


if __name__ == '__main__':
    main()
