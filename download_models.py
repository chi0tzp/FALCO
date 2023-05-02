import sys
import os
import os.path as osp
import hashlib
import tarfile
import time
import urllib.request
from lib import GENFORCE, GENFORCE_MODELS, E4E, SFD, FARL, ARCFACE


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r      \\__%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()


def download(src, sha256sum, dest):
    tmp_tar = osp.join(dest, ".tmp.tar")
    try:
        urllib.request.urlretrieve(src, tmp_tar, reporthook)
    except:
        raise ConnectionError("Error: {}".format(src))

    sha256_hash = hashlib.sha256()
    with open(tmp_tar, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

        sha256_check = sha256_hash.hexdigest() == sha256sum
        print()
        print("      \\__Check sha256: {}".format("OK!" if sha256_check else "Error"))
        if not sha256_check:
            raise Exception("Error: Invalid sha256 sum: {}".format(sha256_hash.hexdigest()))

    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(dest)
    os.remove(tmp_tar)


def main():
    """Download pre-trained GenForce GAN generators [1], e4e GAN inversion encoder [2], SFD face detector [3], and
       FaRL [4] models.

    References:
         [1] https://genforce.github.io/
         [2] Tov, Omer, et al. "Designing an encoder for stylegan image manipulation."
            ACM Transactions on Graphics (TOG) 40.4 (2021): 1-14.
         [3] Zhang, Shifeng, et al. "S3FD: Single shot scale-invariant face detector." Proceedings of the IEEE
             international conference on computer vision. 2017.
         [4] Zheng, Yinglin, et al. "General Facial Representation Learning in a Visual-Linguistic Manner."
             Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
    """
    # Create pre-trained models root directory
    pretrained_models_root = osp.join('models', 'pretrained')
    os.makedirs(pretrained_models_root, exist_ok=True)

    # Download the following pre-trained GAN generators (under models/pretrained/)
    print("#. Download pre-trained GAN generators...")
    print("  \\__.GenForce")
    download_genforce_models = False
    for k, v in GENFORCE_MODELS.items():
        if not osp.exists(osp.join(pretrained_models_root, 'genforce', v[0])):
            download_genforce_models = True
            break
    if download_genforce_models:
        download(src=GENFORCE[0], sha256sum=GENFORCE[1], dest=pretrained_models_root)
    else:
        print("      \\__Already exists.")

    print("#. Download pre-trained e4e inversion encoder...")
    print("  \\__.e4e")
    if osp.exists(osp.join(pretrained_models_root, 'e4e', 'model_ir_se50.pth')) and \
        osp.exists(osp.join(pretrained_models_root, 'e4e', 'e4e_ffhq_encode.pt')) and \
            osp.exists(osp.join(pretrained_models_root, 'e4e', 'shape_predictor_68_face_landmarks.dat')):
        print("      \\__Already exists.")
    else:
        download(src=E4E[0], sha256sum=E4E[1], dest=pretrained_models_root)

    print("#. Download pre-trained SFD face detector model...")
    print("  \\__.Face detector (SFD)")
    if osp.exists(osp.join(pretrained_models_root, 'sfd', 's3fd-619a316812.pth')):
        print("      \\__Already exists.")
    else:
        download(src=SFD[0], sha256sum=SFD[1], dest=pretrained_models_root)

    print("#. Download pre-trained FaRL for Facial Representation Learning ...")
    print("  \\__.FaRL")
    if osp.exists(osp.join(pretrained_models_root, 'farl', 'FaRL-Base-Patch16-LAIONFace20M-ep16.pth')) and \
            osp.exists(osp.join(pretrained_models_root, 'farl', 'FaRL-Base-Patch16-LAIONFace20M-ep64.pth')):
        print("      \\__Already exists.")
    else:
        download(src=FARL[0], sha256sum=FARL[1], dest=pretrained_models_root)

    print("#. Download pre-trained ArcFace model...")
    print("  \\__.ArcFace")
    if osp.exists(osp.join(pretrained_models_root, 'arcface', 'model_ir_se50.pth')):
        print("      \\__Already exists.")
    else:
        download(src=ARCFACE[0], sha256sum=ARCFACE[1], dest=pretrained_models_root)


if __name__ == '__main__':
    main()
