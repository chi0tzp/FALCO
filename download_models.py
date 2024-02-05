import sys
import os
import os.path as osp
import hashlib
import tarfile
import time
import urllib.request
from lib import GENFORCE, SFD, E4E, ARCFACE, FARL, DECA, GAZE


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r  \\__%d%%, %d MB, %d KB/s, %d seconds passed" %
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
        print("  \\__Check sha256: {}".format("OK!" if sha256_check else "Error"))
        if not sha256_check:
            raise Exception("Error: Invalid sha256 sum: {}".format(sha256_hash.hexdigest()))

    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(dest)
    os.remove(tmp_tar)


def files_exist(d, content):
    r = True
    for f in content:
        r = osp.exists(osp.join(d, f)) and r
    return r


def main():
    """Download pre-trained GenForce GAN generators [1], SFD [2], e4e [3], ArcFace [4], FaRL [5], DECA [6], and the
    gaze estimator [7]. The pretrained weights will be stored under `models/pretrained/`.

    References:
         [1] https://genforce.github.io/
         [2] Zhang, Shifeng, et al. "S3FD: Single shot scale-invariant face detector." Proceedings of the IEEE
             international conference on computer vision. 2017.
         [3] Tov, Omer, et al. "Designing an encoder for stylegan image manipulation." ACM Transactions on Graphics
             (TOG) 40.4 (2021): 1-14.
         [4] Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition." Proceedings of
             the IEEE/CVF conference on computer vision and pattern recognition. 2019.
         [5] Zheng, Yinglin, et al. "General Facial Representation Learning in a Visual-Linguistic Manner."
             Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
         [6] Yao Feng, Haiwen Feng, Michael J Black, and Timo Bolkart. Learning an animatable detailed 3d face model
             from in-the-wild images. ACM Transactions on Graphics (TOG), 40(4):1–13, 2021
         [7] Xucong Zhang, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang, and Otmar Hilliges. Eth-xgaze: A large
             scale dataset for gaze estimation under extreme head pose and gaze variation. In European Conference on
             Computer Vision, pages 365–381. Springer, 2020
    """
    # Create pre-trained models root directory
    pretrained_models_root = osp.join('models', 'pretrained')
    os.makedirs(pretrained_models_root, exist_ok=True)

    print("#. Download pre-trained GAN generators...")
    if files_exist(d=osp.join(pretrained_models_root, GENFORCE[2]), content=GENFORCE[3]):
        print("  \\__Already exist.")
    else:
        download(src=GENFORCE[0], sha256sum=GENFORCE[1], dest=pretrained_models_root)

    print("#. Download pre-trained SFD face detector model...")
    if files_exist(d=osp.join(pretrained_models_root, SFD[2]), content=SFD[3]):
        print("  \\__Already exists.")
    else:
        download(src=SFD[0], sha256sum=SFD[1], dest=pretrained_models_root)

    print("#. Download pre-trained e4e GAN inversion model...")
    if files_exist(d=osp.join(pretrained_models_root, E4E[2]), content=E4E[3]):
        print("  \\__Already exists.")
    else:
        download(src=E4E[0], sha256sum=E4E[1], dest=pretrained_models_root)

    print("#. Download pre-trained ArcFace model...")
    if files_exist(d=osp.join(pretrained_models_root, ARCFACE[2]), content=ARCFACE[3]):
        print("  \\__Already exists.")
    else:
        download(src=ARCFACE[0], sha256sum=ARCFACE[1], dest=pretrained_models_root)

    print("#. Download pre-trained FaRL for Facial Representation Learning ...")
    if files_exist(d=osp.join(pretrained_models_root, FARL[2]), content=FARL[3]):
        print("  \\__Already exists.")
    else:
        download(src=FARL[0], sha256sum=FARL[1], dest=pretrained_models_root)

    print("#. Download pre-trained DECA model...")
    if files_exist(d=osp.join(pretrained_models_root, DECA[2]), content=DECA[3]):
        print("  \\__Already exists.")
    else:
        download(src=DECA[0], sha256sum=DECA[1], dest=pretrained_models_root)

    print("#. Download pre-trained Gaze estimation model...")
    if files_exist(d=osp.join(pretrained_models_root, GAZE[2]), content=GAZE[3]):
        print("  \\__Already exists.")
    else:
        download(src=GAZE[0], sha256sum=GAZE[1], dest=pretrained_models_root)


if __name__ == '__main__':
    main()
