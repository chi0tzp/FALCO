from enum import Enum
from torch.utils.model_zoo import load_url
from .sfd.sfd_detector import SFDDetector as FaceDetector
from .fan_model.models import FAN
from .fan_model.utils import *
from PIL import Image
import scipy.ndimage


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Initialise the dimensions of the image to be resized and grab the image size
    (h, w) = image.shape[:2]

    # If both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # Check to see if the width is None
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
        scale = r

    # Otherwise, the height is None
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
        scale = r

    if width is not None and height is not None:
        dim = (width, height)

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized, scale


class NetworkSize(Enum):
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


models_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
}


def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center and the scale is provided the function will
    return the points also in the original coordinate frame.

    Arguments:
        hm (torch.tensor) -- the predicted heatmaps, of shape [B, N, W, H]
        center ()
        scale ()

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})

    """
    _, idx = torch.max(hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx = idx + 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if (0 < pX < 63) and (0 < pY < 63):
                diff = torch.FloatTensor([hm_[pY, pX + 1] - hm_[pY, pX - 1], hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True)

    return preds, preds_orig


class FaceAligner:
    def __init__(self, device='cuda'):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load all needed models - Face detector and Pose detector
        network_size = NetworkSize.LARGE
        network_size = int(network_size)
        self.flip_input = False

        # SFD face detection
        # path_to_detector = os.path.join(sys.path[0], 'lib/sfd/s3fd-619a316812.pth')
        path_to_detector = os.path.join('models', 'pretrained', 'sfd', 's3fd-619a316812.pth')
        self.face_detector = FaceDetector(device='cuda', verbose=False, path_to_detector=path_to_detector)

        # self.transformations_image = transforms.Compose([transforms.Resize((224, 224), antialias=True),
        #                                                  transforms.CenterCrop(224),
        #                                                  transforms.ToTensor(),
        #                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                                       std=[0.229, 0.224, 0.225])])
        # self.transformations = transforms.Compose([transforms.Resize((224, 224), antialias=True),
        #                                            transforms.CenterCrop(224),
        #                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                                 std=[0.229, 0.224, 0.225])])

        # Initialise the face alignment networks
        self.face_alignment_net = FAN(network_size)
        self.face_alignment_net.load_state_dict(load_url(models_urls['2DFAN-' + str(network_size)],
                                                         map_location=lambda storage, loc: storage))
        self.face_alignment_net.to(self.device)
        self.face_alignment_net.eval()

    def find_landmarks(self, face, image):

        center = torch.FloatTensor([(face[2] + face[0]) / 2.0, (face[3] + face[1]) / 2.0])
        center[1] = center[1] - (face[3] - face[1]) * 0.12
        scale = (face[2] - face[0] + face[3] - face[1]) / self.face_detector.reference_scale

        inp = crop_torch(image.unsqueeze(0), center, scale).float().cuda()
        inp = inp.div(255.0)
        out = self.face_alignment_net(inp)[-1]

        if self.flip_input:
            out = out + flip(self.face_alignment_net(flip(inp))[-1], is_label=True)
        pts, pts_img = get_preds_fromhm(out.cpu(), center, scale)

        pts, pts_img = pts.view(-1, 68, 2) * 4, pts_img.view(-1, 68, 2)

        return pts_img

    @staticmethod
    def read_image_opencv(image_path):
        return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB).astype('uint8')

    @staticmethod
    def check_alignment(landmarks, image_file, alignment_errors_file, transform_size=256):
        # Get estimated landmarks
        lm = landmarks
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise

        # Calculate auxiliary vectors
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        qsize = np.hypot(*x) * 2

        shrink = int(np.floor(qsize / transform_size * 0.5))

        if shrink < 1:
            with open(alignment_errors_file, "a") as f:
                f.write('{}\n'.format(image_file))
            return False
        else:
            return True

    @staticmethod
    def align_crop_image(image, landmarks, image_file, transform_size=256):
        # Get estimated landmarks
        lm = landmarks
        lm_chin = lm[0: 17]            # left-right
        lm_eyebrow_left = lm[17: 22]   # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]           # top-down
        lm_nostrils = lm[31: 36]       # top-down
        lm_eye_left = lm[36: 42]       # left-clockwise
        lm_eye_right = lm[42: 48]      # left-clockwise
        lm_mouth_outer = lm[48: 60]    # left-clockwise
        lm_mouth_inner = lm[60: 68]    # left-clockwise

        # Calculate auxiliary vectors
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        img = Image.fromarray(image)
        shrink = int(np.floor(qsize / transform_size * 0.5))

        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.Resampling.LANCZOS)
            quad /= shrink
            qsize /= shrink

        # Crop
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        enable_padding = True
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]

            mask = np.maximum(
                1.0 - np.minimum(np.float32(x) / (pad[0] + 1e-12), np.float32(w - 1 - x) / (pad[2] + 1e-12)),
                1.0 - np.minimum(np.float32(y) / (pad[1] + 1e-12), np.float32(h - 1 - y) / (pad[3] + 1e-12)))

            blur = qsize * 0.01
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')

            quad += pad[:2]

        # Transform
        img = img.transform((transform_size, transform_size), Image.Transform.QUAD, (quad + 0.5).flatten(),
                            Image.Resampling.BILINEAR)

        return np.array(img)

    @torch.no_grad()
    def align_face(self, image_file, alignment_errors_file, face_detection_errors_file):

        # TODO: add comment
        image_opencv = self.read_image_opencv(image_file).copy()

        # TODO: add comment
        image = torch.tensor(np.transpose(image_opencv, (2, 0, 1))).unsqueeze(0).float().cuda()

        # TODO: add comment
        detected_faces, face_detection_error, _ = self.face_detector.detect_from_batch(image)

        if face_detection_error:
            with open(face_detection_errors_file, "a") as f:
                f.write('{}\n'.format(image_file))

        # TODO: add comment
        if not face_detection_error:
            landmarks_np = self.find_landmarks(detected_faces[0][0], image[0]).numpy()[0]
            if self.check_alignment(landmarks_np, image_file, alignment_errors_file):
                return self.align_crop_image(image=image_opencv, landmarks=landmarks_np, image_file=image_file)
            else:
                return image_opencv
        else:
            return image_opencv
