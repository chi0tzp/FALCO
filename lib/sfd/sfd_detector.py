from .core import FaceDetector
from .detect import *
import torch.backends.cudnn as cudnn


class SFDDetector(FaceDetector):
	def __init__(self, device, path_to_detector=None, verbose=False):
		super(SFDDetector, self).__init__(device, verbose)
		
		self.device = device
		if self.device == "cuda":
			cudnn.benchmark = True
			self.is_cuda = True
		else:
			self.is_cuda = False

		model_weights = torch.load(path_to_detector)

		self.face_detector = s3fd()
		self.face_detector.load_state_dict(model_weights)

		if self.is_cuda:
			self.face_detector.cuda()
		self.face_detector.eval()

	def detect_from_image(self, tensor_or_path):
		image = self.tensor_or_path_to_ndarray(tensor_or_path)

		bboxlist = detect(self.face_detector, image, device=self.device)[0]
		keep = nms(bboxlist, 0.3)
		
		bboxlist = bboxlist[keep, :]
		bboxlist = [x for x in bboxlist if x[-1] > 0.5]
		return bboxlist

	def detect_from_batch(self, tensor):
		bboxlists = batch_detect(self.face_detector, tensor, device=self.device)

		error = False
		new_bboxlists = []
		error_index = -1
		for i in range(bboxlists.shape[0]):
			bboxlist = bboxlists[i]
			keep = nms(bboxlist, 0.3)

			if len(keep) > 0:
				bboxlist = bboxlist[keep, :]
				bboxlist = [x for x in bboxlist if x[-1] > 0.5]
				new_bboxlists.append(bboxlist)
				if len(bboxlist) == 0:
					error = True
			else:
				error = True
				error_index = i
				new_bboxlists.append([])

		return new_bboxlists, error, error_index

	@property
	def reference_scale(self):
		return 195

	@property
	def reference_x_shift(self):
		return 0

	@property
	def reference_y_shift(self):
		return 0
