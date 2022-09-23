import numpy as np
import cv2

class RootSIFT:
	def __init__(self):
		# initialize the SIFT feature extractor
		self.extractor = cv2.SIFT_create()
		# sift = cv2.xfeatures2d.SURF_create()
		# sift = cv2.SIFT()


	def compute(self, image, eps=1e-7):
		# compute SIFT descriptors

		kps, descs = self.extractor.detectAndCompute(image, None)
		# if there are no keypoints or descriptors, return an empty tuple
		if len(kps) == 0:
			return ([], None)
		# apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
		#descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
		# return a tuple of the keypoints and descriptors
		return (kps, descs)