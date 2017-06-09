import cv2
import numpy as np

class webcamFeeder(object):

	def __init__(self, color):
		self.vc = cv2.VideoCapture(0)
		self.width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.channels = 3
		self.color = color

	def next_batch(self, batchSize):
		if not self.vc.isOpened():
			print("Webcam Open Error")
			return
		feed = np.zeros((batchSize, self.height, self.width, self.channels))
		for i in range(batchSize):
			rval, frame = self.vc.read()
			if not rval:
				print("Webcam Read Error")
				return
			feed[i, :, :, :] = frame
		if not self.color:
			feed = np.mean(feed, axis=3, keepdims=True)
		feed = feed / 255.0
		return feed

	def __del__(self):
		self.vc.release()

class videoFeeder(object):

	def __init__(self, videoFile, color):
		self.vc = cv2.VideoCapture(videoFile)
		self.width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.channels = 3
		self.color = color

	def next_batch(self, batchSize):
		if not self.vc.isOpened():
			print("Video Open Error")
			return
		feed = np.zeros((batchSize, self.height, self.width, self.channels))
		for i in range(batchSize):
			rval, frame = self.vc.read()
			if not rval:
				print("Video Read Error")
				return
			feed[i, :, :, :] = frame
		if not self.color:
			feed = np.mean(feed, axis=3, keepdims=True)
		feed = feed / 255.0
		return feed

	def __del__(self):
		self.vc.release()