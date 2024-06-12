import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.models import model_from_json
import cv2 as cv2

class FacialExpressionModel(object):

	emos = ['angry', 'disgust','fear','happy','neutral','sad','surprise']

	def __init__(self, model_json_file, model_weights_file):

		with open(model_json_file,'r') as f:
			json_model = f.read()
			self.loaded_model = model_from_json(json_model)

		self.loaded_model.load_weights(model_weights_file)
		# self.loaded_model._make_predict_function()


	def predict_emotion(self, img):
		self.preds = self.loaded_model.predict(img)
		return FacialExpressionModel.emos[np.argmax(self.preds)], np.max(self.preds)
