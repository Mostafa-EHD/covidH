from flask import url_for
from PIL import Image
#DL packages
import base64
import io
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Concatenate, Add
from keras.engine import Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import load_model
from keras import applications
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, model_from_json
import cv2
import json
import time
import os

from flask import request, render_template
from flask import jsonify
from flask import Flask
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#tf.config.experimental.list_physical_devices('GPU')
app = Flask(__name__, static_url_path='/static')

def get_model():
	global model
	# load json and create model
	##json_file = open('model_Covid_InceptionResnetV211.json', 'r')
	##loaded_model_json = json_file.read()
	##json_file.close()
	##model = model_from_json(loaded_model_json)
	# load weights into new model
	##model.load_weights("covid_InceptionResnetV211_weights.h5")
	print("Loaded model from disk")
	#model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	model = load_model('Covid_Checkpoint_InceptionResntV2_chkpt.h5')
	print(" * Model loaded!")


def preprocess_image(im):
	if im.ndim == 3:
		im = im
	else :
		im = np.dstack([im, im, im])
	print(" * image pre-processing...")
	im = cv2.resize(im, (224, 224))
	#image = img_to_array(image)
	image = im.astype(np.float16)
	image /= 255.0
	image = np.expand_dims(image, axis=0)
	print(" * image processed !")
	return image

print(" * Loading Keras model...")
get_model()

@app.route("/predict",methods=['POST'])
def predict():
	print("***Début predict!")
	message = request.get_json(force=True)
	print(" * image test1 !")
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	#image = cv2.imread(io.BytesIO(decoded))
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(np.float32(image))
	prediction = model.predict(processed_image).tolist()
	print(prediction)
	if (np.argmax(prediction)==0):
		pred = 'COVID-19'
	elif (np.argmax(prediction)==1):
		pred = 'Normal'
	else:
		pred = 'Pneumonie'
	response = {
		'prediction': {
		'normal': prediction[0][1]*100,
		'covid': prediction[0][0]*100,
		'pneumonia': prediction[0][2]*100,
		'pred': pred
		}
	}
	return jsonify(response)



#if __name__ == '__main__':
#	app.run(host= '0.0.0.0', port=10000)
