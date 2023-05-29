import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

logdir='logs'
data_dir = "data"
weights_dir = 'weights'
test_dir = 'test_images'
image_exts = ['jpeg','jpg']

class Model:
	def __init__(self):
		self.data = tf.keras.utils.image_dataset_from_directory('data')
		# scale down the data to increase performance
		self.data = self.data.map(lambda x, y: (x/255, y))
		self._model = None
		train_size = int(len(self.data)* .7)
		val_size = int(len(self.data)  * .2)
		test_size = int(len(self.data) * .1)

		self.train_data = self.data.take(train_size)
		self.val_data = self.data.skip(train_size).take(val_size)
		self.test_data = self.data.skip(train_size+val_size).take(test_size)

	@property
	def model(self):
		if self._model: return self._model
		else:
			return self.create()

	@model.setter
	def model(self, value):
		self._model = value
	
	def create(self):
		model = Sequential()

		model.add(Conv2D(16, (3,3), 1, activation="relu", input_shape=(256,256,3)))
		model.add(MaxPooling2D())

		model.add(Conv2D(32, (3,3), 1, activation="relu"))
		model.add(MaxPooling2D())

		model.add(Conv2D(64, (3,3), 1, activation="relu"))
		model.add(MaxPooling2D())
		
		model.add(Conv2D(32, (3,3), 1, activation="relu"))
		model.add(MaxPooling2D())

		model.add(Conv2D(16, (3,3), 1, activation="relu"))
		model.add(MaxPooling2D())

		model.add(Flatten())
		
		model.add(Dense(1024, activation="relu"))
		model.add(Dense(256, activation="relu"))
		model.add(Dense(1, activation="sigmoid"))

		# adam optimizer
		model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
		
		self.model = model
		return model
	

	def train(self, epochs=15):
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
		hist = self.model.fit(self.train_data, epochs=epochs, validation_data=self.val_data, callbacks=[tensorboard_callback])
		self.model.save_weights(os.path.join(weights_dir, 'last_weights'))
		self.evaluate()
		self.load_weights()
		return hist
	
	def evaluate(self):
		loss, acc = self._model.evaluate(self.test_data, verbose=2)
		print("Evaluated model,  accuracy: {:5.2f}%".format(100 * acc))
		print("Evaluated model,  loss: {:5.2f}%".format(100 * loss))


	def load_weights(self):
		self.model.load_weights(os.path.join(weights_dir, 'last_weights')).expect_partial()

	def test_and_show_results(self):
		labels= []
		images=[]

		# test
		for idx, test_image in enumerate(os.listdir(test_dir)):
			img = cv2.imread(os.path.join(test_dir, f'fire_test_{idx+1}.jpg'))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			resize = tf.image.resize(img, (256,256))
			yhat = self.model.predict(np.expand_dims(resize/255.0, 0))
			images.append(img)

			print(f"Image {idx+1}:", yhat)
			if yhat > 0.5:
				print(f"No fire")
				labels.append(1)
			else:
				print(f"WILDIRE DETECTED")
				labels.append(0)

		# show results
		batch = [images, labels]
		fig, ax = plt.subplots(ncols=len(os.listdir(test_dir)), figsize=(8,8))
		for idx, img in enumerate(batch[0][:]):
			ax[idx].imshow(img.astype(int))
			ax[idx].title.set_text("no fire" if batch[1][idx] else "WILDFIRE")
			
		plt.show()
