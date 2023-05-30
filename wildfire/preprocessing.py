import tensorflow as tf
import cv2
import os
import numpy as np

datadir = 'predata'

def rename_in_place():
	for idx, img in enumerate(os.listdir('predata')):
		os.renames(os.path.join(datadir,img), os.path.join(datadir,f'{idx}.jpg'))

def resize_in_place():
	for idx, img in enumerate(os.listdir(datadir)):
		image = cv2.imread(os.path.join(datadir, f'{idx}.jpg'))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		resized = tf.image.resize(image, (250,250)).numpy()
		cv2.imwrite(os.path.join(datadir, f'{idx}.jpg'), resized)

def mass_flip(axis):
	"""flips the image in the desired axis

	Args:
		axis (_int_): 0 for vertical, 1 for horizontal
	"""
	for idx, img in enumerate(os.listdir(datadir)):
		image = cv2.imread(os.path.join(datadir,f'{idx}.jpg'))
		flipped = cv2.flip(image, axis)
		cv2.imwrite(os.path.join(datadir, f'flipped{axis}{idx}.jpg'), flipped)

def contrast():
	for idx, image in enumerate(os.listdir(datadir)):
		img = cv2.imread(os.path.join(datadir, f'{idx}.jpg'), 1)
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		l_channel, a, b = cv2.split(lab)

		# Applying CLAHE to L-channel
		# feel free to try different values for the limit and grid size:
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl = clahe.apply(l_channel)

		# merge the CLAHE enhanced L-channel with the a and b channel
		limg = cv2.merge((cl,a,b))

		# Converting image from LAB Color model to BGRB color spcae
		enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

		# Stacking the original image with the enhanced image
		cv2.imwrite(os.path.join(datadir, f'enhanced{idx}.jpg'), enhanced_img)

def main():
	contrast()
	


main()
