import tensorflow as tf
import cv2
import os

datadir = 'predata'

def rename_in_place():
	file_count = 0
	for idx, img in enumerate(os.listdir('predata')):
		os.renames(os.path.join(datadir,img), os.path.join(datadir,f'{idx}.jpg'))
		file_count += 1
	print(f'{file_count} images have been renamed.')

def resize_in_place():
	file_count = 0
	for idx, img in enumerate(os.listdir(datadir)):
		image = cv2.imread(os.path.join(datadir, img))
		resized = tf.image.resize(image, (250,250)).numpy()
		cv2.imwrite(os.path.join(datadir, img), resized)
		file_count += 1
	print(f'{file_count} images have been resized.')

def prepare_images():
	# rename_in_place()
	resize_in_place()

def mass_flip(axis):
	"""flips the image in the desired axis

	Args:
		axis (_int_): 0 for vertical, 1 for horizontal
	"""
	prepare_images()
	file_count = 0
	for idx, img in enumerate(os.listdir(datadir)):
		image = cv2.imread(os.path.join(datadir, img))
		flipped = cv2.flip(image, axis)
		cv2.imwrite(os.path.join(datadir, f'flipped_{axis}_{img}'), flipped)
		file_count += 1
	print(f'{file_count} images have been flipped.')

def enhance_contrast(image_path):
	img = cv2.imread(image_path)
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l_channel, a, b = cv2.split(lab)

	# Applying CLAHE to L-channel
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl = clahe.apply(l_channel)

	# merge the CLAHE enhanced L-channel with the a and b channel
	limg = cv2.merge((cl,a,b))

	return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def mass_enhance_contrast():
	prepare_images()
	file_count = 0
	for idx, image in enumerate(os.listdir(datadir)):
		enhanced_image = enhance_contrast(os.path.join(datadir,image))
		cv2.imwrite(os.path.join(datadir, f'enhanced_{image}'), enhanced_image)
		file_count += 1
	print(f'{file_count} images have been enhanced in contrast.')
	
def increase_brightness(image_path, value=30):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    final_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return final_image

def mass_increase_brightness(value=30):
	prepare_images()
	file_count = 0
	for idx, img in enumerate(os.listdir(datadir)):
		bright_image = increase_brightness(os.path.join(datadir, img), value)
		cv2.imwrite(os.path.join(datadir, f'bright_{img}'), bright_image)
		file_count += 1
	print(f'{file_count} images have been enhanced in brightness.')
