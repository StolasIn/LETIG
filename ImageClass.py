import numpy as np
from PIL import Image

def load_image(path):
	img = Image.open(path)
	img = img.convert('RGB')
	return img

def shape(img):
	width, height = img.size
	return width, height

def rgb2rgba(img):
	img = img.convert('RGBA')
	return img

def gray2rgb(img):
	img = img.convert('RGB')
	return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    return Image.fromarray(tensor)

def crop_image(img, pos):
	assert len(pos) == 4

	img = img.crop(pos)
	return img

def numpy_to_image(array):
	array = np.clip(array, 0, 255)
	if len(array.shape) == 3:
		if array.shape[0] == 3:
			array = np.transpose(array, (1, 2, 0))
			img = Image.fromarray(np.uint8(array), 'RGB')
		elif array.shape[-1] == 3:
			img = Image.fromarray(np.uint8(array), 'RGB')
		else:
			img = Image.fromarray(np.uint8(array), 'RGBA')
	elif len(array.shape) == 2:
		img = Image.fromarray(np.uint8(array), 'L')
		img = img.convert('RGB')
	return img

def image_to_numpy(img):
	return np.array(img)

def save(img, name):
	img.save(name)