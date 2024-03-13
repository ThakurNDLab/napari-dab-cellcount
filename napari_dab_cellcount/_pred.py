import numpy as np
from PIL import Image
from skimage import exposure
import argparse
import torch
from ._model import VGGUnet

def tile_image(image, tile_size=(256, 256)):
	image = np.asarray(image)
	padded_height = ((image.shape[0] - 1) // tile_size[0] + 1) * tile_size[0]
	padded_width = ((image.shape[1] - 1) // tile_size[1] + 1) * tile_size[1]

	# Pad the image using NumPy
	padded_image = np.zeros((padded_height, padded_width, image.shape[2]), dtype=image.dtype)
	padded_image[:image.shape[0], :image.shape[1], :] = image

	tiles = []
	for y in range(0, padded_image.shape[0], tile_size[0]):
		for x in range(0, padded_image.shape[1], tile_size[1]):
			tile = padded_image[y:y + tile_size[0], x:x + tile_size[1], :]
			tiles.append(tile)
	return tiles

def predict_tiles(tiles, model, cuda=False):
	# Stack the tiles into a single array for batch processing
	batch = np.stack([preprocess_tile(tile) for tile in tiles])
	if cuda==True:
		batch = torch.tensor(batch, dtype=torch.float32).cuda()
	# Predict masks for the entire batch
	masks = model(batch)
	return masks

def stitch_masks(masks, original_shape, tile_size=(256, 256)):
	stitched_image = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
	index = 0
	for y in range(0, stitched_image.shape[0], tile_size[0]):
		for x in range(0, stitched_image.shape[1], tile_size[1]):
			stitched_image[y:y + tile_size[0], x:x + tile_size[1]] = masks[index][:original_shape[0] - y, :original_shape[1] - x]
			index += 1
	return stitched_image

def preprocess_tile(image):
	image = exposure.equalize_adapthist(image, clip_limit=0.3).astype(np.float32)
	image = np.transpose(image, (2, 0, 1))
	mean = np.mean(image)
	std = np.std(image)
	image = (image - mean) / std
	return image

def postprocess_mask(mask):
	mask = mask.squeeze().cpu().detach().numpy()
	threshold = 0.5
	mask = np.where(mask > threshold, 255, 0).astype(np.uint8)
	# watershedding, merging, filling holes, etc.
	return mask

def pred(image, device=cpu, weights):
	tiles = tile_image(image)
	if device==gpu:
		model = VGGUnet().cuda()
	model.load_state_dict(torch.load(weights))
	model.eval()

	with torch.no_grad():
		if device==gpu:
			masks = predict_tiles(tiles, model, True)
	
	stitched_mask = stitch_masks(masks, image.shape)
	mask = postprocess_mask(stitched_mask)
	
	return mask