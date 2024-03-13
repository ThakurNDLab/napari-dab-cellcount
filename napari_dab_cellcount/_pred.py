import numpy as np
from PIL import Image
from skimage import exposure, morphology, draw
from skimage.measure import find_contours
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
import argparse
import torch
from ._model import ResNetUnet
from scipy import stats

def tile_image(image_, tile_size=(256, 256), overlap_percent=0.25):
    image_ = np.asarray(image_)
    overlap = int(tile_size[0] * overlap_percent), int(tile_size[1] * overlap_percent)
    step = tile_size[0] - overlap[0], tile_size[1] - overlap[1]

    tiles_ = []
    for y in range(0, image_.shape[0], step[0]):
        for x in range(0, image_.shape[1], step[1]):
            # Extract tile with overlap
            tile_ = image_[max(y - overlap[0]//2, 0): min(y + tile_size[0] + overlap[0]//2, image_.shape[0]),
                           max(x - overlap[1]//2, 0): min(x + tile_size[1] + overlap[1]//2, image_.shape[1])]
            
            # Calculate padding to ensure uniform size
            pad_y = tile_size[0] + overlap[0] - tile_.shape[0]
            pad_x = tile_size[1] + overlap[1] - tile_.shape[1]
            padding = ((0, max(pad_y, 0)), (0, max(pad_x, 0)), (0, 0))

            # Apply padding
            tile_ = np.pad(tile_, padding, mode='constant', constant_values=0)
            tiles_.append(tile_)
    return tiles_

def stitch_masks(masks_, original_shape, tile_size=(256, 256), overlap_percent=0.25):
    stitched_image_ = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
    overlap = int(tile_size[0] * overlap_percent), int(tile_size[1] * overlap_percent)
    step = tile_size[0] - overlap[0], tile_size[1] - overlap[1]

    index = 0
    for y in range(0, original_shape[0], step[0]):
        for x in range(0, original_shape[1], step[1]):
            tile_ = masks_[index]

            # Define the area covered by the tile in the stitched image
            y_start = max(y - overlap[0] // 2, 0)
            x_start = max(x - overlap[1] // 2, 0)
            y_end = min(y_start + tile_size[0] + overlap[0], original_shape[0])
            x_end = min(x_start + tile_size[1] + overlap[1], original_shape[1])

            # Majority voting for overlap
            if y > 0 and x > 0:
                for i in range(y_start, y_end):
                    for j in range(x_start, x_end):
                        if stitched_image_[i, j] != 0:
                            stitched_image_[i, j] = stats.mode([stitched_image_[i, j], tile_[i - y_start, j - x_start]])[0]
                        else:
                            stitched_image_[i, j] = tile_[i - y_start, j - x_start]
            else:
                stitched_image_[y_start:y_end, x_start:x_end] = tile_[:y_end - y_start, :x_end - x_start]

            index += 1

    return stitched_image_

def predict_tiles(tiles_, model_, cuda=False):
	num_tiles = len(tiles_)
	batch_size = 6  # Process 6 tiles at a time
	predictions_ = []
	for i in range(0, num_tiles, batch_size):
		batch_tiles = tiles_[i:i + batch_size]
		batch_ = np.stack([preprocess_tile(tile) for tile in batch_tiles])
		print(batch_.shape)
		if cuda:
			batch_ = torch.tensor(batch_, dtype=torch.float32).cuda()
		else:
			batch_ = torch.tensor(batch_, dtype=torch.float32)
		with torch.no_grad():
			batch_predictions_ = model_(batch_).cpu().numpy()
		predictions_.extend(batch_predictions_)
		if cuda:
			torch.cuda.empty_cache()
	masks_ = np.concatenate(predictions_, axis=0)
	return masks_

def preprocess_tile(image_):
	image_ = exposure.equalize_adapthist(image_, clip_limit=0.3).astype(np.float32)
	image_ = np.transpose(image_, (2, 0, 1))
	mean = np.mean(image_)
	std = np.std(image_)
	epsilon = 1e-8
	image_ = (image_ - mean) / (std+epsilon)
	return image_

def mean_blob_size(mask_):
	labels_, labels_nr_ = ndi.label(mask_)
	if labels_nr_ < 2:
		mean_area = 1
		mean_radius = 1
	else:
		unique_labels_, counts_ = np.unique(labels_, return_counts=True)
		# Exclude the background label, which is typically 0
		counts_ = counts_[unique_labels_ > 0]
		mean_area = counts_.mean()
		mean_radius = int(np.round(np.sqrt(mean_area) / np.pi))
	return mean_area, mean_radius

def pad_mask(mask_, pad):
	if pad <= 1:
		pad = 2
	h, w = mask_.shape
	h_pad = h + 2 * pad
	w_pad = w + 2 * pad

	padded_mask_ = np.zeros((h_pad, w_pad), dtype=mask_.dtype)
	padded_mask_[pad:-pad, pad:-pad] = mask_
	padded_mask_[0:pad, :] = 1
	padded_mask_[-pad:, :] = 1
	padded_mask_[:, 0:pad] = 1
	padded_mask_[:, -pad:] = 1
	
	return padded_mask_

def crop_mask(mask_, crop):
	if crop <= 1:
		crop = 2
	h, w = mask_.shape
	mask_ = mask_[crop:h - crop, crop:w - crop]
	return mask_

def drop_artifacts(mask_after_, mask_pre_, min_coverage=0.25):
	connected_, nr_connected_ = ndi.label(mask_after_)
	mask_ = np.zeros_like(mask_after_)
	for i in range(1, nr_connected_ + 1):
		conn_blob_ = np.where(connected_ == i, 1, 0)
		initial_space_ = np.where(connected_ == i, mask_pre_, 0)
		blob_size = np.sum(conn_blob_)
		initial_blob_size = np.sum(initial_space_)
		coverage = float(initial_blob_size) / float(blob_size)
		if coverage > min_coverage:
			mask_ = mask_ + conn_blob_
		else:
			mask_ = mask_ + initial_space_
	return mask_

def clean(mask_, con_):
	thresh_mask = threshold_otsu(mask_)
	thresh_con = threshold_otsu(con_)
	m_ = mask_ > thresh_mask
	c_ = mask_ > thresh_con
	mask_ = np.where(m_ | c_, 1, 0).astype(np.uint8)
	mask_ = ndi.binary_fill_holes(mask_).astype(np.uint8)

	area, radius = mean_blob_size(mask_)
	struct_size = int(1.25 * radius)
	struct_el = morphology.disk(struct_size)
	mask_ = pad_mask(mask_, pad=struct_size)
	mask_ = morphology.binary_opening(mask_, footprint=struct_el)
	mask_ = crop_mask(mask_, crop=struct_size)

	area, radius = mean_blob_size(mask_)
	struct_size = int(0.75*radius)
	struct_el = morphology.disk(struct_size)
	mask_ = np.where(c_ & (~m_), 0, mask_)
	mask_ = pad_mask(mask_, pad=struct_size)
	mask_ = morphology.binary_opening(mask_, footprint=struct_el)
	mask_ = crop_mask(mask_, crop=struct_size)

	mask_ = np.where(m_ | mask_, 1, 0)
	mask_ = ndi.binary_fill_holes(mask_)

	mask_ = drop_artifacts(mask_, m_, min_coverage=0.25)

	return mask_

def good_markers(mask_, con_):
	c_b_ = con_ > threshold_otsu(con_)
	mk_ = np.where(c_b_ , 0, mask_)
	area, radius = mean_blob_size(mask_)
	struct_size = int(0.25*radius)
	struct_el = morphology.disk(struct_size)
	mask_ = pad_mask(mk_, pad=struct_size)
	mask_ = morphology.erosion(mask_, footprint=struct_el)
	mk_ = crop_mask(mask_, crop=struct_size)
	mk_,_ = ndi.label(mk_) 
	return mk_

def good_distance(m_b_):
	distance_ = ndi.distance_transform_edt(m_b_)
	return distance_

def add_dropped_water_blobs(water_, mask_cleaned_):
	water_mask_ = (water_ > 0).astype(np.uint8)
	dropped_ = mask_cleaned_ - water_mask_
	dropped_, _ = ndi.label(dropped_)
	dropped_ = np.where(dropped_, dropped_ + water_.max(), 0)
	water_ = water_ + dropped_
	return water_

def drop_artifacts_per_label(labels_, initial_mask_):
	labels_cleaned_ = np.zeros_like(labels_)
	for i in range(1, labels_.max() + 1):
		component_ = np.where(labels_ == i, 1, 0)
		component_initial_mask_ = np.where(labels_ == i, initial_mask_, 0)
		component_ = drop_artifacts(component_, component_initial_mask_)
		labels_cleaned_ = labels_cleaned_ + component_ * i
	return labels_cleaned_

def relabel(img_):
	h, w = img_.shape

	relabel_dict_ = {}

	for i, k in enumerate(np.unique(img_)):
		if k == 0:
			relabel_dict_[k] = 0
		else:
			relabel_dict_[k] = i
	for i, j in product(range(h), range(w)):
		img_[i, j] = relabel_dict_[img_[i, j]]
	return img_

def drop_small(img_, min_size):
	img_ = morphology.remove_small_objects(img_, min_size=min_size)
	img_ = relabel(img_)
	return img_

def contours_to_image(contours_, image_shape_):
	contour_img_ = np.zeros(image_shape_, dtype=np.uint8)
	for contour in contours_:
		rr, cc = draw.polygon(contour[:, 0], contour[:, 1], contour_img_.shape)
		contour_img_[rr, cc] = 1
	return contour_img_

def watershed(mask_):
	con_ = find_contours(mask_, 0.6)
	con_ = contours_to_image(con_, mask_.shape)
	cleaned_mask_ = clean(mask_, con_)
	good_markers_ = good_markers(cleaned_mask_, con_)
	good_distance_ = good_distance(cleaned_mask_)
	
	labels_ = morphology.watershed(-good_distance_, good_markers_, mask=cleaned_mask_)
	
	labels_ = add_dropped_water_blobs(labels_, cleaned_mask_)
	
	m_thresh = threshold_otsu(mask_)
	initial_mask_binary_ = (mask_ > m_thresh).astype(np.uint8)
	labels_ = drop_artifacts_per_label(labels_,initial_mask_binary_)
	
	labels_ = drop_small(labels_, min_size=40)
	labels_ = fill_holes_per_blob(labels_)
		
	return labels_

def postprocess(tiles_, batch_size=10):
	processed_tiles_ = []
	for i in range(0, len(tiles_), batch_size):
		batch_ = tiles_[i:i + batch_size]
		processed_batch_ = [watershed(tile) for tile in batch_]
		processed_tiles_.extend(processed_batch_)
	return processed_tiles_


def pred(image_, weights, device='cpu'):
	tiles_ = tile_image(image_)
	
	#model_ = ResNetUnet()
	model_ = VGGUnet()
	model_.load_state_dict(torch.load(weights))
	model_.eval()

	with torch.no_grad():
		if device=='gpu':
			masks_ = predict_tiles(tiles_, model_.cuda(), True)
		else:
			masks_ = predict_tiles(tiles_, model_, False)

		masks_ = masks_.squeeze()

	mask_ = postprocess(stitched_mask_)

	stitched_mask_ = stitch_masks(masks_, np.array(image_).shape)

	return mask_