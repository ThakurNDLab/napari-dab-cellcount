"""
dab-cellcount dock widget module
"""
from typing import Any
from napari_plugin_engine import napari_hook_implementation
import gc
import time
import numpy as np
import logging
from qtpy.QtWidgets import QVBoxLayout, QLabel, QWidget, QFileDialog

from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
import os
import sys
import torch
from ._pred import pred
from PIL import Image
from scipy.ndimage import label

# Initialize logger
logger = logging.getLogger(__name__)
if '--verbose' in sys.argv or '-v' in sys.argv:
	logger.setLevel(logging.DEBUG)
else:
	logger.setLevel(logging.WARNING)

def read_logging(log_file, logwindow):
	with open(log_file, 'r') as thefile:
		while True:
			line = thefile.readline()
			if not line:
				time.sleep(0.01) # Sleep briefly
				continue
			else:
				logwindow.cursor.movePosition(logwindow.cursor.End)
				logwindow.cursor.insertText(line)
				yield line

cc_strings = ['_cc_masks_', '_cc_outlines_']



def get_reference_():
    try:
        # This will work for normal and compiled Python (.py, .pyc) files
        location = os.path.abspath(__file__)
    except NameError:
        # This will work for bundled executables (e.g., created with PyInstaller)
        location = os.path.abspath(sys.executable)
    return location

def find_file(location, filename):
	for root, dirs, files in os.walk(location):
		if filename in files:
			return os.path.join(root, filename)
	return None


def widget_wrapper():
	from napari.qt.threading import thread_worker
	from torch import no_grad
	#import torch.multiprocessing as mp

	@thread_worker
	@no_grad()
	def run_dab_cellcount(image, reference, model_weights_path):
		logger.debug(f'Computing masks')

		# Getting reference image
		ref = get_reference_()
		reference_ = find_file(ref, 'reference.jpg')
		reference_ = np.asarray(Image.open(reference_))

		if torch.cuda.is_available():
			device = 'gpu'
			mask = pred(image, model_weights_path,reference_,device)
		else:
			mask = pred(image, model_weights_path,reference_)
		return mask

	@magicgui(
		call_button='Run segmentation',
		layout='vertical',
		compute_counts_button=dict(widget_type='PushButton', text='Recompute counts with edited annotations', enabled=False),
		clear_previous_segmentations=dict(widget_type='CheckBox', text='Clear previous results', value=True),
		output_outlines=dict(widget_type='CheckBox', text='Output outlines', value=True),
		load_model_button=dict(widget_type='PushButton', text='Load Model Weights')
	)
	def widget(
		viewer: Viewer,
		image_layer: Image,
		shape_layer: Shapes,
		compute_counts_button,
		clear_previous_segmentations,
		output_outlines,
		load_model_button  # Button to load model weights
	) -> None:
		if not hasattr(widget, 'cc_layers'):
			widget.cc_layers = []
		
		model_weights_path = ''

		def load_model():
			nonlocal model_weights_path
			file_path, _ = QFileDialog.getOpenFileName(result_container, "Select Model Weights", "", "All Files (*.*)")
			if file_path:
				logger.info(f"Model weights loaded from {file_path}")
				model_weights_path = file_path

		load_model_button.clicked.connect(load_model)

		result_container = QWidget()
		result_layout = QVBoxLayout(result_container)
		result_label = QLabel("RESULT")
		result_layout.addWidget(result_label)

		if clear_previous_segmentations:
			layer_names = [layer.name for layer in viewer.layers]
			for layer_name in layer_names:
				if any(cc_string in layer_name for cc_string in cc_strings):
					viewer.layers.remove(viewer.layers[layer_name])
			widget.cc_layers = []
		def _new_layers(masks, roi_key):
			from ._utils import masks_to_outlines
			import cv2
			outlines = masks_to_outlines(masks) * masks
			masks = np.expand_dims(masks, axis=1)
			outlines = np.expand_dims(outlines, axis=1)
			widget.masks_orig = masks
			widget.iseg = '_' + '%03d' % len(widget.cc_layers)
			layers = []

			# Get physical scale (...ZYX)
			physical_scale = image_layer.scale

			if output_outlines.value:
				layer_name = f"{image_layer.name}_cc_outlines{widget.iseg}_{roi_key}"
				layers.append(viewer.add_labels(outlines, name=layer_name, visible=False, scale=physical_scale))
			layer_name = f"{image_layer.name}_cc_masks{widget.iseg}_{roi_key}"
			layers.append(viewer.add_labels(masks, name=layer_name, visible=False, scale=physical_scale))
			widget.cc_layers.append(layers)

		roi_results = {}

		def process_rois(reference, update=False):
			nonlocal model_weights_path
			if not model_weights_path:
				logger.warning("Model weights not loaded. Please load the model weights.")
				return

			if not update:
				for i in reversed(range(result_layout.count())):
					result_layout.itemAt(i).widget().setParent(None)

			for roi in shape_layer.data:
				roi_key = tuple(map(tuple, roi))
				if update and roi_key not in roi_results:
					continue

				min_row, min_col, max_row, max_col = map(int, roi.flatten())
				roi_image = image_layer.data[min_row:max_row, min_col:max_col]

				mask_worker = run_dab_cellcount(roi_image, reference, model_weights_path)
				mask_worker.returned.connect(lambda mask, rk=roi_key: _update_roi_result(mask, rk))
				mask_worker.start()

				del roi_image
				gc.collect()

		def _update_roi_result(mask, roi_key):
			labeled_mask, num_cells = label(mask)
			result_label = roi_results.get(roi_key, QLabel())
			result_label.setText(f"ROI {shape_layer.data.index(roi) + 1}: {num_cells}")
			if roi_key not in roi_results:
				result_layout.addWidget(result_label)
				roi_results[roi_key] = result_label
			_new_layers(mask, roi_key)

			# Clear mask and labeled_mask to free memory
			del mask, labeled_mask
			gc.collect()

		# Function to update segmentation and count for selected ROIs
		def update_selected_rois(reference):
			process_rois(reference, update=True)

		# Connect the recompute counts button to the update function
		widget.compute_counts_button.clicked.connect(update_selected_rois(reference))

		# Function to handle changes in the Shapes layer
		@shape_layer.events.data.connect
		def on_shape_change(event):
			if not update:
				for i in reversed(range(result_layout.count())):
					result_layout.itemAt(i).widget().setParent(None)
				roi_results.clear()

		# Connect the Run Segmentation button to process_rois function
		widget.call_button.clicked.connect(lambda: process_rois(reference, update=False))

		# Add result_container to the widget
		widget.native.layout().addWidget(result_container)

		def update_masks(masks):
			from ._utils import masks_to_outlines

			outlines = masks_to_outlines(masks) * masks
			masks = np.expand_dims(masks, axis=1)
			outlines = np.expand_dims(outlines, axis=1)
			widget.viewer.value.layers[widget.image_layer.value.name + '_cc_masks' + widget.iseg].data = masks
			outline_str = widget.image_layer.value.name + '_cc_outlines' + widget.iseg
			if outline_str in widget.viewer.value.layers:
				widget.viewer.value.layers[outline_str].data = outlines
			widget.masks_orig = masks
			logger.debug('Masks updated')

	return widget

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
	return widget_wrapper, {'name': 'DAB SN Cell Counter'}
