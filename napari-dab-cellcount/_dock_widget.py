"""
dab-cellcount dock widget module
"""
from typing import Any
from napari_plugin_engine import napari_hook_implementation

import time
import numpy as np
import logging

from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
import sys

from pred import pred
import torch

# initialize logger
# use -v or --verbose when starting napari to increase verbosity
logger = logging.getLogger(__name__)
if '--verbose' in sys.argv or '-v' in sys.argv:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)

#@thread_worker
def read_logging(log_file, logwindow):
    with open(log_file, 'r') as thefile:
        #thefile.seek(0,2) # Go to the end of the file
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

def widget_wrapper():
    from napari.qt.threading import thread_worker
    try:
        from torch import no_grad
    except ImportError:
        def no_grad():
            def _deco(func):
                return func
            return _deco

    @thread_worker
    @no_grad()
    def run_dab_cellcount(image):
        logger.debug(f'computing masks')
        if torch.cuda.is_available():
            mask = pred(image, gpu)
        else:
            mask = pred(image, cpu)
        return mask

    @magicgui(
        call_button='run segmentation',  
        layout='vertical',
        compute_counts_button  = dict(widget_type='PushButton', text='recompute counts with edited annotations', enabled=False),
        clear_previous_segmentations = dict(widget_type='CheckBox', text='clear previous results', value=True),
        output_outlines = dict(widget_type='CheckBox', text='output outlines', value=True),
    )
    def widget(#label_logo, 
        viewer: Viewer,
        image_layer: Image,
        shape_layer: Shapes,
        compute_counts_button,
        clear_previous_segmentations,
        output_outlines
    ) -> None:
        # Import when users activate plugin
        if not hasattr(widget, 'cc_layers'):
            widget.cc_layers = []
        
        if clear_previous_segmentations:
            layer_names = [layer.name for layer in viewer.layers]
            for layer_name in layer_names:
                if any([cc_string in layer_name for cc_string in cc_strings]):
                    viewer.layers.remove(viewer.layers[layer_name])
            widget.cc_layers = []

        def _new_layers(masks):
            from utils import masks_to_outlines
            import cv2
            outlines = masks_to_outlines(masks) * masks
            masks = np.expand_dims(masks, axis=1)
            outlines = np.expand_dims(outlines, axis=1)
            widget.masks_orig = masks
            widget.iseg = '_' + '%03d'%len(widget.cc_layers)
            layers = []

            # get physical scale (...ZYX)
            physical_scale = image_layer.scale

            if widget.output_outlines.value:
                layers.append(viewer.add_labels(outlines, name=image_layer.name + '_cc_outlines' + widget.iseg, visible=False, scale=physical_scale))
            layers.append(viewer.add_labels(masks, name=image_layer.name + '_cc_masks' + widget.iseg, visible=False, scale=physical_scale))
            widget.cc_layers.append(layers)

        def _new_segmentation(segmentation):
            masks = segmentation
            try:
                _new_layers(masks)
                
                for layer in viewer.layers:
                    layer.visible = False
                viewer.layers[-1].visible = True
                image_layer.visible = True
                if not float(stitch_threshold_3D):
                    widget.compute_masks_button.enabled = True            
            except Exception as e:
                logger.error(e)
            widget.call_button.enabled = True
            
        image = image_layer.data

        cc_worker = run_dab_cellcount(image=image)
        cc_worker.returned.connect(_new_segmentation)
        cc_worker.start()


    def update_masks(masks):     
        from utils import masks_to_outlines

        outlines = masks_to_outlines(masks) * masks
        masks = np.expand_dims(masks, axis=1)
        outlines = np.expand_dims(outlines, axis=1)
        widget.viewer.value.layers[widget.image_layer.value.name + '_cc_masks' + widget.iseg].data = masks
        outline_str = widget.image_layer.value.name + '_cc_outlines' + widget.iseg
        if outline_str in widget.viewer.value.layers:
            widget.viewer.value.layers[outline_str].data = outlines
        widget.masks_orig = masks
        logger.debug('masks updated')


    @widget.compute_masks_button.changed.connect 
    def _compute_masks(e: Any):
        
        mask_worker = compute_masks(widget.masks_orig)
        mask_worker.returned.connect(update_masks)
        mask_worker.start()

    return widget            


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'DAB SN Cell Counter'}

