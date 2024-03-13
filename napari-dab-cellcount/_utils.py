import cv2
import numpy as np
from scipy.ndimage import find_objects

def masks_to_outlines(masks):
    """Get outlines of masks."""

    if masks.ndim != 2:
        raise ValueError("masks_to_outlines takes 2D array, not %dD array" % masks.ndim)
    outlines = np.zeros(masks.shape, bool)

    slices = find_objects(masks.astype(int))
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
            vr, vc = pvr + sr.start, pvc + sc.start
            outlines[vr, vc] = 1
    return outlines
