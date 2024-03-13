try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._dock_widget import napari_cellcount_dock_widget
from ._sample_data import napari_provide_sample_data