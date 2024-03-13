try:
    from ._version import version as __version__
except ImportError:
    __version__ = 0.1.1

from ._dock_widget import napari_experimental_provide_dock_widget
from ._sample_data import napari_provide_sample_data
