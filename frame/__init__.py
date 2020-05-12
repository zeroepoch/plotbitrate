__all__ = ["Frame"]

from importlib import util

# use the dataclass version of Frame if available.
# this will work on python >= 3.7 or 3.6 with dataclasses 
# backport installed
if util.find_spec("dataclasses") is not None:
    from ._frame_dataclass import Frame
else:
    from ._frame_class import Frame  # type: ignore
