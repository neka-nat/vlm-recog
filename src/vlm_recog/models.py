import numpy as np
from pydantic import BaseModel, ConfigDict


class DetectedItem(BaseModel):
    box_2d: tuple[int, int, int, int]
    segmentation_mask: np.ndarray | None
    label: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


DetectedItems = list[DetectedItem]
