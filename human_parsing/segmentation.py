from enum import Enum
from typing import Tuple

from albumentations import Compose, SmallestMaxSize, CenterCrop, Rotate, HorizontalFlip, BasicTransform
from pietoolbelt.tta import HFlipTTA, CLAHETTA, VFlipTTA, RotateTTA
from torchvision.models.utils import load_state_dict_from_url
import os
import numpy as np

from pietoolbelt.steps.segmentation.inference import SegmentationInference

__all__ = ['Segmentation']


class Segmentation:
    """
    Person whole stature segmentation

    Args:
        accuracy_lvl (Level): level of prediction accuracy. !WARNING! Performance decaying with increasing level of accuracy

    Accuracy levels:
        LEVEL_0 - maximum performance, minimum accuracy

        LEVEL_1 - medium performance, medium accuracy

        LEVEL_2 - low performance, maximum accuracy
    """

    class Level(Enum):
        LEVEL_0 = 1,
        LEVEL_1 = 2,
        LEVEL_2 = 3

    def __init__(self, accuracy_lvl: Level = Level.LEVEL_1):
        self._model = load_state_dict_from_url("https://drive.google.com/file/d/1FmVmoV3p4CBBrKOUsxfaovZz7Ir62FnG/view?usp=sharing",
                                               model_dir=os.path.join(os.path.expanduser("~"), '.human_parsing', 'segmentation'))

        target_transform = Compose([Rotate(limit=(-90, -90), p=1), HorizontalFlip(p=1)])

        self._inference = SegmentationInference(self._model.cuda()).set_target_transform(target_transform)

        if accuracy_lvl == Segmentation.Level.LEVEL_1:
            self._inference.set_tta([RotateTTA(angle_range=(107, 107)), RotateTTA(angle_range=(34, 34)),
                                     RotateTTA(angle_range=(63, 63))])
        elif accuracy_lvl == Segmentation.Level.LEVEL_2:
            self._inference.set_tta([HFlipTTA(), RotateTTA(angle_range=(107, 107)), RotateTTA(angle_range=(34, 34)),
                                     RotateTTA(angle_range=(63, 63))])

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process one image

        Args:
            image (np.ndarray): image in RGB color scheme, and in a shape: (H, W, C)

        Return:
            tuple of [transformed image, prediction result]
        """
        return self._inference.run_image(image)

    def set_device(self, device: str) -> 'Segmentation':
        """
        Set target device

        Args:
            device (str): target device ('cpu', 'cuda', 'cuda:0', e.t.c)

        Return:
            self instance
        """
        self._inference.set_device(device)
        return self
