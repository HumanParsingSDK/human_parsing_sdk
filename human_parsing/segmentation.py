from enum import Enum
from typing import Tuple

import torch
from albumentations import Compose, SmallestMaxSize, CenterCrop, Rotate, HorizontalFlip, BasicTransform
from pietoolbelt.models import ModelsContainer
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
        try:
            if accuracy_lvl == Segmentation.Level.LEVEL_2:
                self._model = ModelsContainer([
                    load_state_dict_from_url("https://github.com/HumanParsingSDK/weights/raw/master/segmentation/model0.pth",
                                             model_dir=os.path.join(os.path.expanduser("~"), '.human_parsing', 'segmentation')),
                    load_state_dict_from_url("https://github.com/HumanParsingSDK/weights/raw/master/segmentation/model1.pth",
                                             model_dir=os.path.join(os.path.expanduser("~"), '.human_parsing', 'segmentation')),
                    load_state_dict_from_url("https://github.com/HumanParsingSDK/weights/raw/master/segmentation/model2.pth",
                                             model_dir=os.path.join(os.path.expanduser("~"), '.human_parsing', 'segmentation'))
                ], reduction=lambda x: torch.mean(x, dim=0))
            else:
                self._model = load_state_dict_from_url("https://github.com/HumanParsingSDK/weights/raw/master/segmentation/model0.pth",
                                                       model_dir=os.path.join(os.path.expanduser("~"), '.human_parsing',
                                                                              'segmentation'))
        except Exception as err:
            raise RuntimeError(
                "Can't download weights. Check internet connection or try to update human_parsing sdk. Error message: [{}]".format(
                    err))

        target_transform = Compose([Rotate(limit=(-90, -90), p=1), HorizontalFlip(p=1)])
        self._inference = SegmentationInference(self._model).set_target_transform(target_transform)

        if accuracy_lvl == Segmentation.Level.LEVEL_1:
            self._inference.set_tta([HFlipTTA(), RotateTTA(angle_range=(107, 107)), RotateTTA(angle_range=(34, 34)),
                                     RotateTTA(angle_range=(63, 63))])
        elif accuracy_lvl == Segmentation.Level.LEVEL_2:
            self._inference.set_tta([HFlipTTA(), RotateTTA(angle_range=(-84, -84)), RotateTTA(angle_range=(69, 69)),
                                     RotateTTA(angle_range=(64, 64))])

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
