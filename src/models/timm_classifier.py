from enum import Enum
from typing import Tuple, Type, Any

import timm

from src.data.load_augsburg import AugsburgClassificationDataset
from src.models.classifier import Classifier


class TimmModel(Enum):
    EFFICIENT_NET_V2 = 'tf_efficientnetv2_s_in21ft1k'
    MOBILE_NET_V3 = 'mobilenetv3_large_100_miil'
    RESNET_50 = 'resnet50'


class TimmClassifier(Classifier):

    def __init__(
            self,
            model: TimmModel,
            batch_size: int,
            dataset: Type[AugsburgClassificationDataset],
    ):
        super().__init__(batch_size, dataset)

        self.model = timm.create_model(
            model_name=model.value,
            pretrained=True,
            num_classes=dataset.NUM_CLASSES
        )

    def forward(self, images) -> Any:
        return self.model(images)
