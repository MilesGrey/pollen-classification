from typing import List, Optional, Type, Dict

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torch import tensor, Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, Metric, F1Score
from torchvision.transforms import ToTensor, Compose, Resize, RandomRotation, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomResizedCrop, RandomApply, Normalize

from src.data.load_augsburg import AugsburgClassificationDataset, Mode
from src.data.augmentation import SquarePad, AdditiveWhiteGaussianNoise


class Classifier(LightningModule):

    # Use these value for normalization of ImageNet-pretrained networks.
    IMAGE_NET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGE_NET_STANDARD_DEVIATION: List[float] = [0.229, 0.224, 0.225]

    def __init__(
            self,
            batch_size: int,
            dataset: Type[AugsburgClassificationDataset]
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.dataset = dataset

        class_weights = tensor(dataset.CLASS_WEIGHTS, device=self.device)
        self.calculate_loss = CrossEntropyLoss(weight=class_weights)

        # TODO: Add confusion matrix logging or maybe just do once in the end?
        validation_metrics: Dict[str, Metric] = {
            'validation_precision': Precision(num_classes=dataset.NUM_CLASSES, average='macro'),
            'validation_recall': Recall(num_classes=dataset.NUM_CLASSES, average='macro'),
            'validation_f1': F1Score(num_classes=dataset.NUM_CLASSES, average='macro'),
        }
        test_metrics: Dict[str, Metric] = {
            'test_precision': Precision(num_classes=dataset.NUM_CLASSES, average='macro'),
            'test_recall': Recall(num_classes=dataset.NUM_CLASSES, average='macro'),
            'test_f1': F1Score(num_classes=dataset.NUM_CLASSES, average='macro'),
        }
        self.metrics: Dict[Mode, Dict[str, Metric]] = {
            Mode.VALIDATION: validation_metrics,
            Mode.TEST: test_metrics
        }

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch

        predictions = self(images)
        loss = self.calculate_loss(predictions, targets)
        self.log('train_loss', loss, on_step=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        images, targets = batch
        predictions = self(images)
        return self._calculate_metrics(predictions, targets, Mode.VALIDATION)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        images, targets = batch
        predictions = self(images)
        return self._calculate_metrics(predictions, targets, Mode.TEST)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def _build_data_augmentation_pipeline(self):
        return RandomApply(
            [
                RandomResizedCrop(
                    size=self.dataset.IMAGE_SIZE,
                    scale=(0.7, 0.9),
                    ratio=(1., 1.),
                ),
                RandomRotation(self.dataset.AUGMENTATION_PARAMETERS['rotation_range']),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                AdditiveWhiteGaussianNoise(
                    mean=self.dataset.AUGMENTATION_PARAMETERS['noise_mean'],
                    standard_deviation=self.dataset.AUGMENTATION_PARAMETERS['noise_standard_deviation']
                )
            ],
            p=self.dataset.AUGMENTATION_PARAMETERS['application_probability']
        )

    def _calculate_metrics(self, predictions, targets, mode):
        metric_results: Dict[str, Tensor] = {}
        for name, metric in self.metrics[mode].items():
            result = metric(predictions, targets)
            self.log(name, result, on_epoch=True, batch_size=self.batch_size)
            metric_results[name] = result
        return metric_results

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = self.dataset(
            mode=Mode.TRAIN,
            transforms=Compose([
                ToTensor(),
                SquarePad(),
                Resize(self.dataset.IMAGE_SIZE),
                self._build_data_augmentation_pipeline(),
                Normalize(mean=self.IMAGE_NET_MEAN, std=self.IMAGE_NET_STANDARD_DEVIATION)
            ])
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        validation_dataset = self.dataset(
            mode=Mode.VALIDATION,
            transforms=Compose([
                ToTensor(),
                SquarePad(),
                Resize((256, 256)),
                Normalize(mean=self.IMAGE_NET_MEAN, std=self.IMAGE_NET_STANDARD_DEVIATION)
            ])
        )
        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=4
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
