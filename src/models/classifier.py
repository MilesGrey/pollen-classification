from typing import List, Type

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torch import tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score
from torchvision.transforms import ToTensor, Compose, Resize, RandomRotation, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomResizedCrop, RandomApply

from src.data.load_augsburg import AugsburgClassificationDataset, Mode
from src.data.augmentation import SquarePad


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

        self.class_weights = tensor(dataset.CLASS_WEIGHTS)
        self.calculate_loss = CrossEntropyLoss(weight=self.class_weights)

        # TODO: Add confusion matrix logging or maybe just do once in the end?
        self.train_precision = Precision(num_classes=dataset.NUM_CLASSES, average='macro')
        self.train_recall = Recall(num_classes=dataset.NUM_CLASSES, average='macro')
        self.train_f1_score = F1Score(num_classes=dataset.NUM_CLASSES, average='macro')

        self.validation_precision = Precision(num_classes=dataset.NUM_CLASSES, average='macro')
        self.validation_recall = Recall(num_classes=dataset.NUM_CLASSES, average='macro')
        self.validation_f1_score = F1Score(num_classes=dataset.NUM_CLASSES, average='macro')

        self.test_precision = Precision(num_classes=dataset.NUM_CLASSES, average='macro')
        self.test_recall = Recall(num_classes=dataset.NUM_CLASSES, average='macro')
        self.test_f1_score = F1Score(num_classes=dataset.NUM_CLASSES, average='macro')

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch

        predictions = self(images)
        loss = self.calculate_loss(predictions, targets)
        self.log('train_loss', loss, on_step=True, batch_size=self.batch_size)

        self.train_precision(predictions, targets)
        self.train_recall(predictions, targets)
        self.train_f1_score(predictions, targets)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1_score', self.train_f1_score, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images)
        self.validation_precision(predictions, targets)
        self.validation_recall(predictions, targets)
        self.validation_f1_score(predictions, targets)
        self.log('validation_precision', self.validation_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log('validation_recall', self.validation_recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log('validation_f1_score', self.validation_f1_score, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images)
        self.test_precision(predictions, targets)
        self.test_recall(predictions, targets)
        self.test_f1_score(predictions, targets)
        self.log('test_precision', self.test_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_recall', self.test_recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_f1_score', self.test_f1_score, on_step=True, on_epoch=True, prog_bar=True)

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
                # AdditiveWhiteGaussianNoise(
                #     mean=self.dataset.AUGMENTATION_PARAMETERS['noise_mean'],
                #     standard_deviation=self.dataset.AUGMENTATION_PARAMETERS['noise_standard_deviation']
                # )
            ],
            p=self.dataset.AUGMENTATION_PARAMETERS['application_probability']
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = self.dataset(
            mode=Mode.TRAIN,
            transforms=Compose([
                ToTensor(),
                SquarePad(),
                Resize(self.dataset.IMAGE_SIZE),
                self._build_data_augmentation_pipeline(),
            ])
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=4,
            prefetch_factor=4
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        validation_dataset = self.dataset(
            mode=Mode.VALIDATION,
            transforms=Compose([
                ToTensor(),
                SquarePad(),
                Resize(self.dataset.IMAGE_SIZE),
            ])
        )
        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=4,
            prefetch_factor=4
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
