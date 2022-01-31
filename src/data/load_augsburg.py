from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Union

from pandas import read_csv, DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image


class Mode(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


class AugsburgClassificationDataset(Dataset, ABC):
    """
    Augsburg pollen classification datasets.
    """
    DATASET_DIRECTORY: Path = Path(__file__).parents[2] / Path('datasets/augsburg')
    AUGMENTATION_PARAMETERS = {
        'crop_scale': (0.7, 0.9),
        'crop_ratio': (1., 1.),
        'rotation_range': 45,
        'noise_standard_deviation': 15.,
        'noise_mean': 0.,
        'application_probability': 0.25
    }
    IMAGE_INFO_CSVS: Dict[Mode, Path]
    IMAGE_SIZE: Tuple[int, int]
    CLASS_MAPPING: Dict[str, int]
    INVERSE_CLASS_MAPPING: List[str]
    CLASS_WEIGHTS: List[float]
    NUM_CLASSES: int

    def __init__(
            self,
            mode: Mode,
            transforms: Callable = None
    ) -> None:
        self.transforms = transforms
        self.image_info_csv = self.IMAGE_INFO_CSVS[mode]
        self.image_info = self._parse_image_info_csv()

    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, idx) -> List[Union[Tensor, int]]:
        image_file = self.DATASET_DIRECTORY / self.image_info['file'][idx]
        image = Image.open(image_file).convert('RGB')
        target = self.image_info['label'][idx]

        if self.transforms is not None:
            image = self.transforms(image)

        return [image, target]

    def _parse_image_info_csv(self) -> DataFrame:
        image_info_file = self.DATASET_DIRECTORY / self.image_info_csv
        image_info = read_csv(image_info_file, header=None)
        image_info.columns = ['file', 'label']
        image_info = image_info.reset_index()
        return image_info


class Augsburg15ClassificationDataset(AugsburgClassificationDataset):
    IMAGE_INFO_CSVS: Dict[Mode, Path] = {
        Mode.TRAIN: Path('original_15_traindata.csv'),
        Mode.VALIDATION: Path('original_15_valdata.csv'),
        Mode.TEST: Path('original_15_testdata.csv')
    }
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    CLASS_MAPPING: Dict[str, int] = {
        'Alnus': 0,
        'Betula': 1,
        'Carpinus': 2,
        'Corylus': 3,
        'Fagus': 4,
        'Fraxinus': 5,
        'Plantago': 6,
        'Poaceae': 7,
        'Populus': 8,
        'Quercus': 9,
        'Salix': 10,
        'Taxus': 11,
        'Tilia': 12,
        'Ulmus': 13,
        'Urticaceae': 14
    }
    INVERSE_CLASS_MAPPING: List[str] = [
        'Alnus',
        'Betula',
        'Carpinus',
        'Corylus',
        'Fagus',
        'Fraxinus',
        'Plantago',
        'Poaceae',
        'Populus',
        'Quercus',
        'Salix',
        'Taxus',
        'Tilia',
        'Ulmus',
        'Urticaceae'
    ]
    CLASS_WEIGHTS: List[float] = [
        1 / 6040,
        1 / 1422,
        1 / 4806,
        1 / 7001,
        1 / 439,
        1 / 276,
        1 / 1034,
        1 / 2160,
        1 / 1239,
        1 / 368,
        1 / 317,
        1 / 3647,
        1 / 108,
        1 / 205,
        1 / 1699
    ]
    NUM_CLASSES = 15


class Augsburg31ClassificationDataset(AugsburgClassificationDataset):
    IMAGE_INFO_CSVS: Dict[Mode, Path] = {
        Mode.TRAIN: Path('original_31_traindata.csv'),
        Mode.VALIDATION: Path('original_31_valdata.csv'),
        Mode.TEST: Path('original_31_testdata.csv')
    }
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    CLASS_MAPPING: Dict[str, int] = {
        'Alnus': 0,
        'Apiaceae': 1,
        'Artemisia': 2,
        'Betula': 3,
        'Cannabaceae': 4,
        'Carpinus': 5,
        'Castanea': 6,
        'Chenopodiaceae': 7,
        'Corylus': 8,
        'Cupressaceae': 9,
        'Cyperaceae': 10,
        'Fagus': 11,
        'Fraxinus': 12,
        'Juglans': 13,
        'Larix': 14,
        'Papaveraceae': 15,
        'Picea': 16,
        'Pinaceae': 17,
        'Plantago': 18,
        'Platanus': 19,
        'Poaceae': 20,
        'Populus': 21,
        'Quercus': 22,
        'Rumex': 23,
        'Salix': 24,
        'Taxus': 25,
        'Tilia': 26,
        'Ulmus': 27,
        'Urticaceae': 28,
        'Spores': 29,
        'NoPollen': 30,
    }
    INVERSE_CLASS_MAPPING: List[str] = [
        'Alnus',
        'Apiaceae',
        'Artemisia',
        'Betula',
        'Cannabaceae',
        'Carpinus',
        'Castanea',
        'Chenopodiaceae',
        'Corylus',
        'Cupressaceae',
        'Cyperaceae',
        'Fagus',
        'Fraxinus',
        'Juglans',
        'Larix',
        'Papaveraceae',
        'Picea',
        'Pinaceae',
        'Plantago',
        'Platanus',
        'Poaceae',
        'Populus',
        'Quercus',
        'Rumex',
        'Salix',
        'Taxus',
        'Tilia',
        'Ulmus',
        'Urticaceae',
        'Spores',
        'NoPollen',
    ]
    CLASS_WEIGHTS: [float] = {
        1 / 10063,
        1 / 12,
        1 / 76,
        1 / 2399,
        1 / 12,
        1 / 8010,
        1 / 56,
        1 / 39,
        1 / 11668,
        1 / 867,
        1 / 30,
        1 / 728,
        1 / 460,
        1 / 110,
        1 / 42,
        1 / 25,
        1 / 23,
        1 / 450,
        1 / 1721,
        1 / 63,
        1 / 3600,
        1 / 2066,
        1 / 611,
        1 / 80,
        1 / 526,
        1 / 6077,
        1 / 181,
        1 / 339,
        1 / 2829,
        1 / 86,
        1 / 578
    }
    NUM_CLASSES = 31
