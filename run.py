import subprocess

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.load_augsburg import Augsburg15ClassificationDataset
from src.models.timm_classifier import TimmClassifier, TimmModel


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


if __name__ == '__main__':
    model = TimmClassifier(
        model=TimmModel.MOBILE_NET_V3,
        batch_size=32,
        dataset=Augsburg15ClassificationDataset
    )

    logger = TensorBoardLogger('logs', f'soft_teacher#{get_git_revision_short_hash()}')
    trainer = Trainer(max_epochs=40, logger=logger)
    trainer.fit(model, model.train_dataloader(), model.val_dataloader())
    # CKPT_PATH = './lightning_logs/soft_teacher_loss_weights#25b59bef/checkpoints/epoch=15-step=11807.ckpt'
    # test = trainer.validate(model, model.val_dataloader(), CKPT_PATH)
    # print()
