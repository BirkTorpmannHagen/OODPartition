from pytorch_lightning_gans.models import WGAN
from pytorch_lightning.trainer import Trainer
from lightning_dataset import VAEDataset
def train_wgan():
    model = WGAN(latent_dim=512, batch_size=16)
    trainer =   Trainer(gpus=[0], max_epochs=1000)
    dataset = VAEDataset("datasets/NICO++", train_batch_size=16, val_batch_size=16, patch_size=(256,256), num_workers=4)
    dataset.setup()
    trainer.validate(model, dataloaders=dataset.train_dataloader(), ckpt_path="lightning_logs/version_0/checkpoints/epoch=134-step=749925.ckpt")

if __name__ == '__main__':
    train_wgan()