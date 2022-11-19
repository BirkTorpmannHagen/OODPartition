import torch
from domain_datasets import build_dataset
from torch.utils.data import DataLoader
from vae.models.vanilla_vae import VanillaVAE
from vae.vae_experiment import VAEXperiment
from torchvision.transforms import Resize
import yaml

class OODSplitter():
    def __init__(self, dataloader, split=(80, 10, 10), vae_dim=128):
        assert len(split)==3, "split must be threefold, [%training data, %validaation data, %test data]"
        latents = torch.zeros([len(dataloader), vae_dim])
        vae_model = VanillaVAE(3, 128).to("cuda")
        config = yaml.safe_load("vae/configs/vae.yaml")
        vae_exp = VAEXperiment(vae_model, config)
        vae_exp.load_state_dict(torch.load("/home/birk/BatchDiversitySampling/logs/VanillaVAE/version_0/checkpoints/last.ckpt")["state_dict"])
        for i, (x, y) in enumerate(dataloader):
            with torch.no_grad():
                latents[i]=vae_model(x.to("cuda"))


    def get_trainloader(self):
        pass

    def get_valloader(self):
        pass

    def get_testloader(self):
        pass

if __name__ == '__main__':

    train_set = build_dataset(1, "datasets/NICO++", 0, lambda x: Resize(512)(x), lambda x: Resize(512)(x),0)[0]
    splitter = OODSplitter(DataLoader(train_set))