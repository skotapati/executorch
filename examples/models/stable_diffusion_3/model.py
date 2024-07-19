import torch
from diffusers import DiffusionPipeline, ModelMixin

from ..model_base import EagerModelBase

class StableDiffusion3Medium(EagerModelBase):

    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        print("Loading model from HuggingFace")

        model_id = "stabilityai/stable-diffusion-3-medium"
        device = "mps"

        sd_model = ModelMixin.from_pretrained(model_id, device_map=device)

        return sd_model