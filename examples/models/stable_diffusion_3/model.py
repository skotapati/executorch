import torch
from diffusers import StableDiffusion3Pipeline

from ..model_base import EagerModelBase

class StableDiffusion3Medium(EagerModelBase):

    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        print("Loading model from HuggingFace")

        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                        torch_dtype=torch.float32)
        return pipe.transformer

    def get_example_inputs(self):
        return (
            torch.randn(1, 16, 154, 154, dtype=torch.float32, device="cpu"),
            torch.randn(1, 1, 4096, dtype=torch.float32, device="cpu"),
            torch.randn(1, 2048, dtype=torch.float32, device="cpu"),
            torch.randn(1, dtype=torch.float32, device="cpu")
        )