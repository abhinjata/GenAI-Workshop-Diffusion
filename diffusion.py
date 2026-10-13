"""
Core DDPM pieces, shared by train.py and visualize.py.

A DDPM has only two ingredients:

    1. A SCHEDULER  -> knows how much noise to add at each timestep t
                       (forward process) and how to remove predicted noise
                       (reverse process).

    2. A U-NET      -> looks at a noisy image + timestep t
                       and predicts the noise that was added.
"""

import os

import torch
from torchvision import datasets, transforms
from diffusers import DDPMScheduler, UNet2DModel


# Keep files next to this script, no matter which folder you run it from.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "ddpm_mnist.pt")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
NUM_TRAIN_TIMESTEPS = 1000  # T: after 1000 noising steps an image is pure noise


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_scheduler():
    # The scheduler holds the "noise schedule": for every timestep t it knows
    # how much of the original image is kept and how much noise is mixed in.
    #   t = 0    -> almost the clean image
    #   t = 999  -> almost pure Gaussian noise
    return DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)


def get_model():
    # A tiny U-Net.
    #   Input : noisy image (1 x 28 x 28) + timestep t
    #   Output: predicted noise (1 x 28 x 28)  <- same shape as the input!
    #
    # The image is shrunk 28 -> 14 -> 7 on the way down and grown back
    # 7 -> 14 -> 28 on the way up. Kept small (no attention) so it trains
    # in a few minutes on a laptop CPU.
    return UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(16, 32, 64),
        norm_num_groups=8,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


CONDITIONAL_MODEL_PATH = os.path.join(PROJECT_DIR, "ddpm_mnist_conditional.pt")
NULL_LABEL = 10  # the "empty prompt": label 10 means "no digit specified"


def get_conditional_model():
    # Same tiny U-Net, plus a learned embedding for the class label.
    # The label embedding is ADDED to the timestep embedding, so every
    # ResNet block gets told both "how noisy is this?" and "which digit?".
    #
    # This is a miniature of text-to-image: in Stable Diffusion the label
    # is replaced by a CLIP text embedding, injected through cross-attention.
    #
    # 11 classes = digits 0-9 + one NULL class (the "empty prompt") that
    # classifier-free guidance needs.
    return UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(16, 32, 64),
        norm_num_groups=8,
        num_class_embeds=11,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def get_mnist():
    # MNIST pixels are in [0, 1]. Diffusion works with images in [-1, 1],
    # the same range as Gaussian noise, so we rescale with Normalize(0.5, 0.5).
    transform = transforms.Compose([
        transforms.ToTensor(),                 # [0, 255] -> [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # [0, 1]   -> [-1, 1]
    ])
    return datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)


def to_picture(image):
    # Turn a (1, 1, 28, 28) tensor in [-1, 1] into a 28 x 28 array in [0, 1]
    # so matplotlib can show it.
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    return image[0, 0].detach().cpu().numpy()
