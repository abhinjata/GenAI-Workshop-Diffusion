"""
Demo 5 - Train a CLASS-CONDITIONAL DDPM with classifier-free guidance.

    python train_conditional.py

Identical to train.py except for two lines:

    * the U-Net also receives the digit label ("the prompt"), and
    * 10% of the time we hide the label (replace it with NULL_LABEL),
      so the same network also learns to generate WITHOUT a prompt.

Having both a conditional and an unconditional prediction from one network
is what makes classifier-free guidance (CFG) possible at sampling time:

    eps = eps_uncond + w * (eps_cond - eps_uncond)

Stable Diffusion's "guidance_scale" is exactly this w.
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion import (CONDITIONAL_MODEL_PATH, NULL_LABEL, get_conditional_model,
                       get_device, get_mnist, get_scheduler)

epochs = int(os.environ.get("EPOCHS", 6))  # ~4 min/epoch on a laptop CPU
batch_size = 128
learning_rate = 1e-3
label_dropout = 0.1  # how often we train with the "empty prompt"

device = get_device()
print(f"Training on: {device}")

dataloader = DataLoader(get_mnist(), batch_size=batch_size, shuffle=True)
scheduler = get_scheduler()
model = get_conditional_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):

    for step, (images, labels) in enumerate(dataloader):   # labels are USED now

        images, labels = images.to(device), labels.to(device)

        # Hide the label for a random 10% of the batch -> unconditional training.
        drop = torch.rand(labels.shape[0], device=device) < label_dropout
        labels = torch.where(drop, torch.full_like(labels, NULL_LABEL), labels)

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                  (images.shape[0],), device=device)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        predicted_noise = model(noisy_images, timesteps, class_labels=labels).sample
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"epoch {epoch + 1}/{epochs}  step {step}/{len(dataloader)}  loss {loss.item():.4f}",
                  flush=True)

    torch.save(model.state_dict(), CONDITIONAL_MODEL_PATH)
    print(f"Saved model to {CONDITIONAL_MODEL_PATH}", flush=True)
