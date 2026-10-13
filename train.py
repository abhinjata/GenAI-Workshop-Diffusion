"""
Demo 2 - Train a tiny DDPM on MNIST.

    python train.py

The whole idea of training, in one sentence:

    Add random noise to a real image, and teach the U-Net
    to guess exactly which noise we added.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion import MODEL_PATH, get_device, get_mnist, get_model, get_scheduler


epochs = 10         # ~4 min per epoch on a laptop CPU; more epochs = cleaner digits
batch_size = 128
learning_rate = 1e-3

device = get_device()
print(f"Training on: {device}")

dataloader = DataLoader(get_mnist(), batch_size=batch_size, shuffle=True)
scheduler = get_scheduler()
model = get_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):

    for step, (images, _) in enumerate(dataloader):  # labels are ignored!

        images = images.to(device)

        # 1. Make some random Gaussian noise, same shape as the images.
        noise = torch.randn_like(images)

        # 2. Pick a random timestep for every image in the batch.
        #    Small t = a little noise, large t = a lot of noise.
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=device,
        )

        # 3. Forward diffusion: mix the image with the noise (amount set by t).
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # 4. Ask the U-Net: "which noise was added to this image?"
        predicted_noise = model(noisy_images, timesteps).sample

        # 5. Compare the guess with the noise we actually added.
        loss = F.mse_loss(predicted_noise, noise)

        # 6. Standard backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"epoch {epoch + 1}/{epochs}  step {step}/{len(dataloader)}  loss {loss.item():.4f}")

    # Save after every epoch, so you can stop training early (Ctrl+C)
    # and still generate digits with what has been learned so far.
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
