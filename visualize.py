"""
Every demo from the workshop, one flag each. Follows the stations in the
browser lab (web/index.html), so you can show the "real code" version of
whatever the room just played with.

    python visualize.py --noise        # 1  Gaussian noise: image + histogram
    python visualize.py --forward      # 2  image -> noise, schedule, histograms
    python visualize.py --denoise      # 3  one U-Net pass: eps_hat and x0_hat at many t
    python visualize.py --generate     # 4  noise -> image (shows x_t AND x0_hat)
    python visualize.py --represent    # 5  t-SNE of U-Net features + linear probe
    python visualize.py --sdedit       # 5b repair a corrupted digit (img2img)
    python visualize.py --guidance     # 6  conditional model, sweep guidance scale w

--denoise, --generate, --represent and --sdedit need ddpm_mnist.pt (python train.py).
--guidance needs ddpm_mnist_conditional.pt (python train_conditional.py).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion import (CONDITIONAL_MODEL_PATH, MODEL_PATH, NULL_LABEL, get_conditional_model,
                       get_device, get_mnist, get_model, get_scheduler, to_picture)


def load(conditional=False):
    device = get_device()
    model = get_conditional_model() if conditional else get_model()
    model.load_state_dict(torch.load(CONDITIONAL_MODEL_PATH if conditional else MODEL_PATH, map_location=device))
    return model.to(device).eval(), device


def show_row(axes, images, titles):
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(to_picture(im), cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")


# ---------------------------------------------------------------- 1. noise
def show_noise():
    # Gaussian noise: every pixel is an independent draw from N(0, 1).
    noise = torch.randn(1, 1, 28, 28)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1, 2]})
    a.imshow(noise[0, 0], cmap="gray", vmin=-2.5, vmax=2.5)
    a.set_title(f"mean {noise.mean():.3f}, std {noise.std():.3f}")
    a.axis("off")
    xs = np.linspace(-4, 4, 200)
    b.hist(noise.flatten().numpy(), bins=40, density=True, color="gray")
    b.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color="tab:orange", lw=2, label="N(0, 1)")
    b.legend()
    b.set_title("pixel values of the noise image")
    fig.suptitle("Gaussian noise")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 2. forward
def show_forward_diffusion():
    scheduler = get_scheduler()

    # Take one clean MNIST digit: shape (1, 1, 28, 28), values in [-1, 1].
    image, label = get_mnist()[0]
    image = image.unsqueeze(0)

    # ONE noise tensor, reused for every timestep, so you see the same
    # image getting more and more buried under the same noise.
    noise = torch.randn_like(image)

    timesteps = [0, 100, 300, 500, 700, 900, 999]
    fig, axes = plt.subplots(2, len(timesteps), figsize=(2 * len(timesteps), 4.5))

    for col, t in enumerate(timesteps):
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        picture = image if t == 0 else scheduler.add_noise(image, noise, torch.tensor([t]))
        a = scheduler.alphas_cumprod[t].sqrt().item()
        show_row([axes[0, col]], [picture], [f"t = {t}\n{a:.2f}·x0 + {np.sqrt(1 - a * a):.2f}·eps"])
        # Pixel histogram: two spikes (background/ink) melt into a bell curve.
        axes[1, col].hist(picture.flatten().numpy(), bins=30, range=(-3.5, 3.5), color="tab:blue")
        axes[1, col].set_yticks([])

    fig.suptitle("Forward diffusion: image  ->  noise   (bottom: pixel histograms)")
    plt.tight_layout()

    # The noise schedule itself.
    ab = scheduler.alphas_cumprod
    plt.figure(figsize=(6, 3))
    plt.plot(ab.sqrt(), label="sqrt(alpha_bar_t)  signal kept")
    plt.plot((1 - ab).sqrt(), label="sqrt(1 - alpha_bar_t)  noise added")
    plt.xlabel("timestep t")
    plt.legend()
    plt.title("Linear beta schedule")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 3. denoise
@torch.no_grad()
def show_denoise():
    # One U-Net pass at several noise levels. Shows what the model THINKS the
    # clean image is:   x0_hat = (x_t - sqrt(1 - a_bar) * eps_hat) / sqrt(a_bar)
    model, device = load()
    scheduler = get_scheduler()
    image = get_mnist()[0][0].unsqueeze(0).to(device)
    noise = torch.randn_like(image)
    timesteps = [50, 200, 350, 500, 650, 800, 950]

    fig, axes = plt.subplots(3, len(timesteps), figsize=(2 * len(timesteps), 6.5))
    for col, t in enumerate(timesteps):
        tt = torch.tensor([t], device=device)
        noisy = scheduler.add_noise(image, noise, tt)
        eps_hat = model(noisy, tt).sample
        a = scheduler.alphas_cumprod[t].to(device)
        x0_hat = ((noisy - (1 - a).sqrt() * eps_hat) / a.sqrt()).clamp(-1, 1)
        loss = torch.mean((eps_hat - noise) ** 2).item()
        show_row([axes[0, col]], [noisy], [f"x_t, t={t}"])
        axes[1, col].imshow((eps_hat - noise).abs()[0, 0].cpu(), cmap="gray", vmin=0, vmax=2)
        axes[1, col].set_title(f"|eps_hat - eps|\nmse {loss:.3f}", fontsize=9)
        axes[1, col].axis("off")
        show_row([axes[2, col]], [x0_hat], ["x0_hat"])
    fig.suptitle("Denoising: the U-Net predicts the noise; subtracting it gives x0_hat")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 4. generate
@torch.no_grad()
def show_generation(num_digits=4, num_steps=50):
    model, device = load()
    scheduler = get_scheduler()

    # Use 50 denoising steps instead of all 1000 (much faster, still works).
    scheduler.set_timesteps(num_steps)

    # Start from PURE random noise. No digit is hidden in here.
    image = torch.randn(num_digits, 1, 28, 28, device=device)

    # Remember a few snapshots: (title, x_t, x0_hat)
    snapshots = [("Random noise", image, image)]

    for i, timestep in enumerate(scheduler.timesteps):

        # 1. The U-Net guesses the noise in the current image.
        noise_prediction = model(image, timestep).sample

        # 2. The scheduler removes a bit of that predicted noise.
        out = scheduler.step(noise_prediction, timestep, image)
        image = out.prev_sample

        steps_left = num_steps - (i + 1)
        if steps_left in (45, 40, 30, 20, 10):
            snapshots.append((f"{steps_left} steps left", image, out.pred_original_sample))

    snapshots.append(("Generated", image, image))

    # Top half: x_t (what is being denoised). Bottom half: x0_hat (the model's
    # current guess of the final image). The digit is decided early.
    fig, axes = plt.subplots(2 * num_digits, len(snapshots), figsize=(1.8 * len(snapshots), 3.6 * num_digits),
                             squeeze=False)
    for col, (title, batch, guess) in enumerate(snapshots):
        for row in range(num_digits):
            show_row([axes[row, col]], [batch[row:row + 1]], [title if row == 0 else ""])
            show_row([axes[num_digits + row, col]], [guess[row:row + 1]], ["x0_hat" if row == 0 else ""])

    fig.suptitle("Reverse diffusion: noise  ->  image   (top: x_t, bottom: model's guess x0_hat)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 5. represent
@torch.no_grad()
def mid_features(model, images, t, device):
    # Grab the U-Net bottleneck (mid-block output, 64 x 7 x 7) with a forward hook.
    scheduler = get_scheduler()
    feats = []
    hook = model.mid_block.register_forward_hook(lambda m, i, o: feats.append(o.mean((2, 3)).cpu()))
    for i in range(0, len(images), 500):
        x = images[i:i + 500].to(device)
        tt = torch.full((len(x),), t, device=device)
        model(scheduler.add_noise(x, torch.randn_like(x), tt), tt)
    hook.remove()
    return torch.cat(feats).numpy()


def show_representations(n_train=5000, n_test=2000, t=50):
    # The model never saw a label. Do its features know which digit is which?
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from torchvision import datasets, transforms

    from diffusion import DATA_DIR

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train, test = get_mnist(), datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    xtr = torch.stack([train[i][0] for i in range(n_train)]); ytr = np.array([train[i][1] for i in range(n_train)])
    xte = torch.stack([test[i][0] for i in range(n_test)]); yte = np.array([test[i][1] for i in range(n_test)])

    trained, device = load()
    untrained = get_model().to(device).eval()   # same architecture, random weights

    results = {"raw pixels (784)": (xtr.flatten(1).numpy(), xte.flatten(1).numpy())}
    for name, m in [("untrained U-Net (64)", untrained), ("trained U-Net (64)", trained)]:
        results[name] = (mid_features(m, xtr, t, device), mid_features(m, xte, t, device))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, (ftr, fte)) in zip(axes, results.items()):
        acc = LogisticRegression(max_iter=2000).fit(ftr, ytr).score(fte, yte)
        print(f"linear probe  {name:24s} {acc:.3f}")
        f = (fte - fte.mean(0)) / (fte.std(0) + 1e-6)
        z = TSNE(2, perplexity=35, init="pca", random_state=0).fit_transform(f[:1500])
        sc = ax.scatter(z[:, 0], z[:, 1], c=yte[:1500], cmap="tab10", s=4)
        ax.set_title(f"{name}\nlinear probe {acc:.1%}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, ticks=range(10))
    fig.suptitle(f"What the denoiser learned without labels (features at t = {t})")
    plt.show()


# ---------------------------------------------------------------- 5b. sdedit
@torch.no_grad()
def show_sdedit(strengths=(200, 400, 600, 800)):
    # SDEdit / img2img: corrupt a digit, noise it to t_start, then denoise.
    # Small t_start keeps your image; large t_start lets the model redraw it.
    model, device = load()
    scheduler = get_scheduler()
    scheduler.set_timesteps(50)
    image = get_mnist()[4][0].unsqueeze(0).to(device)
    corrupted = image.clone()
    corrupted[..., 10:18, :] = -1                       # erase a horizontal band

    fig, axes = plt.subplots(1, len(strengths) + 2, figsize=(2 * (len(strengths) + 2), 2.6))
    show_row(axes[:2], [image, corrupted], ["original", "corrupted input"])
    for ax, t_start in zip(axes[2:], strengths):
        tt = torch.tensor([t_start], device=device)
        x = scheduler.add_noise(corrupted, torch.randn_like(corrupted), tt)
        for t in scheduler.timesteps[scheduler.timesteps <= t_start]:
            x = scheduler.step(model(x, t).sample, t, x).prev_sample
        show_row([ax], [x], [f"strength t={t_start}"])
    fig.suptitle("SDEdit: noise partway, then denoise (= img2img 'strength')")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 6. guidance
@torch.no_grad()
def show_guidance(digit=3, scales=(0.0, 1.0, 3.0, 7.5), n=6, num_steps=50):
    # Classifier-free guidance:  eps = eps_uncond + w * (eps_cond - eps_uncond)
    model, device = load(conditional=True)
    fig, axes = plt.subplots(len(scales), n, figsize=(1.6 * n, 1.8 * len(scales)), squeeze=False)
    for row, w in enumerate(scales):
        scheduler = get_scheduler()
        scheduler.set_timesteps(num_steps)
        torch.manual_seed(0)                                  # same noise for every w
        x = torch.randn(n, 1, 28, 28, device=device)
        cond = torch.full((n,), digit, device=device)
        uncond = torch.full((n,), NULL_LABEL, device=device)
        for t in scheduler.timesteps:
            eps = model(torch.cat([x, x]), t, class_labels=torch.cat([cond, uncond])).sample
            eps_c, eps_u = eps.chunk(2)
            x = scheduler.step(eps_u + w * (eps_c - eps_u), t, x).prev_sample
        for col in range(n):
            show_row([axes[row, col]], [x[col:col + 1]], [f"w = {w}" if col == 0 else ""])
    fig.suptitle(f'Prompt "{digit}" at different guidance scales')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demos = {
        "noise": show_noise, "forward": show_forward_diffusion, "denoise": show_denoise,
        "generate": show_generation, "represent": show_representations, "sdedit": show_sdedit,
        "guidance": show_guidance,
    }
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in demos:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    chosen = [f for name, f in demos.items() if getattr(args, name)]
    if not chosen:
        parser.print_help()
    for f in chosen:
        f()
