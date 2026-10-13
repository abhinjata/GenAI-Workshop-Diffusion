# Diffusion Lab: Enigma workshop kit

An activity-based introduction to diffusion models in seven stations. It covers noising, denoising, Gaussian noise, representation learning, conditioning and guidance, and ends with Stable Diffusion, leading into Prompt Pictionary.

There are two ways to run it:

* **Browser lab** (`web/index.html`, also published as an artifact). Students open one link. The trained U-Nets run live in the browser with TensorFlow.js; nothing to install.
* **Python demos** (this folder). Show the real PyTorch code behind each station on the projector.

## Session plan (~30 min, before Prompt Pictionary)

| # | Station | Concept | Activity | Python |
|---|---------|---------|----------|--------|
| 1 | Gaussian noise | pixels ~ N(0,1), histogram | **Spot the digit**: guess six digits under rising noise | `visualize.py --noise` |
| 2 | Noising | x_t = √ᾱ·x₀ + √(1−ᾱ)·ε, schedule | **Bury your own digit**: draw, find where it vanishes | `visualize.py --forward` |
| 3 | Denoising | U-Net predicts ε, MSE loss, x̂₀ | **Beat the U-Net**: room vs. model at t≈700 | `visualize.py --denoise` |
| 4 | Sampling | DDPM reverse loop, seeds, step count | **Same seed, same digit?** | `visualize.py --generate` |
| 5 | Representations | features learned without labels, SDEdit | **Find the clusters**, repair a scribble | `visualize.py --represent --sdedit` |
| 6 | Conditioning | label embedding, classifier-free guidance | **Guidance tug-of-war** | `visualize.py --guidance` |
| 7 | Stable Diffusion | latent space, CLIP, cross-attention, knobs | Prompt builder → Prompt Pictionary | `sd_demo.py --all` (GPU) |

## Files

| File | What it is |
|------|------------|
| `diffusion.py` | Scheduler, unconditional U-Net, **new** conditional U-Net (`num_class_embeds=11`, label 10 = empty prompt), MNIST loader |
| `train.py` | Unchanged: unconditional DDPM, labels ignored |
| `train_conditional.py` | **New.** Same loop, plus the label and 10% label dropout for CFG |
| `visualize.py` | **Expanded** from 2 demos to 7 (flags above) |
| `sd_demo.py` | **New.** Stable Diffusion 1.5 experiments mapped to each station; saves labelled grids to `sd_outputs/` |
| `train_toy2d.py` | **New.** 2D toy diffusion (particles → the word ENIGMA) used in station 4 |
| `export_web.py` | **New.** Writes trained weights to `web/` for the browser lab |
| `ddpm_mnist.pt`, `ddpm_mnist_conditional.pt` | Trained checkpoints (10 and 6 epochs) |
| `web/` | Browser lab: `index.html` + weights + precomputed t-SNE maps |

## Setup

```bash
pip install -r requirements.txt
python train.py                 # ~4 min/epoch on a laptop CPU (checkpoint already included)
python train_conditional.py     # same speed (checkpoint already included)
python visualize.py --forward
```

For Stable Diffusion, use a GPU runtime (Colab T4 is enough):

```bash
pip install diffusers transformers accelerate
python sd_demo.py --tokens --latents --guidance
```

To run the browser lab locally: `cd web && python -m http.server`, then open `localhost:8000`. It must be served over HTTP; opening the file directly blocks the weight downloads. After retraining, run `python export_web.py` to refresh the weights.

## Numbers quoted on the page (so you can defend them)

* **Linear probe** (logistic regression, 5,000 train / 2,000 test MNIST digits, mid-block features at t = 50, global-average-pooled to 64 numbers): untrained U-Net 55.4%, raw pixels 85.5%, trained U-Net 94.5%. Flattened mid-block (3,136 numbers): trained 97.8%, untrained 88.3%. Rerun with `visualize.py --represent`.
* **Browser ≡ PyTorch**: the TF.js U-Net matches `diffusers` to 2×10⁻⁶ on a forward pass. A full 50-step DDPM sample (with and without CFG), replayed with identical noise, matches `DDPMScheduler` to 1×10⁻⁵.
* **SD 1.5**: ~860M-parameter U-Net, CLIP ViT-L/14 text encoder (~123M), 77-token context × 768 dims, latent 4×64×64 for 512×512 images (48× fewer numbers), default guidance 7.5.

## Facilitator notes

* The browser lab needs WebGL, which any laptop or phone from the last several years has. Without it, TF.js falls back to CPU and sampling becomes slow.
* Station 3 at t ≈ 500 often shows x̂₀ as the *wrong* digit (a 7 that looks like a 9). That's a good teaching moment: at that noise level the MSE-optimal answer is a blend of the plausible digits.
* Station 4's "Same start, new step noise" usually changes some digits. DDPM injects fresh noise each step, so the starting noise alone doesn't fix the output. DDIM (σ = 0) would.
* Station 6 with prompt ∅: guidance does nothing because ε_cond = ε_uncond, so the difference term is zero.
