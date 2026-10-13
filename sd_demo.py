"""
Demo 7 - Stable Diffusion, the same loop at scale. Needs a GPU (Colab T4 is fine).

    pip install diffusers transformers accelerate
    python sd_demo.py --all              # or pick: --tokens --steps --guidance --negative --seeds --img2img --latents

Every experiment maps onto a station from the MNIST lab:

    --tokens    what CLIP actually sees (77-token limit)        -> station 7
    --latents   decode the latent at intermediate steps         -> station 4 (x_t over time)
    --steps     num_inference_steps sweep                       -> station 4 (steps)
    --seeds     same prompt, different starting noise           -> station 4 (seed)
    --guidance  guidance_scale sweep                            -> station 6 (w)
    --negative  negative_prompt replaces the empty prompt       -> station 6 (the ∅ branch)
    --img2img   strength sweep on a starting image              -> station 5 (repair / SDEdit)

Images are saved to sd_outputs/ as labelled grids.
"""

import argparse
import os

import torch
from PIL import Image, ImageDraw

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"   # SD 1.5 (512 x 512)
PROMPT = "a corgi in an astronaut suit floating above a desert at dusk, loose watercolor illustration"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd_outputs")


def get_pipe():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def grid(images, labels, name, cols=None):
    # Paste images side by side with a caption under each, then save.
    cols = cols or len(images)
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * (h + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(images, labels)):
        x, y = (i % cols) * w, (i // cols) * (h + 28)
        canvas.paste(im, (x, y))
        draw.text((x + 6, y + h + 6), lab, fill="black")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    canvas.save(path)
    print("saved", path)


def gen(pipe, seed=0, **kw):
    g = torch.Generator("cuda").manual_seed(seed)   # seed -> the starting latent noise
    kw.setdefault("prompt", PROMPT)
    return pipe(generator=g, **kw).images[0]


def show_tokens(pipe):
    # CLIP sees at most 77 tokens: <start> + 75 + <end>. Everything after is cut.
    ids = pipe.tokenizer(PROMPT).input_ids
    print(f"{len(ids)} tokens (limit {pipe.tokenizer.model_max_length})")
    print(pipe.tokenizer.convert_ids_to_tokens(ids))
    with torch.no_grad():
        emb = pipe.text_encoder(pipe.tokenizer(PROMPT, padding="max_length", return_tensors="pt").input_ids.to("cuda"))[0]
    print("text embedding the U-Net cross-attends to:", tuple(emb.shape))   # (1, 77, 768)


def show_latents(pipe, steps=30, every=5):
    # Decode the 4 x 64 x 64 latent with the VAE at intermediate steps: x_t over time.
    snaps, labels = [], []

    def cb(p, i, t, kwargs):
        if i % every == 0 or i == steps - 1:
            with torch.no_grad():
                img = p.vae.decode(kwargs["latents"] / p.vae.config.scaling_factor).sample
            img = ((img[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
            snaps.append(Image.fromarray(img).resize((256, 256)))
            labels.append(f"step {i + 1}/{steps}  t={int(t)}")
        return kwargs

    gen(pipe, num_inference_steps=steps, callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    grid(snaps, labels, "latents_over_time")
    print("latent shape: (1, 4, 64, 64) vs image (1, 3, 512, 512) -> 48x fewer numbers")


def show_steps(pipe):
    vals = [5, 10, 20, 50]
    grid([gen(pipe, num_inference_steps=s).resize((256, 256)) for s in vals], [f"steps={s}" for s in vals], "steps")


def show_seeds(pipe):
    vals = [0, 1, 2, 3]
    grid([gen(pipe, seed=s).resize((256, 256)) for s in vals], [f"seed={s}" for s in vals], "seeds")


def show_guidance(pipe):
    vals = [1.0, 3.0, 7.5, 15.0]
    grid([gen(pipe, guidance_scale=w).resize((256, 256)) for w in vals], [f"guidance={w}" for w in vals], "guidance")


def show_negative(pipe):
    negs = ["", "blurry, low detail", "orange, warm colors", "dog"]
    grid([gen(pipe, negative_prompt=n).resize((256, 256)) for n in negs],
         [f"neg='{n}'" if n else "no negative" for n in negs], "negative_prompt")


def show_img2img(pipe):
    from diffusers import StableDiffusionImg2ImgPipeline
    i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
    # A crude "scribble": orange sun over a sand-coloured ground. Replace with any photo.
    init = Image.new("RGB", (512, 512), (70, 120, 170))
    d = ImageDraw.Draw(init)
    d.rectangle([0, 340, 512, 512], fill=(200, 170, 110))
    d.ellipse([190, 120, 320, 250], fill=(240, 140, 40))
    vals = [0.3, 0.5, 0.7, 0.9]
    outs = [init.resize((256, 256))]
    for s in vals:
        g = torch.Generator("cuda").manual_seed(0)
        outs.append(i2i(prompt="a desert sunset, oil painting", image=init, strength=s, generator=g).images[0].resize((256, 256)))
    grid(outs, ["input"] + [f"strength={s}" for s in vals], "img2img_strength")


if __name__ == "__main__":
    demos = {"tokens": show_tokens, "latents": show_latents, "steps": show_steps, "seeds": show_seeds,
             "guidance": show_guidance, "negative": show_negative, "img2img": show_img2img}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true")
    for name in demos:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    chosen = [f for n, f in demos.items() if args.all or getattr(args, n)]
    if not chosen:
        parser.print_help()
    else:
        pipe = get_pipe()
        for f in chosen:
            f(pipe)
