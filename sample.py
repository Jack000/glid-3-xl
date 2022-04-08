import gc
import io
import math
import sys

from PIL import Image
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import numpy as np

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from dalle_pytorch import DiscreteVAE, VQGanVAE

from einops import rearrange
from math import log2, sqrt

import argparse
import pickle

import os

from encoders.modules import BERTEmbedder

import clip

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default = 'diffusion.pt',
                   help='path to the diffusion model')

parser.add_argument('--kl_path', type=str, default = 'kl-f8.pt',
                   help='path to the LDM first stage model')

parser.add_argument('--bert_path', type=str, default = 'bert.pt',
                   help='path to the LDM first stage model')

parser.add_argument('--text', type = str, required = False,
                    help='your text prompt')

parser.add_argument('--negative', type = str, required = False, default = '',
                    help='negative text prompt')

parser.add_argument('--prefix', type = str, required = False, default = '',
                    help='prefix for output files')

parser.add_argument('--num_batches', type = int, default = 1, required = False,
                    help='number of batches')

parser.add_argument('--batch_size', type = int, default = 1, required = False,
                    help='batch size')

parser.add_argument('--width', type = int, default = 256, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--height', type = int, default = 256, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--seed', type = int, default=-1, required = False,
                    help='random seed')

parser.add_argument('--guidance_scale', type = float, default = 5.0, required = False,
                    help='classifier-free guidance scale')

parser.add_argument('--steps', type = int, default = 0, required = False,
                    help='number of diffusion steps')

parser.add_argument('--cpu', dest='cpu', action='store_true')

parser.add_argument('--clip_score', dest='clip_score', action='store_true')

parser.add_argument('--clip_guidance', dest='clip_guidance', action='store_true')

parser.add_argument('--clip_guidance_scale', type = float, default = 150, required = False,
                    help='Controls how much the image should look like the prompt') # may need to use lower value for ddim

parser.add_argument('--cutn', type = int, default = 16, required = False,
                    help='Number of cuts')

parser.add_argument('--ddim', dest='ddim', action='store_true') # turn on to use 50 step ddim

args = parser.parse_args()


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
print('Using device:', device)

model_params = {
    'attention_resolutions': '32,16,8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '27',  # Modify this value to decrease the number of
                                   # timesteps.
    'image_size': 32,
    'learn_sigma': False,
    'noise_schedule': 'linear',
    'num_channels': 320,
    'num_heads': 8,
    'num_res_blocks': 2,
    'resblock_updown': False,
    'use_fp16': False,
    'use_scale_shift_norm': False
}

if args.ddim:
    model_params['timestep_respacing'] = 'ddim50'

if args.steps:
    model_params['timestep_respacing'] = str(args.steps)

model_config = model_and_diffusion_defaults()
model_config.update(model_params)

if args.cpu:
    model_config['use_fp16'] = False

# Load models
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)
model.requires_grad_(args.clip_guidance).eval().to(device)

if model_config['use_fp16']:
    model.convert_to_fp16()
else:
    model.convert_to_fp32()

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

# vae
ldm = torch.load(args.kl_path, map_location="cpu")
ldm.to(device)
ldm.eval()
ldm.requires_grad_(args.clip_guidance)
set_requires_grad(ldm, args.clip_guidance)

bert = BERTEmbedder(1280, 32)
sd = torch.load(args.bert_path, map_location="cpu")
bert.load_state_dict(sd)

bert.to(device)
bert.half().eval()
set_requires_grad(bert, False)

if args.clip_guidance or args.clip_score:
    clip_model, clip_preprocess = clip.load('ViT-L/14', device=device, jit=False)
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

def do_run():
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    text_emb = bert.encode([args.text]*args.batch_size).to(device).float()
    text_blank = bert.encode([args.negative]*args.batch_size).to(device).float()

    kwargs = { "context": torch.cat([text_emb, text_blank], dim=0) }

    if args.clip_guidance or args.clip_score:
        text = clip.tokenize([args.text]*args.batch_size, truncate=True).to(device)
        text_emb_clip = clip_model.encode_text(text)
        make_cutouts = MakeCutouts(clip_model.visual.input_resolution, args.cutn)

        text_emb_norm = text_emb_clip[0] / text_emb_clip[0].norm(dim=-1, keepdim=True)

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + args.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    cur_t = None

    def cond_fn(x, t, context=None):
        with torch.enable_grad():
            x = x[:args.batch_size].detach().requires_grad_()

            n = x.shape[0]

            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t

            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'context': context[:args.batch_size]})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)


            x_in /= 0.18215

            x_img = ldm.decode(x_in)

            clip_in = normalize(make_cutouts(x_img.add(1).div(2)))
            clip_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(clip_embeds.unsqueeze(1), text_emb_clip.unsqueeze(0))
            dists = dists.view([args.cutn, n, -1])

            losses = dists.sum(2).mean(0)

            loss = losses.sum() * args.clip_guidance_scale

            return -torch.autograd.grad(loss, x)[0]
 
    if args.ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    def save_sample(i, sample, clip_score=False):
        for k, image in enumerate(sample['pred_xstart'][:args.batch_size]):
            image /= 0.18215
            im = image.unsqueeze(0)
            out = ldm.decode(im)

            out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))

            filename = f'output/{args.prefix}_progress_{i * args.batch_size + k:05}.png'
            out.save(filename)

            if clip_score:
                image_emb = clip_model.encode_image(clip_preprocess(out).unsqueeze(0).to(device))
                image_emb_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)

                similarity = torch.nn.functional.cosine_similarity(image_emb_norm, text_emb_norm, dim=-1)

                final_filename = f'output/{args.prefix}_{similarity.item():0.3f}_{i * args.batch_size + k:05}.png'
                os.rename(filename, final_filename)

    for i in range(args.num_batches):
        cur_t = diffusion.num_timesteps - 1

        samples = sample_fn(
            model_fn,
            (args.batch_size*2, 4, int(args.height/8), int(args.width/8)),
            clip_denoised=False,
            model_kwargs=kwargs,
            cond_fn=cond_fn if args.clip_guidance else None,
            device=device,
            progress=True,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 5 == 0 and j != diffusion.num_timesteps - 1:
                save_sample(i, sample)

        save_sample(i, sample, args.clip_score)

gc.collect()
do_run()
