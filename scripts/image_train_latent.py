"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_text_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch
import random

from encoders.modules import BERTEmbedder

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("loading vae...")

    encoder = torch.load(args.kl_model, map_location="cpu")
    encoder.to(dist_util.dev())
    encoder.eval()
    set_requires_grad(encoder, False)

    del encoder.decoder
    del encoder.loss

    logger.log("loading text encoder...")

    
    bert = BERTEmbedder(1280, 32)
    sd = torch.load('bert.ckpt', map_location="cpu")
    bert.load_state_dict(sd)

    bert.to(dist_util.dev())
    bert.eval()
    set_requires_grad(bert, False)

    #print('test: ', bert.encode(['test test']))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print('model!', model)

    model.to(dist_util.dev())

    logger.log('total base parameters', sum(x.numel() for x in model.parameters()))

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_latent_data(
        encoder,
        bert,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def load_latent_data(encoder, bert, data_dir, batch_size, image_size):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=256,
        class_cond=False,
    )
    for batch, model_kwargs, text in data:

        text_emb = bert.encode(list(text)).to(dist_util.dev()).half()
        text_blank = bert.encode(['']*batch.shape[0]).to(dist_util.dev()).half()


        for i in range(batch.shape[0]):
            if random.randint(0,100) < 20:
                text_emb[i] = text_blank[i]

        model_kwargs["context"] = text_emb

        batch = batch.to(dist_util.dev())
        emb = encoder.encode(batch).sample().half()
        emb *= 0.18215

        yield emb, model_kwargs

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        kl_model=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
