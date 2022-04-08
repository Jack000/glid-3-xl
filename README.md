# GLID-3-XL

GLID-3-xl is the [1.4B latent diffusion](https://github.com/CompVis/latent-diffusion#april-2022) model back-ported to the guided diffusion codebase

the text encoder, first stage vae and the diffusion model itself have been split into three checkpoints. This lets us train/fine tune the diffusion model on different/cleaner datasets.

# Install

First install [latent diffusion](https://github.com/CompVis/latent-diffusion)
```
# then
git clone https://github.com/Jack000/glid-3-xl
cd glid-3-xl
pip install -e .
```

# Sampling from pre-trained models

```
# text encoder
wget https://dall-3.com/models/glid-3-xl/bert.pt

# ldm first stage
wget https://dall-3.com/models/glid-3-xl/kl-f8.pt

# diffusion model
wget https://dall-3.com/models/glid-3-xl/diffusion.pt


# fast PLMS sampling
python sample.py --model_path diffusion.pt --kl_path kl-f8.pt --width 256 --height 256 --batch_size 6 --num_batches 6 --text "a cyberpunk girl with a scifi neuralink device on her head"

# classifier free guidance + CLIP guidance (better adherence to prompt, much slower)
python sample.py --clip_guidance --model_path diffusion.pt --kl_path kl-f8.pt --width 256 --height 256 --batch_size 1 --num_batches 12 --text "a cyberpunk girl with a scifi neuralink device on her head | trending on artstation"

# generated images saved to ./output/
```

# Training/Fine tuning
Train with same flags as guided diffusion. Data directory should contain image and text files with the same name (image1.png image1.txt)

```
# not possible to train on 24gb vram currently!
MODEL_FLAGS="--ema_rate 0.9999 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --num_channels 320 --num_heads 8 --num_res_blocks 2 --resblock_updown False --use_fp16 True --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 1e-5 --batch_size 1 --log_interval 1 --save_interval 5000 --kl_model kl-f8.pt --resume_checkpoint diffusion.pt"
export OPENAI_LOGDIR=./logs/
export TOKENIZERS_PARALLELISM=false
python scripts/image_train_latent.py --data_dir /path/to/data $MODEL_FLAGS $TRAIN_FLAGS
```
