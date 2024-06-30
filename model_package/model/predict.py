import gc
import datetime
import numpy as np
from tqdm import tqdm
import torch

from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer

from sd_model.clip import CLIP
from sd_model.sampler import DDPMSampler
from sd_model.unet_diffusion import Diffusion
from sd_model.vae import VAE_Encoder, VAE_Decoder

from util.load_config import config as cfg
import util.model_converter as model_converter


def main():
    """ Main function to run the model."""

    if torch.cuda.is_available() and (cfg.device == "cuda"):
        device = "cuda"
    else:
        device = "cpu"

    input_image = ""
    prompt = cfg.prompt
    uncond_prompt = cfg.uncond_prompt
    
    if cfg.mode == "image_to_image":
        image_path = cfg.image_path
        input_image = Image.open(image_path)

    elif cfg.mode == "impaint":
        raise Exception("Impainting mode is not supported yet.")

    output_image = generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=cfg.strength,
        do_cfg=cfg.do_cfg,
        cfg_scale=cfg.cfg_scale,
        n_inference_steps=cfg.num_inference_steps,
        seed=cfg.seed,
        device=device,
        idle_device=cfg.idle_device)

    output_image = Image.fromarray(output_image)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_image.save(f"{cfg.output_dir}/output_{timestamp}.jpg")


def generate(prompt,
            uncond_prompt=None,
            input_image=None,
            strength=0.8,
            do_cfg=True,
            cfg_scale=7.5,
            n_inference_steps=50,
            seed=None,
            device=None,
            idle_device=None):
    
    model_file = cfg.ckpt_path
    models = preload_models_from_standard_weights(model_file, device)

    with torch.no_grad():

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        tokenizer = models["tokenizer"]
        clip = models["clip"].to(device)
        
        # Tokenize the prompt
        cond_tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=cfg.max_length).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        context = clip(cond_tokens)

        if do_cfg:
            # Tokenize the uncond_prompt
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=cfg.max_length).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
  
            context = torch.cat([context, uncond_context])

        to_idle(clip)

        sampler = DDPMSampler(generator=generator,
                              num_training_steps=cfg.num_training_steps,
                              beta_start=cfg.beta_start,
                              beta_end=cfg.beta_end)
        
        sampler.set_inference_timesteps(n_inference_steps)

        latent_dim = cfg.width // cfg.downsampling_ratio
        latents_shape = (1, 4, latent_dim, latent_dim)

        if input_image:
            encoder = models["encoder"].to(device)

            input_image_tensor = input_image.resize((cfg.width, cfg.height))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"].to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"].to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    tokenizer = CLIPTokenizer(cfg.vocab_file_path, merges_file=cfg.merges_file_path)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'tokenizer': tokenizer,
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

if __name__ == '__main__':
    gc.collect()
    main()
