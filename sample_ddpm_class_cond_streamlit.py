import torch
import torchvision
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_images(config_path, model_path, num_samples=4, num_steps=None, output_dir=None):
    """
    Modified sampling function that accepts parameters from the Streamlit GUI

    Args:
        config_path: Path to the config YAML file
        model_path: Path to the model checkpoint
        num_samples: Number of images to generate
        num_steps: Number of diffusion steps (overrides config if provided)
        output_dir: Directory to save generated images

    Returns:
        List of PIL Images
    """
    # Read the config file
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise Exception(f"Error reading config: {exc}")

    # Update config with GUI parameters
    config['train_params']['num_samples'] = num_samples
    if num_steps is not None:
        config['diffusion_params']['num_timesteps'] = num_steps

    if model_path:
        config['train_params']['ldm_ckpt_name'] = os.path.basename(model_path)
        config['train_params']['task_name'] = os.path.dirname(model_path)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )

    # Load Unet
    model = Unet(
        im_channels=autoencoder_model_config['z_channels'],
        model_config=diffusion_model_config
    ).to(device)
    model.eval()

    if os.path.exists(model_path):
        print('Loading unet checkpoint')
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise Exception(f'Model checkpoint {model_path} not found')

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load VQVAE
    vae = VQVAE(
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_model_config
    ).to(device)
    vae.eval()

    vae_path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
    if os.path.exists(vae_path):
        print('Loading vae checkpoint')
        vae.load_state_dict(torch.load(vae_path, map_location=device), strict=True)
    else:
        raise Exception(f'VAE checkpoint {vae_path} not found')

    # Generate images
    generated_images = []

    with torch.no_grad():
        # Get image size
        im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])

        # Sample random noise
        xt = torch.randn((num_samples,
                          autoencoder_model_config['z_channels'],
                          im_size,
                          im_size)).to(device)

        # Set up conditioning
        condition_config = get_config_value(diffusion_model_config, 'condition_config', None)
        if condition_config is not None:
            num_classes = condition_config['class_condition_config']['num_classes']
            sample_classes = torch.randint(0, num_classes, (num_samples,))
            print(f'Generating images for classes: {list(sample_classes.numpy())}')

            cond_input = {
                'class': torch.nn.functional.one_hot(sample_classes, num_classes).to(device)
            }
            uncond_input = {
                'class': cond_input['class'] * 0
            }
        else:
            cond_input = {}
            uncond_input = {}

        cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)

        # Sampling loop
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            t = (torch.ones((xt.shape[0],)) * i).long().to(device)
            noise_pred_cond = model(xt, t, cond_input)

            if cf_guidance_scale > 1:
                noise_pred_uncond = model(xt, t, uncond_input)
                noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            if i == 0:
                ims = vae.decode(xt)
            else:
                ims = x0_pred

            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2

            # Save final images
            if i == 0:
                for idx in range(num_samples):
                    img = torchvision.transforms.ToPILImage()(ims[idx])
                    generated_images.append(img)
                    if output_dir:
                        img.save(os.path.join(output_dir, f'sample_{idx}.png'))

    return generated_images


def sample_with_config(config_path, **kwargs):
    """Wrapper function to handle config loading and sampling"""
    try:
        return sample_images(config_path, **kwargs)
    except Exception as e:
        raise Exception(f"Error during sampling: {str(e)}")