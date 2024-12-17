import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import save_image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, output_dir):
    r"""
    Sample stepwise by going backward one timestep at a time.
    Only save the final images in full resolution.
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])

    # Sample random noise latent
    xt = torch.randn((train_config['num_samples'],
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)

    # Validate the config
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for class conditional "
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'class' in condition_types, ("This sampling script is for class conditional "
                                        "but no class condition found in config")
    validate_class_config(condition_config)

    # Create Conditional input
    num_classes = condition_config['class_condition_config']['num_classes']
    condition = diffusion_config['class']

    if condition in [0, 1, 2, 3]:
        sample_classes = torch.tensor([condition]).repeat(train_config['num_samples'])
    else:
        sample_classes = torch.randint(0, num_classes, (train_config['num_samples'], ))
    print('Generating images for {}'.format(list(sample_classes.numpy())))
    cond_input = {
        'class': torch.nn.functional.one_hot(sample_classes, num_classes).to(device)
    }
    # Unconditional input for classifier free guidance
    uncond_input = {
        'class': cond_input['class'] * 0
    }

    # Classifier free guidance scale
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)

    # Sampling Loop
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        t = (torch.ones((xt.shape[0],)) * i).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)

        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    # Decode final images
    final_images = vae.decode(xt)
    final_images = torch.clamp(final_images, -1., 1.).detach().cpu()
    final_images = (final_images + 1) / 2  # Scale to [0, 1]

    # Save images with formatted names
    existing_images = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    start_index = max([int(f.split('_')[0]) for f in existing_images] + [0])

    for idx, img_tensor in enumerate(final_images):
        image_class = sample_classes[idx].item()
        file_name = f"{start_index + idx + 1:04d}_{image_class}.png"
        save_image(img_tensor, os.path.join(output_dir, file_name))
        print(f"Saved: {file_name}")


def infer(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    # Load Unet
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded Unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                         map_location=device), strict=False)
    else:
        raise Exception('Model checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                            train_config['ldm_ckpt_name'])))

    # Load VQVAE
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded VQVAE checkpoint')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device), strict=True)
    else:
        raise Exception('VQVAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                            train_config['vqvae_autoencoder_ckpt_name'])))

    # Create output directory
    output_dir = 'images_output'
    os.makedirs(output_dir, exist_ok=True)

    # Generate samples
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM image generation for class conditional generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/kermany_class_cond_256_V2.yaml', type=str)
    args = parser.parse_args()
    infer(args)
