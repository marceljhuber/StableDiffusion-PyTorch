import os
import glob
import argparse
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the model's input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

reverse_transform = transforms.Compose([
    transforms.Normalize((-1,), (2,)),  # Unnormalize
    transforms.ToPILImage()
])


@torch.no_grad()  # Avoid tracking gradients for reconstruction
def reconstruct_images(images, model, scheduler, vae, num_timesteps, counterfactuals, classes):
    """Reconstruct a batch of images."""
    xt = vae.encode(images.to(device))[0]  # Encode images with VQVAE
    print(images.shape)

    # Handle counterfactuals or specific class conditions
    if counterfactuals:
        all_classes = torch.arange(4, device=device)  # Classes [0, 1, 2, 3]
        reconstructions = []
        for c, i in enumerate(all_classes):
            cond_input = {'class': torch.nn.functional.one_hot(torch.tensor([c], device=device), num_classes=4).repeat(xt.shape[0], 1)}
            xt_copy = xt.clone()
            for t in tqdm(reversed(range(num_timesteps)), desc=f"Class {c}"):
                t_tensor = torch.full((xt_copy.shape[0],), t, device=device, dtype=torch.long)
                noise_pred = model(xt_copy, t_tensor, cond_input)
                print(f"xt_copy.shape:", xt_copy.shape)
                print(f"noise_pred.shape:", noise_pred.shape)
                print(f"t_tensor.shape:", t_tensor.shape)
                xt_copy, _ = scheduler.sample_prev_timestep(xt_copy, noise_pred, t_tensor)
            reconstructions.append(torch.clamp(vae.decode(xt_copy), -1., 1.))
        return torch.cat(reconstructions)  # Combine reconstructions for all classes
    else:
        # Ensure class tensor is properly shaped for batch processing
        if isinstance(classes, int):
            classes = torch.tensor([classes] * images.size(0), device=device)
        else:
            classes = torch.tensor(classes, device=device)
        cond_input = {'class': torch.nn.functional.one_hot(classes, num_classes=4)}
        for t in tqdm(reversed(range(num_timesteps))):
            t_tensor = torch.full((xt.shape[0],), t, device=device, dtype=torch.long)
            noise_pred = model(xt, t_tensor, cond_input)
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t_tensor)
        return torch.clamp(vae.decode(xt), -1., 1.)


def generate_comparison_plot(original_images, reconstructed_images, output_path):
    """Generate a 2xN plot comparing original and reconstructed images."""
    n = len(original_images)
    fig, axes = plt.subplots(2, n, figsize=(n * 4, 8))

    for i in range(n):
        axes[0, i].imshow(np.asarray(original_images[i]))
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        recon_image = np.moveaxis(reconstructed_images[i].cpu().numpy(), 0, -1)  # CxHxW -> HxWxC
        axes[1, i].imshow((recon_image * 0.5 + 0.5).clip(0, 1))  # Unnormalize to [0, 1]
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstruction")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(args):
    # Load configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_config = config['train_params']
    autoencoder_config = config['autoencoder_params']
    diffusion_params = config['diffusion_params']

    counterfactuals = args.counterfactuals
    batch_size = args.batch_size
    num_timesteps = diffusion_params['num_timesteps']

    # Initialize the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=num_timesteps,
        beta_start=diffusion_params['beta_start'],
        beta_end=diffusion_params['beta_end']
    )

    # Load U-Net model
    model = Unet(im_channels=autoencoder_config['z_channels'],
                 model_config=config['ldm_params']).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']),
                                     map_location=device))
    model.eval()

    # Load VQVAE
    vae = VQVAE(im_channels=config['dataset_params']['im_channels'],
                model_config=autoencoder_config).to(device)
    vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                train_config['vqvae_autoencoder_ckpt_name']),
                                   map_location=device))
    vae.eval()

    # Create output directory
    output_dir = "reconstruction/output"
    os.makedirs(output_dir, exist_ok=True)

    # Load and process images
    input_images = sorted(glob.glob(os.path.join("reconstruction/input", "*.*")))
    if not input_images:
        print("No input images found in reconstruction/input.")
        return

    if isinstance(args.class_input, list) and len(args.class_input) != len(input_images):
        raise ValueError("The length of class_input must match the number of input images.")

    original_images = [Image.open(img_path) for img_path in input_images]
    image_tensors = torch.stack([transform(img) for img in original_images])

    # Process images in batches
    reconstructed_images = []
    for i in range(0, len(image_tensors), batch_size):
        batch_images = image_tensors[i:i + batch_size]
        batch_classes = args.class_input[i:i + batch_size] if isinstance(args.class_input, list) else args.class_input
        reconstructed_batch = reconstruct_images(batch_images, model, scheduler, vae, num_timesteps,
                                                  counterfactuals, batch_classes)
        reconstructed_images.extend(reconstructed_batch)

    # Save reconstructed images and generate comparison plot
    for original_img, recon_tensor, img_path in zip(original_images, reconstructed_images, input_images):
        recon_image = reverse_transform(recon_tensor)
        output_image_path = os.path.join(output_dir, os.path.basename(img_path))
        recon_image.save(output_image_path)

    comparison_plot_path = "reconstruction/comparison_plot.png"
    generate_comparison_plot(original_images, reconstructed_images, comparison_plot_path)
    print(f"Comparison plot saved at {comparison_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Reconstruction with Conditional Diffusion")
    parser.add_argument("--config", dest="config_path", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing images")
    parser.add_argument("--class_input", type=lambda x: eval(x), required=True,
                        help="Class input: single integer or list of integers. Example: 1 or [0, 1, 2].")
    parser.add_argument("--counterfactuals", action="store_true",
                        help="If set, reconstruct images for all classes [0, 1, 2, 3]")
    args = parser.parse_args()
    main(args)

