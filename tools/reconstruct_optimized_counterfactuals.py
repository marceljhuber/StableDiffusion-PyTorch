import os
import glob
import argparse
import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import get_config_value, validate_class_config
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
def reconstruct_image(image_tensor, model, scheduler, vae, train_config, diffusion_config, counterfactuals=False):
    """Reconstruct an image using the provided model and scheduler."""
    xt = vae.encode(image_tensor.unsqueeze(0).to(device))[0]

    if not counterfactuals:
        # Conditional input for the original class (the class from the config)
        original_class = diffusion_config['class']
        sample_class = torch.tensor([original_class], device=device)
        cond_input = {'class': torch.nn.functional.one_hot(sample_class, num_classes=4)}

        # Diffusion sampling loop for the original class
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            t = torch.full((xt.shape[0],), i, device=device, dtype=torch.long)
            noise_pred = model(xt, t, cond_input)
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t)

        # Decode the final image for the original class
        final_image = vae.decode(xt)
        original_reconstruction = torch.clamp(final_image, -1., 1.).squeeze(0).detach().cpu()

        if not counterfactuals:
            return original_reconstruction

    # If counterfactuals is True, reconstruct for all classes [0, 1, 2, 3]
    all_reconstructions = []
    for class_id in range(4):
        sample_class = torch.tensor([class_id], device=device)
        cond_input = {'class': torch.nn.functional.one_hot(sample_class, num_classes=4)}

        # Diffusion sampling loop for the current class
        xt = vae.encode(image_tensor.unsqueeze(0).to(device))[0]  # Reset the latent representation
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            t = torch.full((xt.shape[0],), i, device=device, dtype=torch.long)
            noise_pred = model(xt, t, cond_input)
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t)

        # Decode the final image for the current class
        final_image = vae.decode(xt)
        class_reconstruction = torch.clamp(final_image, -1., 1.).squeeze(0).detach().cpu()
        all_reconstructions.append(class_reconstruction)

    return all_reconstructions


def generate_comparison_plot(original_images, reconstructed_images, output_path):
    """Generate a plot comparing original and reconstructed images (including counterfactuals)."""
    n = len(original_images)
    fig, axes = plt.subplots(5, n, figsize=(n * 4, 12))  # 5 rows: Original + 4 classes

    for i in range(n):
        # Display original image
        axes[0, i].imshow(original_images[i], cmap='gray')
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        # Display reconstructions for each class
        for j in range(4):
            # Squeeze the image tensor to remove any extra dimensions
            img = reconstructed_images[i][j].squeeze().numpy() * 255  # Remove extra batch dimension and convert to numpy
            axes[j + 1, i].imshow(img, cmap='gray')
            axes[j + 1, i].axis("off")
            axes[j + 1, i].set_title(f"Class {j}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def main(args):
    # Load configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_config = config['train_params']
    diffusion_config = config['diffusion_params']
    autoencoder_config = config['autoencoder_params']

    # Initialize the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
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

    # Load input images
    input_images = sorted(glob.glob(os.path.join("reconstruction/input", "*.*")))
    original_images = []
    reconstructed_images = []

    if not input_images:
        print("No input images found in reconstruction/input.")
        return

    # Set the counterfactuals flag based on argument or configuration
    counterfactuals = args.counterfactuals

    for image_path in input_images:
        print(f"Processing: {image_path}")
        original_image = Image.open(image_path)
        image_tensor = transform(original_image)

        # Reconstruct the image (original + counterfactuals if enabled)
        reconstructed_tensors = reconstruct_image(image_tensor, model, scheduler, vae, train_config, diffusion_config,
                                                  counterfactuals)

        # If counterfactuals is True, reconstructed_tensors will be a list of 5 images (original + 4 classes)
        reconstructed_images_for_image = reconstructed_tensors if counterfactuals else [reconstructed_tensors]

        # Save the reconstructed image(s)
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        if counterfactuals:
            for class_id, reconstructed_img in enumerate(reconstructed_images_for_image):
                save_image((reconstructed_img + 1) / 2, f"{output_image_path.split('.')[0]}_class_{class_id}.png")
        else:
            reconstructed_image = reconstructed_images_for_image[0]
            reconstructed_image.save(output_image_path)

        # Add to comparison list
        original_images.append(original_image)
        reconstructed_images.append(reconstructed_images_for_image)

    # Generate comparison plot
    comparison_plot_path = "reconstruction/comparison_plot.png"
    generate_comparison_plot(original_images, reconstructed_images, comparison_plot_path)
    print(f"Comparison plot saved at {comparison_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Reconstruction with Conditional Diffusion")
    parser.add_argument("--config", dest="config_path", required=True,
                        help="Path to the YAML configuration file")
    parser.add_argument("--counterfactuals", dest="counterfactuals", action='store_true',
                        help="Generate counterfactual reconstructions for all classes (0, 1, 2, 3)")
    args = parser.parse_args()
    main(args)
