import streamlit as st
import subprocess
import yaml
import os
from PIL import Image
import glob
import tempfile
import time


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_diffusion_model(checkpoint_path, num_steps, num_images, config_path):
    """Run the diffusion model with specified parameters"""
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "python", "-m", "sample_ddpm_class_cond_streamlit",
            "--config", config_path,
            "--model_path", checkpoint_path,
            "--num_steps", str(num_steps),
            "--num_samples", str(num_images),
            "--output_dir", temp_dir
        ]
        print(cmd)
        print(temp_dir)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Show progress
            with st.spinner('Generating images...'):
                while process.poll() is None:
                    time.sleep(0.1)

            # Get generated images
            image_files = glob.glob(os.path.join(temp_dir, "*.png"))
            print("Images path:", image_files)
            images = [Image.open(f) for f in image_files]
            return images

        except Exception as e:
            st.error(f"Error running model: {str(e)}")
            return None


def main():
    st.title("Diffusion Model Image Generator")
    st.write("Generate images using a pre-trained diffusion model")

    # Sidebar for parameters
    with st.sidebar:
        st.header("Model Parameters")

        checkpoint_path = st.text_input(
            "Model Checkpoint Path",
            value="models/weights/ddpm_ckpt_class_cond_001.pth",
            help="Path to your trained model checkpoint"
        )

        config_path = st.text_input(
            "Config Path",
            value="config/kermany_class_cond_256.yaml",
            help="Path to your model config file"
        )

        num_steps = st.slider(
            "Number of Diffusion Steps",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of steps in the diffusion process"
        )

        num_images = st.slider(
            "Number of Images",
            min_value=1,
            max_value=16,
            value=4,
            step=1,
            help="Number of images to generate"
        )

        generate_button = st.button("Generate Images")

    # Main content area
    if generate_button:
        if not os.path.exists(config_path):
            st.error(f"Config file not found: {config_path}")
            return

        if not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint file not found: {checkpoint_path}")
            return

        images = run_diffusion_model(checkpoint_path, num_steps, num_images, config_path)

        if images:
            # Display images in a grid
            cols = min(4, num_images)  # Max 4 images per row
            rows = (num_images + cols - 1) // cols

            for row in range(rows):
                with st.container():
                    columns = st.columns(cols)
                    for col in range(cols):
                        idx = row * cols + col
                        if idx < len(images):
                            # Display thumbnail
                            columns[col].image(
                                images[idx],
                                caption=f"Generated Image {idx + 1}",
                                use_column_width=True
                            )

                            # Add button to show full-size image
                            if columns[col].button(f"View Full Size #{idx + 1}"):
                                st.image(
                                    images[idx],
                                    caption=f"Full Size Image {idx + 1}",
                                    use_column_width=True
                                )


if __name__ == "__main__":
    main()