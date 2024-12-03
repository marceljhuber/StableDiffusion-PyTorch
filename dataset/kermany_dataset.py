import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class KermanyDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for kermany images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """

    def __init__(
        self,
        split,
        im_path,
        im_size,
        im_channels,
        use_latents=False,
        latent_path=None,
        condition_config=None,
    ):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels

        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False

        # Conditioning for the dataset
        self.condition_types = (
            [] if condition_config is None else condition_config["condition_types"]
        )

        self.images, self.labels = self.load_images(im_path)

        # Whether to load images and call vae or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print("Found {} latents".format(len(self.latent_maps)))
            else:
                print("Latents not found")

        # Define the transformation pipeline for resizing and tensor conversion
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.im_size, self.im_size)
                ),  # Resize to the target size
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(
                    mean=[0.5] * im_channels, std=[0.5] * im_channels
                ),  # Normalize to [-1, 1]
            ]
        )

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            fnames = glob.glob(os.path.join(im_path, d_name, "*.{}".format("png")))
            fnames += glob.glob(os.path.join(im_path, d_name, "*.{}".format("jpg")))
            fnames += glob.glob(os.path.join(im_path, d_name, "*.{}".format("jpeg")))
            for fname in fnames:
                ims.append(fname)
                if "class" in self.condition_types:
                    labels.append(int(d_name))
        print("Found {} images for split {}".format(len(ims), self.split))
        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if "class" in self.condition_types:
            cond_inputs["class"] = self.labels[index]
        #######################################

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            im = Image.open(self.images[index]).convert("L")
            im_tensor = self.transform(im)

            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
