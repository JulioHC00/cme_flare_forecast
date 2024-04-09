from typing import Union, Tuple

import torch
from torch.nn import Module
import torchvision.transforms.functional as TF
import torch.nn.functional as NF
from torchvision.transforms import InterpolationMode


class RandomRotation(Module):
    """
    A class to perform random rotation on a given image tensor.

    Can handle both single images and sequences of images. So either shape
    (C, H, W) or shape (S, C, H, W). This means it can also handle batches
    but the same rotation will be applied to all images of the sequence.

    Attributes:
    rotation_range (int): The range of rotation in degrees. A random angle within
                          this range will be used for rotation.

    Methods:
    __call__(image: torch.Tensor) -> torch.Tensor: Transforms the input image by rotating it.
    """

    def __init__(self, device, rotation_range: int = 30):
        """
        Initialize the RandomRotation class.

        Parameters:
        rotation_range (int): The range of rotation in degrees. A random angle within
                              this range will be used for rotation. Default is 30 degrees.
        """
        super(RandomRotation, self).__init__()
        self.rotation_range: int = rotation_range
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input image by rotating it.

        Parameters:
        image (torch.Tensor): The input image tensor.

        Returns:
        torch.Tensor: The rotated image tensor.
        """

        angle = torch.randint(
            -self.rotation_range, self.rotation_range + 1, (1,)
        ).item()

        rotated_image = TF.rotate(image, angle, interpolation=InterpolationMode.NEAREST)

        return rotated_image


class RandomZoom(Module):
    """
    A class to perform random zooming on a given image tensor.

    Can handle both single images and sequences of images. So either shape
    (C, H, W) or shape (S, C, H, W). This means it can also handle batches
    but the same zoom will be applied to all images of the sequence.

    Attributes:
    scale_range (Tuple[float, float]): The range of scales to choose from when zooming.
    target_size (Union[Tuple[int, int], str]): The target size to resize the image after zooming.
    min_size (Tuple[int, int]): The minimum size the image can be scaled to.

    Methods:
    __call__(image: torch.Tensor) -> torch.Tensor:
        Transforms the input image by zooming it.
    """

    def __init__(
        self,
        device,
        scale_range: Tuple[float, float] = (0.8, 1.0),
        target_size: Union[Tuple[int, int], str] = "default",
        min_size: Tuple[int, int] = (10, 10),
    ):
        """
        Initialize the RandomZoom class.

        Parameters:
        scale_range (Tuple[float, float]): The range of scales to choose from when zooming. Default is (0.8, 1.0).
        target_size (Union[Tuple[int, int], str]): The target size to resize the image after zooming.
                                                   If "default", the image is not resized. Default is "default".
        min_size (Tuple[int, int]): The minimum size the image can be scaled to. Default is (50, 50).
        """
        super(RandomZoom, self).__init__()
        self.scale_range = torch.tensor(scale_range)
        self.scale_range_diff = self.scale_range[1] - self.scale_range[0]
        self.scale_0 = self.scale_range[0]
        self.target_size = target_size
        self.min_size = min_size
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input image by zooming it.

        Parameters:
        image (torch.Tensor): The input image.

        Returns:
        torch.Tensor: The zoomed image.
        """
        image = image.to(self.device)

        rand_scale = torch.rand(1)

        # Calculate random cropping dimensions
        scale = (rand_scale * self.scale_range_diff + self.scale_0).item()

        # Single image
        if len(image.shape) == 3:
            _, height, width = image.size()
        # Sequence
        elif len(image.shape) == 4:
            _, _, height, width = image.size()
        else:
            raise ValueError(
                "Expected image tensor to have 3 or 4 dimensions, but got {} instead.".format(
                    len(image.shape)
                )
            )

        new_h = int(height * scale)
        new_w = int(width * scale)

        if new_h < self.min_size[0] or new_w < self.min_size[1]:
            return image

        # Randomly select the top-left corner for cropping
        top = torch.randint(0, height - new_h + 1, (1,)).item()
        left = torch.randint(0, width - new_w + 1, (1,)).item()

        # Perform cropping and resizing (zooming)
        if self.target_size == "default":
            image = TF.resized_crop(image, top, left, new_h, new_w, (new_h, new_w))
        elif self.target_size == "same":
            image = TF.resized_crop(image, top, left, new_h, new_w, (height, width))
        else:
            image = TF.resized_crop(image, top, left, new_h, new_w, self.target_size)

        return image


class ResizeAndPad(Module):
    """
    A class to resize an image tensor to a specified size while maintaining the aspect ratio.
    Padding is added to fill up the remaining area.

    Attributes:
    target_size (tuple): The desired output size in the format (height, width).
    device (str): The device on which tensor operations should be performed.

    Methods:
    __call__(image: torch.Tensor) -> torch.Tensor: Transforms the input image by resizing and padding it.
    """

    def __init__(self, target_size: tuple, device: str):
        """
        Initialize the ResizeAndPad class.

        Parameters:
        target_size (tuple): The desired output size in the format (height, width).
        device (str): The device on which tensor operations should be performed.
        """
        super(ResizeAndPad, self).__init__()

        self.target_height, self.target_width = target_size
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input image by resizing and padding it.
        Can handle both single images and sequences of images. So either shape
        (C, H, W) or shape (S, C, H, W). This means it can also handle batches.

        Parameters:
        image (torch.Tensor): The input image tensor.

        Returns:
        torch.Tensor: The resized and padded image tensor.
        """

        # Move the image to the specified device
        image = image.to(self.device)

        # Calculate aspect ratio

        # Single image
        if len(image.shape) == 3:
            _, height, width = image.size()
            input_image = image.unsqueeze(0)
        elif len(image.shape) == 4:
            _, _, height, width = image.size()
            input_image = image

        aspect_ratio = height / width

        # Calculate new dimensions
        if height > width:
            new_height = self.target_height
            new_width = int(new_height / aspect_ratio)
        else:
            new_width = self.target_width
            new_height = int(new_width * aspect_ratio)

        # Resize the image
        image_resized = NF.interpolate(
            input_image,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Calculate padding
        pad_height = self.target_height - new_height
        pad_width = self.target_width - new_width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding
        image_padded = NF.pad(
            image_resized, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0
        )

        return image_padded


class RandomHorizontalFlip(Module):
    """
    A class to perform random horizontal flip on a given image tensor.

    Attributes:
    p (float): Probability of the image being flipped. Default value is 0.5
    device (str): Device to which tensors should be moved for computation.

    Methods:
    __call__(image: torch.Tensor) -> torch.Tensor: Transforms the input image by possibly flipping it horizontally.
    """

    def __init__(self, device: str, p: float = 0.5):
        """
        Initialize the RandomHorizontalFlip class.
        Can handle both single images and sequences of images. So either shape
        (C, H, W) or shape (S, C, H, W). This means it can also handle batches.
        However, the same flip will be applied to all images of the sequence.

        Parameters:
        p (float): Probability of the image being flipped. Default value is 0.5
        device (str): Device to which tensors should be moved for computation.
        """
        super(RandomHorizontalFlip, self).__init__()
        self.p = p
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input image by possibly flipping it horizontally.

        Parameters:
        image (torch.Tensor): The input image tensor.

        Returns:
        torch.Tensor: The transformed image tensor.
        """

        if len(image.shape) == 3:
            flip_ax = 2
        elif len(image.shape) == 4:
            flip_ax = 3
        else:
            raise ValueError(
                "Expected image tensor to have 3 or 4 dimensions, but got {} instead.".format(
                    len(image.shape)
                )
            )

        # Generate a random number to decide whether to flip or not
        if torch.rand(1).item() < self.p:
            return torch.flip(image, [flip_ax]).to(self.device)
        else:
            return image.to(self.device)


class NormalizeTensorImage(Module):
    """
    A class to normalize an image tensor.
    Can handle both single images and sequences of images. So either shape
    (C, H, W) or shape (S, C, H, W). This means it can also handle batches.

    Attributes:
    mean (List[float]): The mean value for each channel.
    std (List[float]): The standard deviation value for each channel.

    Methods:
    __call__(image: torch.Tensor) -> torch.Tensor:
        Transforms the input image by normalizing it.
    """

    def __init__(self, device, mean: list, std: list):
        """
        Initialize the NormalizeImage class.

        Parameters:
        mean (List[float]): The mean value for each channel.
        std (List[float]): The standard deviation value for each channel.
        """
        super(NormalizeTensorImage, self).__init__()
        if len(mean) != len(std):
            raise ValueError("Length of mean and std should be the same")

        self.mean = torch.tensor(mean).to(device)
        self.std = torch.tensor(std).to(device)
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input image by normalizing it.

        Parameters:
        image (torch.Tensor): The input image.

        Returns:
        torch.Tensor: The normalized image.
        """

        image = image.to(self.device)

        if len(image.shape) not in [3, 4]:
            raise ValueError(
                "Expected image tensor to have 3 or 4 dimensions, but got {} instead.".format(
                    len(image.shape)
                )
            )

        if len(self.mean) != image.shape[-3]:
            raise ValueError(
                "Number of channels in image and length of mean and std should match"
            )

        # Normalize the image
        image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        return image
