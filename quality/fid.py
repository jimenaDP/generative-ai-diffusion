import torch
import torch.nn.functional as F

def rescale(x):
    """
    Prepare grayscale MNIST images for FID.

    Input:
        x: Tensor of shape (B, 1, 28, 28), float in [0, 1]

    Output:
        Tensor of shape (B, 3, 299, 299), uint8 in [0, 255]
    """

    #If images are grayscale, repeat the channel 3 times
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    #Resize to the input size expected by Inception
    x = F.interpolate(
        x,
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
    )

    # Clamp values to [0, 1]
    x = torch.clamp(x, 0.0, 1.0)

    #Convert to uint8 [0, 255]
    x = (x * 255).to(torch.uint8)

    return x