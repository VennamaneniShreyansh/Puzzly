import kornia.augmentation as K


def build_gpu_transform():
    return K.AugmentationSequential(
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=0.5),
        K.RandomGrayscale(p=0.1),
        data_keys=["input"],
    )