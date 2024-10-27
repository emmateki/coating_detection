import albumentations as A


def setup_augmentation(
    patch_size,
    crop=False,
    elastic=False,
    brightness_contrast=False,
    flip_horizontal=False,
    blur_sharp=False,
    gauss_noise=False,
    interpolation=2,
):
    transform_list = []
    if crop:
        transform_list.append(
            A.CropNonEmptyMaskIfExists(
                height=patch_size,
                width=patch_size,
                ignore_values=None,
                ignore_channels=None,
                p=1,
            )
        )
    if elastic:
        transform_list.append(
            A.ElasticTransform(
                p=0.5,
                alpha=10,
                sigma=120 * 0.1,
                alpha_affine=120 * 0.1,
                interpolation=interpolation,
            )
        )

    if brightness_contrast:
        transform_list.append(
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3
            )
        )

    if gauss_noise:
        transform_list.append(A.GaussNoise(var_limit=(0.001), p=0.3))

    if blur_sharp:
        transform_list.append(A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3))

    if flip_horizontal:
        transform_list.append(A.HorizontalFlip(p=0.3))

    return A.Compose(transform_list)
