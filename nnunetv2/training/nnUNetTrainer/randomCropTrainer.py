import numpy as np
from .customTrainer import CustomTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import ComposeTransforms
from .albumentationsTransform import AlbumentationsTransform
import albumentations as A
from typing import Tuple, Union, List, cast, override
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

RANDOM_CROP_SIZE = 512


class RandomCropTrainer(CustomTrainer):
    @override
    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        basic: ComposeTransforms = cast(
            ComposeTransforms,
            super().get_training_transforms(
                patch_size,
                rotation_for_DA,
                deep_supervision_scales,
                mirror_axes,
                do_dummy_2d_data_aug,
                use_mask_for_norm,
                is_cascaded,
                foreground_labels,
                regions,
                ignore_label,
            ),
        )
        basic.transforms.insert(
            2,
            AlbumentationsTransform(
                A.Compose(
                    transforms=[A.RandomCrop(1024, 1024)],
                    p=1,
                )
            ),
        )
        return basic

    @override
    def get_validation_transforms(
        self,
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        basic: ComposeTransforms = cast(
            ComposeTransforms,
            super().get_validation_transforms(
                deep_supervision_scales,
                is_cascaded,
                foreground_labels,
                regions,
                ignore_label,
            ),
        )
        basic.transforms.insert(
            0,
            AlbumentationsTransform(
                A.Compose(
                    transforms=[A.RandomCrop(RANDOM_CROP_SIZE, RANDOM_CROP_SIZE)],
                    p=1,
                )
            ),
        )
        return basic
