import torch
import numpy as np
import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    ComposeTransforms,
    nnUNetTrainer,
)
from batchgenerators.utilities.file_and_folder_operations import join
import datetime
import albumentations as A
from .albumentationsTransform import AlbumentationsTransform
from typing import Tuple, Union, List, cast
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class CustomTrainerAlbumentations_Large(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.output_folder = join(
            self.output_folder,
            os.environ.get("model_name", ""),
            datetime.datetime.now().strftime("%d_ %H_ %M_ %f"),
        )
        self.num_epochs = 600

    @staticmethod
    def get_training_transforms(
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
            nnUNetTrainer.get_training_transforms(
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
                    transforms=[
                        A.OneOf(
                            [
                                A.RandomCrop(512, 512, p=1.0),
                                A.RandomCrop(512, 1024, p=1.0),
                                A.RandomCrop(1024, 512, p=1.0),
                                A.RandomCrop(1024, 1024, p=1.0),
                            ],
                            p=1.0,
                        ),
                        A.Resize(height=patch_size[0],
                                 width=patch_size[1], p=1.0),
                    ],
                    p=0.5,
                )
            ),
        )
        return basic
